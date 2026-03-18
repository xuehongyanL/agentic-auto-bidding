"""
AuctionNet Think Model 实现

基于 Thinking LLM 的出价模型，负责 prompt 构造和 response 解析。
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from agb_core.model.think_model import ThinkModel


class AuctionNetThinkModel(ThinkModel):
    """
    AuctionNet 出价模型（Think 模式）

    继承 ThinkModel，负责：
    - prompt 构造（基于 numeral 业务 context）
    - LLM response 解析为 action
    - context 结构假设和处理
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = 'vllm',
        device: str = 'cuda',
        temperature: float = 0.0,
        max_tokens: int = 1024,
        state_dim: int = 16,
        action_dim: int = 1,
        verbose: int = 0,
    ):
        """
        初始化 AuctionNet Think Model

        Args:
            model_path: 本地模型路径
            model_type: 模型加载方式，可选 'vllm', 'transformers'
            device: 设备
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            state_dim: 状态维度
            action_dim: 动作维度
            verbose: 是否打印 prompt，0 不打印，1 完整打印

        Note:
            window_size 由 strategy 传入，通过 numeral 中的 context_dict 获取
        """
        # 占位符属性，与 DTStrategy 兼容
        self._target_rtg = 0.0
        self._scale = 1.0
        self._state_mean = None
        self._state_std = None

        super().__init__(
            model_path=model_path,
            model_type=model_type,
            device=device,
            temperature=temperature,
            max_tokens=max_tokens,
            state_dim=state_dim,
            action_dim=action_dim,
            verbose=verbose,
        )

    def predict(
        self,
        prompt: Optional[str],
        numeral: Optional[Any] = None
    ) -> Tuple[Optional[str], Optional[Any]]:
        """
        根据上下文预测 pacer

        Args:
            prompt: 忽略此参数（保留接口兼容性），prompt 在内部构造
            numeral: 二元组 (context_dict, dt_input)
                - 第一个元素：原始 dict（来自 base_strategy 的 _build_context）
                - 第二个元素：DT 多元组 (states, actions, rtgs, timesteps, attention_mask)

        Returns:
            (response, action): response 是 LLM 的文本响应，action 是方向值（-1/0/1）
        """
        # 调用父类获取 response
        response, _ = super().predict(prompt, numeral)

        # 解析 response 得到 action
        action = self._parse_response(response)

        return response, action

    def _build_prompt(self, numeral: Any) -> str:
        """
        根据 numeral（二元组）构建 prompt

        Args:
            numeral: 二元组 (context_dict, dt_input)

        Returns:
            user_prompt 字符串（由 _call_llm 自动应用 chat template）
        """
        # 解包二元组
        context_dict, dt_input = numeral

        return self._format_user_prompt(context_dict)

    def _format_user_prompt(self, context_dict: Dict[str, Any]) -> str:
        """根据 context_dict 构建用户 prompt"""
        window_size = context_dict.get('window_size', 20)
        num_timesteps = context_dict.get('num_timesteps', 48)

        # 获取历史列表
        history_pacer = context_dict.get('history_pacer', [])
        history_pv_num = context_dict.get('history_pv_num', [])
        history_conversion = context_dict.get('history_conversion', [])
        history_total_cost = context_dict.get('history_total_cost', [])
        budget = context_dict.get('budget', 0)

        # 根据 window_size 切片
        num_history = len(history_pacer)
        actual_window = min(num_history, window_size)
        start_idx = max(0, num_history - actual_window)

        # 切片历史数据
        pacer_list = [float(p.flatten()[0]) for p in history_pacer[start_idx:num_history]]
        pv_num_list = history_pv_num[start_idx:num_history]
        conversion_list = history_conversion[start_idx:num_history]
        total_cost_list = history_total_cost[start_idx:num_history]

        # 计算 time_left_list: (num_timesteps - step) / num_timesteps
        # step 索引从 start_idx 到 num_history - 1
        time_left_list = [(num_timesteps - step) / num_timesteps for step in range(start_idx, num_history)]

        # 计算 budget_left_list: (budget - total_cost) / budget
        budget_left_list = []
        for tc in total_cost_list:
            if budget > 0:
                bl = max(0, (budget - tc) / budget)
            else:
                bl = 0
            budget_left_list.append(bl)

        # 计算 pvalue_list: pv_num（每个 step 的曝光数）
        pvalue_list = pv_num_list

        # 格式化列表为字符串
        time_left_str = '[' + ', '.join([f'{t:.3f}' for t in time_left_list]) + ']' if time_left_list else '[]'
        budget_left_str = '[' + ', '.join([f'{b:.3f}' for b in budget_left_list]) + ']' if budget_left_list else '[]'
        pvalue_str = '[' + ', '.join([f'{p:.0f}' for p in pvalue_list]) + ']' if pvalue_list else '[]'
        conversion_str = '[' + ', '.join([f'{c:.0f}' for c in conversion_list]) + ']' if conversion_list else '[]'
        pacer_str = '[' + ', '.join([f'{p:.3f}' for p in pacer_list]) + ']' if pacer_list else '[]'

        return self._USER_PROMPT.format(
            budget=context_dict.get('budget', 0),
            cpa_constraint=context_dict.get('cpa_constraint', 0),
            num_steps=actual_window,
            start_step=start_idx,
            end_step=num_history - 1,
            time_left_list=time_left_str,
            budget_left_list=budget_left_str,
            pvalue_list=pvalue_str,
            conversion_list=conversion_str,
            total_conversions=context_dict.get('total_conversions', 0),
            bid_list=pacer_str,
        )

    def _parse_response(self, response: str) -> np.ndarray:
        """
        解析 LLM 响应，提取方向值

        LLM 输出 -1/0/1：
        - -1 → 降低出价
        - 0 → 保持
        - 1 → 提高出价

        Args:
            response: LLM 的原始响应

        Returns:
            direction: 一维 numpy array，包含 -1, 0, 或 1
        """
        import re

        response = response.strip()

        # 尝试从 <answer> 标签中提取 -1/0/1
        answer_match = re.search(r'<answer>\s*(-?1|0)\s*</answer>', response)
        if answer_match:
            return np.array([int(answer_match.group(1))])

        # 如果没有找到 answer 标签，尝试直接匹配 -1/0/1
        direction_match = re.search(r'\b(-1|0|1)\b', response)
        if direction_match:
            return np.array([int(direction_match.group(1))])

        # 默认返回 0
        return np.array([0])

    def _get_system_prompt(self) -> str:
        """获取 system prompt"""
        return self._SYSTEM_PROMPT

    _SYSTEM_PROMPT = '''
You are an auto-bidding agent determining the optimal bidding parameter for the advertiser. There are 48 timesteps of a day, the aim is to maximize the total acquired number of conversions with a lower realized CPA.
'''
    _USER_PROMPT = '''
The advertiser's budget is {budget:.0f} with a CPA constraint {cpa_constraint:.2f}. Its historical {num_steps} timesteps' performance change along with time (i.e., from timestep {start_step} to timestep {end_step}) are:
- timesteps remaining ratio: {time_left_list}
- budget remaining ratio: {budget_left_list}
- predicted total impression value: {pvalue_list}
- achieved conversions of each step: {conversion_list}
From the 0-th timestep to now, the total acquired conversions is {total_conversions:.0f}.
For each historical timestep, their corresponding bidding parameter is: {bid_list}.

You should summarize the history and then reason for the best future adjustment direction.
Here are some basic knowledge:
1. You should carefully and sufficiently spend your budget but do not spend all the budget too early;
2. The realized CPA is calculated as (total spent budget / total acquired number of conversions), where total spent budget = budget * (1 - current budget remaining ratio). If you think the realized CPA would be bigger than the CPA constraint when spent all the budget, you should decrease the bidding parameter.
As a part of summarization, you need to output the latest timestep's cpa ratio = realized_cpa/cpa_constraint in <ratio></ratio> tags.

After your summarization and reasoning, you MUST output the direction in <answer></answer> tags with only three choices at the end of your response:
- <answer>1</answer> indicates you are sure increasing the bidding parameter is better
- <answer>-1</answer> indicates you are sure decreasing the parameter is better
- <answer>0</answer> indicates you are uncertain about the optimal adjustment direction.
'''
