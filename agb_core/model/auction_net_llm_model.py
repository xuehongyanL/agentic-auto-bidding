"""
AuctionNet LLM Model 实现

基于 Thinking LLM 的出价模型，负责 prompt 构造和 response 解析。
"""

from typing import Any, Dict, Tuple

from agb_core.model.llm_model import LLMModel


class AuctionNetLLMModel(LLMModel):
    """
    AuctionNet 出价模型

    继承 LLMModel，负责：
    - prompt 构造（基于业务 context）
    - LLM response 解析
    - context 结构假设和处理
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = 'vllm',
        device: str = 'cuda',
        temperature: float = 0.0,
        max_tokens: int = 1024,
        window_size: int = 20,
    ):
        """
        初始化 AuctionNet LLM Model

        Args:
            model_path: 本地模型路径
            model_type: 模型加载方式，可选 'vllm', 'transformers', 'llamacpp'
            device: 设备
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            window_size: 历史窗口大小
        """
        self._window_size = window_size

        # 占位符属性，与 DTStrategy 兼容
        self._target_return = 0.0
        self._scale = 1.0
        self._state_mean = None
        self._state_std = None

        super().__init__(
            model_path=model_path,
            model_type=model_type,
            device=device,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def predict(self, context) -> float:
        """
        根据上下文预测 pacer

        Args:
            context: 二元组 (原始 dict, DT 多元组)
                - 第一个元素：原始 dict（来自 base_strategy 的 _build_context）
                - 第二个元素：DT 多元组 (states, actions, rewards, curr_score, timesteps, attention_mask)

        Returns:
            pacer: 出价系数
        """
        # 解包二元组
        context_dict, dt_input = context

        prompt = self._build_prompt(context_dict, dt_input)
        response = super().predict(prompt)
        pacer = self._parse_response(response)
        return pacer

    def _build_prompt(self, context_dict: Dict[str, Any], dt_input: Tuple) -> str:
        """
        根据上下文构建中文 prompt

        Args:
            context_dict: 原始上下文字典，包含历史列表
            dt_input: DT 多元组 (states, actions, rewards, curr_score, timesteps, attention_mask)

        Returns:
            构建好的 prompt 字符串
        """
        system_prompt = self._SYSTEM_PROMPT
        user_prompt = self._format_user_prompt(context_dict)

        return f'<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n'

    def _format_user_prompt(self, context_dict: Dict[str, Any]) -> str:
        """根据 context_dict 构建用户 prompt"""
        window_size = self._window_size
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
        pacer_list = history_pacer[start_idx:num_history]
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

    def _format_dt_input(self, dt_input: Tuple) -> str:
        """
        将 DT 多元组格式化为字符串

        Args:
            dt_input: DT 多元组 (states, actions, rewards, curr_score, timesteps, attention_mask)

        Returns:
            格式化的 DT 上下文字符串
        """
        states, actions, rewards, curr_score, timesteps, attention_mask = dt_input
        lines = []
        lines.append(f'- states shape: {states.shape}')
        lines.append(f'- actions shape: {actions.shape}')
        lines.append(f'- rewards shape: {rewards.shape}')
        lines.append(f'- curr_score: {curr_score[-1][0] if len(curr_score) > 0 else 0:.4f}')
        lines.append(f'- timesteps: {list(timesteps)}')
        lines.append(f'- attention_mask: {list(attention_mask)}')
        return '\n'.join(lines)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        将上下文字典格式化为字符串

        Args:
            context: 决策上下文字典

        Returns:
            格式化的上下文字符串
        """
        lines = []
        for key, value in context.items():
            if isinstance(value, float):
                lines.append(f'- {key}: {value:.4f}')
            else:
                lines.append(f'- {key}: {value}')
        return '\n'.join(lines)

    def _parse_response(self, response: str) -> float:
        """
        解析 LLM 响应，提取 pacer 值

        LLM 输出 -1/0/1，对应：
        - -1 → 0.8 (降低出价)
        - 0 → 1.0 (保持)
        - 1 → 1.2 (提高出价)

        Args:
            response: LLM 的原始响应

        Returns:
            pacer: 出价系数
        """
        import re

        response = response.strip()

        # 尝试从 <answer> 标签中提取 -1/0/1
        answer_match = re.search(r'<answer>\s*(-?1|0)\s*</answer>', response)
        if answer_match:
            direction = int(answer_match.group(1))
            if direction == -1:
                return 0.8
            elif direction == 0:
                return 1.0
            elif direction == 1:
                return 1.2

        # 如果没有找到 answer 标签，尝试直接匹配 -1/0/1
        direction_match = re.search(r'\b(-1|0|1)\b', response)
        if direction_match:
            direction = int(direction_match.group(1))
            if direction == -1:
                return 0.8
            elif direction == 0:
                return 1.0
            elif direction == 1:
                return 1.2

        # 默认返回 1.0
        return 1.0

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
    _USER_PROMPT_EXAMPLE = '''
The advertiser's budget is 2850 with a CPA constraint 8. Its historical 4 timesteps' performance change along with time (i.e., from timestep 31 to timestep 34) are:
 timesteps remaining ratio: [0.354, 0.333, 0.312, 0.292],
 budget remaining ratio: [0.953, 0.953, 0.953, 0.953],
 predicted total impression value: [28, 54, 46, 44],
 achieved conversions of each step: [0, 0, 0, 0].
From the 0-th timestep to now, the total acquired conversions is 25.
For each historical timestep, their corresponding bidding parameter is: [6.845, 6.967, 7.063, 7.238].

You should summarize the history and then reason for the best future adjustment direction.
Here are some basic knowledge:
1. You should carefully and sufficiently spend your budget but do not spend all the budget too early;
2. The realized CPA is calculated as (total spent budget / total acquired number of conversions), where total spent budget = budget * (1 - current budget remaining ratio). If you think the realized CPA would be bigger than the CPA constraint when spent all the budget, you should decrease the bidding parameter.
As a part of summarization, you need to output the latest timestep's cpa ratio = realized_cpa/cpa_constraint in <ratio></ratio> tags.

After your summarization and reasoning, you MUST output the direction in <answer></answer> tags with only three choices at the end of your response:
- <answer>1</answer>indicates you are sure increasing the bidding parameter is better
- <answer>-1</answer> indicates you are sure decreasing the parameter is better
- <answer>0</answer> indicates you are uncertain about the optimal adjustment direction.
'''
