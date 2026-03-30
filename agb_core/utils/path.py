from pathlib import Path


def _expand_range_in_brace(pattern: str) -> str:
    """将 brace 内的数字范围 {a..b} 展开为枚举形式 {a,b,...,b}。

    wcmatch 支持 {a,b,c} 枚举，不支持 {a..b} 范围语法。
    例如: 'part_{0..3}_end' -> 'part_{0,1,2,3}_end'
          'a_{0..2,5}_b'    -> 'a_{0,1,2,5}_b'
    """
    import re

    def replace_brace_content(brace_content: str) -> str:
        segments = brace_content.split(',')
        expanded = []
        for seg in segments:
            seg = seg.strip()
            if '..' in seg:
                try:
                    start, end = seg.split('..')
                    start, end = int(start), int(end)
                    expanded.append(','.join(str(i) for i in range(start, end + 1)))
                except ValueError:
                    expanded.append(seg)
            else:
                expanded.append(seg)
        return ','.join(expanded)

    result = re.sub(r'\{([^}]+)\}', lambda m: '{' + replace_brace_content(m.group(1)) + '}', pattern)
    return result


def glob_data_paths(data_path: str) -> list[Path]:
    """解析 data_path，支持 glob brace expansion 模式，返回匹配的文件路径列表。

    模式示例：
      - part_{0..3}_thought.pkl   -> 匹配 part_0 ~ part_3
      - part_{0..10,15..20}_thought.pkl  -> 匹配 0~10 和 15~20
      - part_{0..100}_thought.pkl  -> 匹配 0~100
      - single_file.pkl           -> 精确路径
    """
    from wcmatch import glob as wcglob

    p = Path(data_path)
    if p.exists() and p.is_file():
        return [p]

    data_dir = p.parent if p.is_absolute() else Path(data_path).parent
    pattern = p.name

    if '{' not in pattern and '*' not in pattern:
        raise ValueError(
            f'data_path \'{data_path}\' is not an existing file and contains no glob pattern. '
            f'Use a pattern like \'part_{{0..3}}_thought.pkl\''
        )

    # 将 {a..b} 范围展开为 {a,b,...,b}，wcmatch 只支持枚举形式
    pattern = _expand_range_in_brace(pattern)

    pattern_paths = wcglob.glob(pattern, root_dir=data_dir, flags=wcglob.BRACE | wcglob.GLOBSTAR)
    if not pattern_paths:
        raise FileNotFoundError(
            f'No files matched pattern \'{pattern}\' in directory \'{data_dir}\''
        )
    return sorted(Path(data_dir, f) for f in pattern_paths)
