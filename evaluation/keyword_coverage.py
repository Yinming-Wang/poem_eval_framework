"""关键词约束生成任务的纯规则评价引擎。"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List


CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def calculate_repetition_rate(
    generated_text: str,
    fragment_size: int = 2,
) -> Dict[str, Any]:
    """
    计算生成文本的重复率。

    重复率 = 重复片段数量 / 总片段数量。
    这里的“片段”指中文字符连续 n-gram，默认 n=2。
    “重复片段数量”按除首次出现外的重复次数计算，例如某片段出现 3 次，
    则贡献 2 个重复片段。
    """
    if fragment_size <= 0:
        raise ValueError("fragment_size 必须大于 0。")

    if generated_text is None:
        generated_text = ""

    chinese_text = "".join(CHINESE_CHAR_PATTERN.findall(generated_text))
    if len(chinese_text) < fragment_size:
        return {
            "repetition_rate": 0.0,
            "repeated_fragment_count": 0,
            "total_fragment_count": 0,
            "repeated_fragments": {},
            "fragment_size": fragment_size,
        }

    fragments = [
        chinese_text[index:index + fragment_size]
        for index in range(len(chinese_text) - fragment_size + 1)
    ]
    fragment_counter = Counter(fragments)
    repeated_fragments = {
        fragment: count
        for fragment, count in fragment_counter.items()
        if count > 1
    }
    repeated_fragment_count = sum(count - 1 for count in repeated_fragments.values())
    total_fragment_count = len(fragments)
    repetition_rate = repeated_fragment_count / total_fragment_count

    return {
        "repetition_rate": repetition_rate,
        "repeated_fragment_count": repeated_fragment_count,
        "total_fragment_count": total_fragment_count,
        "repeated_fragments": repeated_fragments,
        "fragment_size": fragment_size,
    }


def evaluate_keyword_coverage(
    generated_text: str,
    keywords: List[str],
    include_repetition_rate: bool = False,
    repetition_fragment_size: int = 2,
) -> Dict[str, Any]:
    """
    使用 Python 字符串匹配计算关键词覆盖率，绝不调用大模型。

    Args:
        generated_text: 模型生成的诗词正文。
        keywords: 输入要求包含的关键词列表。
        include_repetition_rate: 是否额外计算重复率。
        repetition_fragment_size: 重复率统计使用的中文连续片段长度。

    Returns:
        包含命中数量、关键词总数、命中率和缺失关键词列表的字典。
        当 include_repetition_rate=True 时，会额外返回重复率相关字段。
    """
    if generated_text is None:
        generated_text = ""

    hit_count = 0
    missed_keywords: List[str] = []

    for keyword in keywords:
        keyword_text = str(keyword)
        # 关键词覆盖率采用严格子串匹配：关键词必须原样出现在生成文本中。
        if keyword_text in generated_text:
            hit_count += 1
        else:
            missed_keywords.append(keyword_text)

    total_count = len(keywords)
    hit_rate = hit_count / total_count if total_count > 0 else 0.0

    result: Dict[str, Any] = {
        "hit_count": hit_count,
        "total_count": total_count,
        "hit_rate": hit_rate,
        "missed_keywords": missed_keywords,
    }

    if include_repetition_rate:
        result.update(
            calculate_repetition_rate(
                generated_text=generated_text,
                fragment_size=repetition_fragment_size,
            )
        )

    return result


if __name__ == "__main__":
    sample_keywords = ["明月", "故乡", "清风"]

    success_text = "床前明月光，疑是地上霜。举头望明月，清风满故乡。"
    failure_text = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    repetitive_text = "明月明月照故乡，清风清风入故乡。"

    test_results = {
        "success_case": {
            "generated_text": success_text,
            "expected_hit_rate": 1.0,
            "evaluation": evaluate_keyword_coverage(
                success_text,
                sample_keywords,
                include_repetition_rate=True,
            ),
        },
        "failure_case": {
            "generated_text": failure_text,
            "expected_hit_rate": 2 / 3,
            "evaluation": evaluate_keyword_coverage(
                failure_text,
                sample_keywords,
                include_repetition_rate=True,
            ),
        },
        "repetition_case": {
            "generated_text": repetitive_text,
            "evaluation": evaluate_keyword_coverage(
                repetitive_text,
                sample_keywords,
                include_repetition_rate=True,
            ),
        },
    }

    print(json.dumps(test_results, ensure_ascii=False, indent=2))
