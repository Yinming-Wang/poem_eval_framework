"""上句续写任务的 BLEU 与 ROUGE 评价引擎。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def clean_to_char_tokens(text: str) -> List[str]:
    """
    将中文诗句清洗为字粒度 token 列表。

    古诗评价不做现代汉语分词，只保留中文汉字并逐字切分。
    标点、空格、数字和英文字母会被剔除。
    """
    if text is None:
        text = ""
    return CHINESE_CHAR_PATTERN.findall(text)


def clean_to_spaced_chars(text: str) -> str:
    """将中文诗句转为以空格分隔的单字字符串，供 ROUGE 使用。"""
    return " ".join(clean_to_char_tokens(text))


def calculate_rouge_l(
    hypothesis_tokens: List[str],
    reference_tokens: List[str],
) -> float:
    """
    基于字粒度 token 计算 ROUGE-L F1。

    这里直接使用最长公共子序列（LCS）实现，避免 rouge-score 默认 tokenizer
    对中文字符支持不稳定导致完全相同文本也得到 0 分。
    """
    if not hypothesis_tokens or not reference_tokens:
        return 0.0

    lcs_length = _lcs_length(hypothesis_tokens, reference_tokens)
    if lcs_length == 0:
        return 0.0

    precision = lcs_length / len(hypothesis_tokens)
    recall = lcs_length / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)


def _lcs_length(left: List[str], right: List[str]) -> int:
    """计算两个 token 序列的最长公共子序列长度。"""
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def evaluate_bleu_rouge(
    generated_text: str,
    references: List[str],
    include_percentage: bool = True,
) -> Dict[str, float]:
    """
    计算生成文本与参考答案之间的 BLEU-4 和 ROUGE-L。

    依赖：
        pip install nltk rouge-score

    Args:
        generated_text: 模型生成的续写诗句。
        references: 参考答案列表。
        include_percentage: 是否额外返回 0-100 的百分制分数。

    Returns:
        默认返回 BLEU-4、ROUGE-L 及其百分制分数。
    """
    if not references:
        raise ValueError("references 不能为空，BLEU/ROUGE 评价至少需要一个参考答案。")

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError as exc:
        raise ImportError(
            "缺少 nltk，无法计算 BLEU。请运行：pip install nltk"
        ) from exc

    hypothesis_tokens = clean_to_char_tokens(generated_text)
    reference_tokens_list = [clean_to_char_tokens(reference) for reference in references]
    reference_tokens_list = [tokens for tokens in reference_tokens_list if tokens]

    if not hypothesis_tokens or not reference_tokens_list:
        return _format_scores(bleu_4=0.0, rouge_l=0.0, include_percentage=include_percentage)

    smoothing = SmoothingFunction().method1
    bleu_4 = sentence_bleu(
        reference_tokens_list,
        hypothesis_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    rouge_l = 0.0
    for reference_tokens in reference_tokens_list:
        rouge_l = max(rouge_l, calculate_rouge_l(hypothesis_tokens, reference_tokens))

    return _format_scores(
        bleu_4=float(bleu_4),
        rouge_l=float(rouge_l),
        include_percentage=include_percentage,
    )


def _format_scores(
    bleu_4: float,
    rouge_l: float,
    include_percentage: bool,
) -> Dict[str, float]:
    """整理指标输出，同时提供 0-1 原始分和 0-100 百分比分。"""
    scores = {
        "BLEU-4": bleu_4,
        "ROUGE-L": rouge_l,
    }
    if include_percentage:
        scores.update(
            {
                "BLEU-4-%": round(bleu_4 * 100, 2),
                "ROUGE-L-%": round(rouge_l * 100, 2),
            }
        )
    return scores


def _build_stub_result(
    generated_text: str,
    references: List[str],
) -> Dict[str, Any]:
    """构造测试桩输出，展示清洗后的字粒度字符串和指标分数。"""
    return {
        "generated_text": generated_text,
        "generated_char_level": clean_to_spaced_chars(generated_text),
        "references": references,
        "references_char_level": [clean_to_spaced_chars(item) for item in references],
        "scores": evaluate_bleu_rouge(generated_text, references),
    }


if __name__ == "__main__":
    sample_references = ["低头思故乡。"]
    ideal_output = "低头思故乡"
    partial_output = "低头望故乡"
    creative_output = "影落大江流"

    try:
        test_results = {
            "ideal_case": _build_stub_result(ideal_output, sample_references),
            "partial_case": _build_stub_result(partial_output, sample_references),
            "creative_case": _build_stub_result(creative_output, sample_references),
        }
    except ImportError as exc:
        print(str(exc))
        print("安装 BLEU 评价依赖可运行：pip install nltk")
    else:
        print(json.dumps(test_results, ensure_ascii=False, indent=2))
