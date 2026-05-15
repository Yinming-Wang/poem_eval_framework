"""诗词格式正确率检查。"""

from __future__ import annotations

import re
from collections import Counter


CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")
SENTENCE_SPLIT_PATTERN = re.compile(r"[，。！？；、,.!?;\n]+")


def extract_chinese_sentences(text: str) -> list[str]:
    """
    将生成文本按中文标点、英文标点或换行切分为句子。

    标点不会计入句子内容，空句会被过滤。
    """
    if not text:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]


def count_chinese_chars(sentence: str) -> int:
    """统计句子中的中文汉字数量。"""
    return len(CHINESE_CHAR_PATTERN.findall(sentence or ""))


def has_garbled_text(text: str) -> bool:
    """判断是否存在明显乱码或大量非中文字符。"""
    if not text or not text.strip():
        return True

    stripped = text.strip()
    chinese_count = count_chinese_chars(stripped)
    if chinese_count == 0:
        return True

    visible_chars = [ch for ch in stripped if not ch.isspace()]
    if not visible_chars:
        return True

    chinese_ratio = chinese_count / len(visible_chars)
    if chinese_ratio < 0.5:
        return True

    special_chars = re.findall(r"[^\u4e00-\u9fffA-Za-z0-9，。！？；、,.!?;\s]", stripped)
    special_ratio = len(special_chars) / len(visible_chars)
    return special_ratio > 0.2


def has_obvious_repetition(text: str) -> bool:
    """判断是否存在完全重复句、连续重复短语或单字异常重复。"""
    if not text:
        return False

    sentences = extract_chinese_sentences(text)
    sentence_counts = Counter(sentences)
    if any(sentence and count > 1 for sentence, count in sentence_counts.items()):
        return True

    chinese_text = "".join(CHINESE_CHAR_PATTERN.findall(text))
    if re.search(r"([\u4e00-\u9fff])\1{3,}", chinese_text):
        return True

    # 检测 2-4 字短语连续重复 3 次以上，如“春风春风春风”。
    for size in range(2, 5):
        if re.search(rf"([\u4e00-\u9fff]{{{size}}})\1{{2,}}", chinese_text):
            return True

    # 检测整体文本中短片段异常高频，避免非常机械的输出。
    fragments = [
        chinese_text[index:index + 4]
        for index in range(max(0, len(chinese_text) - 3))
    ]
    fragment_counts = Counter(fragments)
    return any(fragment and count >= 4 for fragment, count in fragment_counts.items())


def check_format(poem: str, poem_type: str) -> dict:
    """
    检查生成诗词是否符合指定诗体格式。

    返回值包含是否正确、句数、每句汉字数和错误列表。
    """
    errors: list[str] = []
    sentences = extract_chinese_sentences(poem)
    char_counts = [count_chinese_chars(sentence) for sentence in sentences]

    result = {
        "is_format_correct": False,
        "expected_type": poem_type,
        "num_sentences": len(sentences),
        "char_counts": char_counts,
        "errors": errors,
    }

    if not poem or not poem.strip():
        errors.append("空输出")
        return result

    if has_garbled_text(poem):
        errors.append("乱码或非中文字符过多")

    if has_obvious_repetition(poem):
        errors.append("明显重复")

    if poem_type == "五言绝句":
        _check_jueju(sentences, char_counts, expected_chars=5, errors=errors)
    elif poem_type == "七言绝句":
        _check_jueju(sentences, char_counts, expected_chars=7, errors=errors)
    elif poem_type == "自由诗":
        # 自由诗只检查基础可读性，不限制句数和每句字数。
        pass
    else:
        errors.append(f"不支持的诗体类型：{poem_type}")

    result["is_format_correct"] = len(errors) == 0
    return result


def _check_jueju(
    sentences: list[str],
    char_counts: list[int],
    expected_chars: int,
    errors: list[str],
) -> None:
    """检查绝句的句数和每句字数。"""
    expected_sentences = 4
    if len(sentences) != expected_sentences:
        errors.append(f"句数错误：期望 {expected_sentences} 句，实际 {len(sentences)} 句")

    for index, count in enumerate(char_counts):
        if count != expected_chars:
            errors.append(
                f"第 {index + 1} 句字数错误：期望 {expected_chars} 字，实际 {count} 字"
            )
