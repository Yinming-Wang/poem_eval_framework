"""上句续写任务的 Prompt 构建器。"""

from __future__ import annotations

from typing import Any, Dict, List


def build_continuation_prompt(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    根据 prefix_continuation 测试样本构造标准大模型 messages 结构。

    上句续写只要求模型补下一句，不再提示“五言绝句/七言绝句”，避免模型误解为补全整首诗。
    """
    prefix = str(sample.get("prefix", "")).strip()

    if not prefix:
        raise ValueError("prefix_continuation 样本中的 prefix 不能为空。")

    system_prompt = (
        "你是一名擅长中文古典诗词续写的助手。"
        "请严格根据用户给定的上句续写下一句。"
        "你的输出必须只有续写的下一句本身，禁止输出标题、解释、原句、序号、引号或任何说明文字。"
    )
    user_prompt = (
        f"请根据上句“{prefix}”，续写下一句。"
        "硬性要求：\n"
        "1. 只输出一行。\n"
        "2. 只输出续写下句本身。\n"
        "3. 不要包含原有上句。\n"
        "4. 不要补全整首诗。\n"
        "5. 不要输出“下句：”“续写：”“答案：”“解释：”等任何前缀。\n"
        "6. 不要使用引号包裹。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
