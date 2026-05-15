"""关键词约束生成任务的 Prompt 构建器。"""

from __future__ import annotations

from typing import Dict, List


def build_keyword_prompt(sample: Dict) -> List[Dict[str, str]]:
    """
    根据 keyword_generation 测试样本构造标准大模型对话结构。

    返回格式兼容常见 Chat Completions 风格接口：
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
    """
    poem_type = str(sample.get("poem_type", "")).strip() or "五言绝句"
    keywords = sample.get("keywords", [])
    if not isinstance(keywords, list):
        raise TypeError("sample['keywords'] 必须是 List[str] 类型。")

    keyword_text = "、".join(str(keyword).strip() for keyword in keywords if str(keyword).strip())
    if not keyword_text:
        raise ValueError("关键词列表不能为空。")

    system_prompt = (
        "你是一名擅长创作中文古典诗词的助手。"
        "请严格遵循用户给定的诗体、主题和关键词约束，只输出诗词正文。"
    )
    user_prompt = (
        f"请创作一首{poem_type}。\n"
        f"要求：诗句中必须包含以下所有词汇：{keyword_text}。\n"
        "注意：每个关键词都必须原样出现在诗词正文中，不要遗漏，不要解释。"
    )

    theme = str(sample.get("theme", "")).strip()
    if theme:
        user_prompt = f"请围绕主题“{theme}”创作一首{poem_type}。\n" + user_prompt.split("\n", 1)[1]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
