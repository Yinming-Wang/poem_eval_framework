"""风格控制生成与 LLM-as-a-Judge 的 Prompt 构建器。"""

from __future__ import annotations

from typing import Any, Dict, List


JUDGE_PROMPT_TEMPLATE = """
你是一名中文古诗词评价专家。现在需要评价一个诗词生成模型的输出质量。
请采用严格标准评分，不要因为文本看起来像诗就给高分。所有维度均使用 0 到 100 的百分制评分，分数应拉开差距，避免大量集中在同一档。

评分档位说明：
90-100：非常优秀，几乎没有明显问题，严格符合诗体，主题和风格都鲜明，语言凝练，有古诗意境。95 分以上应非常少见。
80-89：较好，有少量瑕疵，但整体符合要求。
70-79：基本合格，能看出主题或风格，但存在格式、用词、意境或表达问题。
60-69：勉强可用，只部分满足要求，主题/风格不够明显，或格式存在明显问题。
40-59：较差，问题较多，诗体、主题、风格或语言中有多个维度不合格。
0-39：无效或严重不合格，包括空输出、乱码、明显重复、现代说明文字、偏题、不是诗。

请根据以下五个维度进行 0 到 100 分评分：
1. 格式正确性：必须严格符合指定诗体。五言绝句应为 4 句且每句 5 个汉字；七言绝句应为 4 句且每句 7 个汉字。句数或字数明显错误时，format_score 不得高于 45 分；完全不成诗时不得高于 30 分。
2. 主题相关性：必须清楚围绕输入主题展开。只是出现一个相关字词但整体无关时，不得高于 65 分；完全偏题不得高于 40 分。
3. 语言流畅性：语句应自然、通顺、无乱码、无明显病句。出现解释性文字、提示语、英文、代码、口语化现代白话或不成句内容时应明显扣分。
4. 古诗风格：应具有古典诗词的意象、节奏和表达。若像现代口号、散文、说明文或简单拼接词语，style_score 不得高于 45 分。
5. 意境创造性：应有画面感、情绪层次和一定新意。平铺直叙、套话堆砌、意象重复或空泛表达不得高于 65 分。

严格扣分规则：
- 如果生成结果为空、乱码、包含大量非中文字符，所有维度应接近 0-20 分。
- 如果输出包含“好的、下面是、这首诗、解释如下”等非诗正文内容，format_score 和 style_score 均不得高于 45 分。
- 如果明显重复同一句、同一短语或大量机械重复，fluency_score、style_score 和 creativity_score 均不得高于 45 分。
- 如果没有体现指定风格要求，style_score 不得高于 65 分；完全看不出风格时不得高于 40 分。
- 如果格式不符合指定诗体，即使主题相关，也不能给高总分。
- 请充分利用 0-100 的分数区间：优秀、一般、较差的诗应有明显分差，不要只给 70、80、90 这类粗粒度分数。

输入信息：
任务类型：{task_type} | 主题词：{theme} | 风格要求：{style} | 指定诗体：{poem_type}

模型生成结果：
{generated_poem}

请务必只输出一个 JSON 对象，不要输出多余解释（即使有分析也请放在 JSON 之外或省略）。JSON 结构如下：
{{
  "format_score": 分数,
  "theme_score": 分数,
  "fluency_score": 分数,
  "style_score": 分数,
  "creativity_score": 分数,
  "total_score": 加权总分，百分制,
  "brief_comment": "简短评价",
  "failure_type": "none/format_error/topic_drift/repetition/modern_language/incoherence/empty_output"
}}
"""


def build_style_prompt(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    根据 style_control 测试样本构造标准大模型 messages 结构。

    要求模型以 theme 为主题，按照 style 风格，创作指定 poem_type。
    """
    theme = str(sample.get("theme", "")).strip()
    style = str(sample.get("style", "")).strip()
    poem_type = str(sample.get("poem_type", "")).strip() or "五言绝句"

    if not theme:
        raise ValueError("style_control 样本中的 theme 不能为空。")
    if not style:
        raise ValueError("style_control 样本中的 style 不能为空。")

    system_prompt = (
        "你是一名擅长中文古典诗词创作的助手。"
        "请严格遵循用户给定的主题、风格和诗体要求，只输出诗词正文。"
    )
    user_prompt = (
        f"请以“{theme}”为主题，按照“{style}”的风格，"
        f"创作一首【{poem_type}】。请直接输出诗词正文，不要解释。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_judge_prompt(sample: Dict[str, Any], generated_poem: str) -> List[Dict[str, str]]:
    """
    将测试样本与模型生成结果组装为 LLM-as-a-Judge 打分 prompt。

    返回 OpenAI-compatible messages 结构。
    """
    content = JUDGE_PROMPT_TEMPLATE.format(
        task_type=str(sample.get("task_type", "")),
        theme=str(sample.get("theme", "")),
        style=str(sample.get("style", "")),
        poem_type=str(sample.get("poem_type", "")),
        generated_poem=generated_poem or "",
    ).strip()

    return [
        {
            "role": "system",
            "content": "你是严格、保守的中文古诗词评测员。评分时宁可偏严，不要给人情分。",
        },
        {"role": "user", "content": content},
    ]
