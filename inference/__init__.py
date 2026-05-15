"""模型加载与文本生成接口。"""

from .continuation_prompt import build_continuation_prompt
from .keyword_prompt import build_keyword_prompt
from .style_prompt import JUDGE_PROMPT_TEMPLATE, build_judge_prompt, build_style_prompt

__all__ = [
    "JUDGE_PROMPT_TEMPLATE",
    "build_continuation_prompt",
    "build_judge_prompt",
    "build_keyword_prompt",
    "build_style_prompt",
]
