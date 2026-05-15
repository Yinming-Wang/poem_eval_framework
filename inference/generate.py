"""Prompt 构造与统一生成接口。"""

from __future__ import annotations

import re
from typing import Any, Dict, List


DEFAULT_THEME_PROMPT_TEMPLATE = (
    "请围绕主题“{theme}”创作一首{poem_type}。\n"
    "要求：只输出诗词正文，不要标题、解释、赏析、序号或任何说明文字。\n"
    "格式要求：五言绝句为四句、每句五个汉字；七言绝句为四句、每句七个汉字。"
)


def build_prompt(sample: dict, config: dict) -> str:
    """根据样本和配置文件中的 prompt 模板构造模型输入。"""
    template = config.get("prompt", {}).get("template", DEFAULT_THEME_PROMPT_TEMPLATE)
    try:
        return template.format(**sample)
    except KeyError as exc:
        raise KeyError(f"Prompt 模板引用了样本中不存在的字段: {exc}") from exc


def build_theme_messages(sample: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, str]]:
    """为 theme_generation 构造更适合 Instruct 模型的 messages。"""
    user_prompt = build_prompt(sample, config)
    return [
        {
            "role": "system",
            "content": (
                "你是一名中文古典诗词创作助手。"
                "必须严格遵循用户指定诗体，只输出诗词正文，禁止输出解释、标题、赏析、注释。"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def messages_to_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """将 messages 转为模型输入 prompt，优先使用 tokenizer 的 chat template。"""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append("{0}: {1}".format(role, content))
    parts.append("assistant:")
    return "\n".join(parts)


def generate_poem(model, tokenizer, sample: dict, config: dict) -> str:
    """
    输入模型、tokenizer、测试样本和配置，返回生成诗词文本。

    该函数只负责生成和基础清洗，不做格式正确性判断。
    """
    if hasattr(model, "generate_poem"):
        return clean_generated_text(
            model.generate_poem(sample.get("theme", ""), sample.get("poem_type", ""))
        )

    raw_prompt = build_prompt(sample, config)
    model_prompt = messages_to_prompt(tokenizer, build_theme_messages(sample, config))
    decoded = _generate_with_huggingface(model, tokenizer, model_prompt, config)
    for prompt_part in (model_prompt, raw_prompt):
        if decoded.startswith(prompt_part):
            decoded = decoded[len(prompt_part):]
        else:
            decoded = decoded.replace(prompt_part, "", 1)
    return clean_generated_text(decoded)


def _generate_with_huggingface(model, tokenizer, prompt: str, config: dict) -> str:
    """使用 Hugging Face generate 接口生成文本。"""
    if tokenizer is None:
        raise ValueError("Hugging Face 生成需要 tokenizer，但当前 tokenizer 为空。")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("缺少 torch，无法执行 Hugging Face 模型生成。") from exc

    generation_config = config.get("generation", {})
    device = getattr(model, "eval_device", None)
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]

    pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_config.get("max_new_tokens", 128),
            temperature=generation_config.get("temperature", 0.8),
            top_p=generation_config.get("top_p", 0.95),
            top_k=generation_config.get("top_k", 50),
            repetition_penalty=generation_config.get("repetition_penalty", 1.1),
            do_sample=generation_config.get("do_sample", True),
            pad_token_id=pad_token_id,
        )

    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def clean_generated_text(text: str) -> str:
    """清洗生成结果中的空行、常见前缀、标题、说明和多余空白。"""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = _remove_common_assistant_prefix(cleaned)
    cleaned = _keep_poem_like_lines(cleaned)
    cleaned = re.sub(r"[ \t]+", "", cleaned)
    return cleaned.strip()


def clean_continuation_text(text: str, prefix: str = "") -> str:
    """
    专门清洗上句续写结果，只保留续写下句本身。

    该函数会移除常见前缀、原上句、引号，并只返回第一条像诗句的中文内容。
    """
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = _remove_common_assistant_prefix(cleaned)
    cleaned = re.sub(r"^(下句|续写|答案|下一句|对句|承句)\s*[:：]\s*", "", cleaned)
    cleaned = cleaned.strip(" \t\n\r\"'“”‘’")

    if prefix:
        cleaned = cleaned.replace(prefix, "", 1).strip()

    candidates: List[str] = []
    for line in cleaned.split("\n"):
        line = line.strip().strip("\"'“”‘’")
        if not line:
            continue
        line = re.split(r"(解释|赏析|说明|注释|这句|这首诗)", line, maxsplit=1)[0].strip()
        if not line:
            continue
        if any(word in line for word in ("解释", "赏析", "这句", "这首诗", "上句")):
            break
        line = re.sub(r"^(下句|续写|答案|下一句|对句|承句)\s*[:：]\s*", "", line).strip()
        parts = re.split(r"[。！？!?；;]", line)
        for part in parts:
            part = part.strip(" ，,、\"'“”‘’")
            if re.search(r"[\u4e00-\u9fff]", part):
                candidates.append(part)
        if candidates:
            break

    if not candidates:
        return clean_generated_text(cleaned).split("\n")[0].strip() if cleaned else ""

    result = candidates[0]
    return re.sub(r"[ \t]+", "", result).strip()


def _remove_common_assistant_prefix(text: str) -> str:
    """移除“好的/下面是/标题”等常见非诗正文前缀。"""
    cleaned = text.strip()
    prefix_patterns = [
        r"^好的[，,。！!\s]*",
        r"^当然[，,。！!\s]*",
        r"^以下是.*?[:：]\s*",
        r"^下面是.*?[:：]\s*",
        r"^为你创作.*?[:：]\s*",
        r"^这是一首.*?[:：]\s*",
        r"^[《〈].*?[》〉]\s*",
        r"^(诗词|诗歌|古诗|作品|答案|输出|标题)\s*[:：]\s*",
    ]
    for pattern in prefix_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _keep_poem_like_lines(text: str) -> str:
    """
    尽量只保留诗句。

    如果模型输出解释、赏析、注释等内容，会优先保留含中文且不含明显说明词的前几行。
    """
    raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not raw_lines:
        return ""

    poem_lines: List[str] = []
    stop_words = ("解释", "赏析", "注释", "说明", "这首诗", "诗中", "表达了", "描绘了", "体现了")
    skip_patterns = [
        r"^\d+[\.、)]",
        r"^[-*]\s*",
        r"^主题[:：]",
        r"^题目[:：]",
    ]

    for line in raw_lines:
        if any(word in line for word in stop_words):
            break
        if any(re.search(pattern, line) for pattern in skip_patterns):
            continue
        if not re.search(r"[\u4e00-\u9fff]", line):
            continue
        poem_lines.append(line)
        if len(poem_lines) >= 4:
            break

    if poem_lines:
        return "\n".join(poem_lines)

    return "\n".join(raw_lines[:4])
