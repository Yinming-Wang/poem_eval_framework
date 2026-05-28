"""Prompt 构造与统一生成接口。"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List

try:
    import torch
except ImportError:  # pragma: no cover - 仅类型/可选依赖
    torch = None  # type: ignore[assignment]

_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

# 评测前截取：绝句常见一行连写为「句1，句2。句3，句4。」共 4×N 个汉字 + 4 个标点。
_JUEJU_EVAL_MAX_CHARS = {
    "五言绝句": 4 * 5 + 4,
    "七言绝句": 4 * 7 + 4,
}


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
            apply_fn = tokenizer.apply_chat_template
            kwargs: Dict[str, Any] = {
                "messages": messages,
                "tokenize": False,
                "add_generation_prompt": True,
            }
            # Qwen3：不设 enable_thinking=False 时会进入思考链，解码后易出现 <think>...</think> 块。
            try:
                params = inspect.signature(apply_fn).parameters
                if "enable_thinking" in params:
                    kwargs["enable_thinking"] = False
            except (TypeError, ValueError):
                pass
            return apply_fn(**kwargs)
        except TypeError:
            # 不支持 enable_thinking 等额外参数时再试一次仅基础参数
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        except Exception:
            pass

    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append("{0}: {1}".format(role, content))
    parts.append("assistant:")
    return "\n".join(parts)


def generate_huggingface_text(model: Any, tokenizer: Any, prompt: str, config: Dict[str, Any]) -> str:
    """
    使用 Hugging Face generate 生成文本，仅解码新增 token。

    GLM-4 + LoRA（run_eval 对齐路径）需禁用 KV cache 并使用 GenerationConfig，
    通过 model._poem_eval_legacy_glm_generate 标记启用。
    """
    if tokenizer is None:
        raise ValueError("Hugging Face 生成需要 tokenizer，但当前 tokenizer 为空。")

    if torch is None:
        raise ImportError("缺少 torch，无法执行 Hugging Face 模型生成。")

    generation_dict = config.get("generation", {})

    eval_dev = getattr(model, "eval_device", None)
    if eval_dev is not None:
        device = eval_dev
    else:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]

    pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id

    legacy_glm = bool(getattr(model, "_poem_eval_legacy_glm_generate", False))

    with torch.no_grad():
        if legacy_glm:
            from inference.glm_compat import build_glm_generation_config

            gen_cfg = build_glm_generation_config(tokenizer, generation_dict)
            outputs = model.generate(
                **inputs,
                generation_config=gen_cfg,
                use_cache=False,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_dict.get("max_new_tokens", 128),
                temperature=generation_dict.get("temperature", 0.8),
                top_p=generation_dict.get("top_p", 0.95),
                top_k=generation_dict.get("top_k", 50),
                repetition_penalty=generation_dict.get("repetition_penalty", 1.1),
                do_sample=generation_dict.get("do_sample", True),
                pad_token_id=pad_token_id,
            )

    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_from_chat_messages(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    config: Dict[str, Any],
    *,
    poem_type: str = "",
    continuation_prefix: str = "",
) -> str:
    """
    从 Chat messages 生成文本：自动区分 LSTM 与 HuggingFace。

    continuation_prefix 非空时（上句续写），对输出做续写专用清洗。
    """
    if getattr(model, "model_type", None) == "lstm":
        from inference.lstm_backend import messages_to_lstm_prompt

        prompt = messages_to_lstm_prompt(messages)
        decoded = model.generate_chars(prompt, config)
        if continuation_prefix:
            return clean_continuation_text(decoded, prefix=continuation_prefix)
        return clean_generated_text(decoded, poem_type=poem_type)

    prompt = messages_to_prompt(tokenizer, messages)
    decoded = generate_huggingface_text(model, tokenizer, prompt, config)
    if continuation_prefix:
        return clean_continuation_text(decoded, prefix=continuation_prefix)
    return clean_generated_text(decoded, poem_type=poem_type)


def generate_poem(model, tokenizer, sample: dict, config: dict) -> str:
    """
    输入模型、tokenizer、测试样本和配置，返回生成诗词文本。

    该函数只负责生成和基础清洗，不做格式正确性判断。
    """
    poem_type = str(sample.get("poem_type") or "")
    if hasattr(model, "generate_poem"):
        return clean_generated_text(
            model.generate_poem(sample.get("theme", ""), poem_type),
            poem_type=poem_type,
        )

    if getattr(model, "model_type", None) == "lstm":
        from inference.lstm_backend import SYSTEM_PROMPT, build_conversation_prompt

        instruction = build_prompt(sample, config)
        lstm_prompt = build_conversation_prompt(SYSTEM_PROMPT, instruction)
        decoded = model.generate_chars(lstm_prompt, config)
        return clean_generated_text(decoded, poem_type=poem_type)

    raw_prompt = build_prompt(sample, config)
    model_prompt = messages_to_prompt(tokenizer, build_theme_messages(sample, config))
    decoded = _generate_with_huggingface(model, tokenizer, model_prompt, config)
    for prompt_part in (model_prompt, raw_prompt):
        if decoded.startswith(prompt_part):
            decoded = decoded[len(prompt_part):]
        else:
            decoded = decoded.replace(prompt_part, "", 1)
    return clean_generated_text(decoded, poem_type=poem_type)


def _generate_with_huggingface(model, tokenizer, prompt: str, config: dict) -> str:
    """使用 Hugging Face generate 接口生成文本。"""
    return generate_huggingface_text(model, tokenizer, prompt, config)


def truncate_jueju_for_evaluation(text: str, poem_type: str) -> str:
    """
    绝句评测前截断：去掉空白后只保留前若干字符，避免模型后续赏析/对话进入指标。

    五言绝句：4×5 汉字 + 4 个标点 = 24；七言绝句：4×7 + 4 = 32。
    """
    limit = _JUEJU_EVAL_MAX_CHARS.get(str(poem_type or "").strip())
    if limit is None or not text:
        return text
    compact = re.sub(r"\s+", "", text.strip())
    if len(compact) <= limit:
        return compact
    return compact[:limit]


def _infer_next_line_hanzi_from_prefix(prefix: str) -> int | None:
    """由上句汉字数推断下一句应为五言(5)或七言(7)，否则不截断。"""
    count = len(_CHINESE_CHAR_RE.findall(prefix or ""))
    if count == 5:
        return 5
    if count == 7:
        return 7
    return None


def _truncate_to_n_hanzi_cut_tail(text: str, n: int) -> str:
    """
    仅保留开头至多 n 个汉字；若第 n 个汉字后紧跟句末标点（。！？），一并保留后结束。
    """
    if n <= 0 or not text:
        return text
    out: List[str] = []
    hz = 0
    for ch in text:
        if _CHINESE_CHAR_RE.match(ch):
            if hz >= n:
                break
            hz += 1
            out.append(ch)
        else:
            if hz == n:
                if ch in "。！？":
                    out.append(ch)
                break
            if 0 < hz < n and ch in "，、":
                out.append(ch)
    return "".join(out).strip()


def _strip_embedded_reasoning_blocks(text: str) -> str:
    """
    移除 Qwen3 等模型的「推理/思考」外层标记（解码后常为 XML 片段）。

    即使 prompt 侧已禁用 thinking，部分权重或解码仍可能残留此类块。
    """
    if not text:
        return ""

    # Qwen3：解码后为带 redacted_ 前缀的推理块标记（常见为 thinking）。
    stripped = text
    for suf in ("thinking", "think"):
        open_tag = "<" + "redacted_" + suf + ">"
        close_tag = "</" + "redacted_" + suf + ">"
        stripped = re.sub(
            re.escape(open_tag) + r".*?" + re.escape(close_tag),
            "",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
    return stripped.strip()


def clean_generated_text(text: str, poem_type: str | None = None) -> str:
    """清洗生成结果中的空行、常见前缀、标题、说明和多余空白。"""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = _strip_embedded_reasoning_blocks(cleaned)
    cleaned = _remove_common_assistant_prefix(cleaned)
    cleaned = _keep_poem_like_lines(cleaned)
    cleaned = re.sub(r"[ \t]+", "", cleaned)
    cleaned = cleaned.strip()
    return truncate_jueju_for_evaluation(cleaned, str(poem_type or "").strip())


def clean_continuation_text(text: str, prefix: str = "") -> str:
    """
    专门清洗上句续写结果，只保留续写下句本身。

    该函数会移除常见前缀、原上句、引号，并只返回第一条像诗句的中文内容。
    """
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = _strip_embedded_reasoning_blocks(cleaned)
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
        line = re.sub(r"^(下句|续写|答案|下一句|对句|承句)\s*[:：]\s*", "", line).strip()
        if not line:
            continue
        # 截掉赏析/说明等后续文字，保留首个诗句片段（含句末标点）
        line = re.split(
            r"(解释|赏析|说明|注释|这首诗|上句[:：]?|这句(?:表达了|说明|是))",
            line,
            maxsplit=1,
        )[0].strip()
        if not line:
            continue
        if not re.search(r"[\u4e00-\u9fff]", line):
            continue
        candidates.append(line)
        break

    if not candidates:
        base = clean_generated_text(cleaned, poem_type=None)
        base = truncate_jueju_for_evaluation(base, "")
        return base.split("\n")[0].strip() if base else ""

    result = candidates[0]
    result = re.sub(r"[ \t]+", "", result).strip()
    width = _infer_next_line_hanzi_from_prefix(prefix)
    if width is not None:
        result = _truncate_to_n_hanzi_cut_tail(result, width)
    return result


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
