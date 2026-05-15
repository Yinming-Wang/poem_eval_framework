"""大模型中文古诗词生成评测系统 WebUI。

运行方式：
    python webui.py

说明：
    1. WebUI 会扫描 models/ 文件夹中的子目录，并在界面中提供本地模型选择。
    2. 选择模型会覆盖 configs/model_config.yaml 中的 model_path/tokenizer_path/model_name。
    3. 为控制 API 成本，裁判大模型默认仅在“风格控制”任务中调用；其它任务只展示规则指标。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.bleu_rouge import evaluate_bleu_rouge
from evaluation.format_check import check_format
from evaluation.keyword_coverage import evaluate_keyword_coverage
from evaluation.llm_judge import call_judge_api, parse_judge_result
from inference.continuation_prompt import build_continuation_prompt
from inference.generate import clean_continuation_text, clean_generated_text, generate_poem
from inference.keyword_prompt import build_keyword_prompt
from inference.model_loader import load_model_and_tokenizer
from inference.style_prompt import build_judge_prompt, build_style_prompt
from evaluation.run_theme_eval import evaluate_theme_generation
from evaluation.run_keyword_eval import evaluate_keyword_generation
from evaluation.run_prefix_eval import build_prefix_continuation_samples, evaluate_prefix_continuation
from evaluation.run_style_eval import (
    evaluate_style_control,
    resolve_judge_config as resolve_style_judge_config,
    validate_judge_config,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
POEM_TYPES = ["五言绝句", "七言绝句"]

_LOCAL_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def load_config() -> Dict[str, Any]:
    """读取统一配置文件。"""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("配置文件不存在：{0}".format(CONFIG_PATH))
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("configs/model_config.yaml 内容必须是 YAML 字典。")
    return config


def list_available_models() -> List[str]:
    """扫描 models/ 文件夹，返回可选本地模型目录名称。"""
    if not MODELS_DIR.exists():
        return []
    model_names = [
        path.name
        for path in MODELS_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    return sorted(model_names)


def get_default_model_selection() -> str:
    """根据配置文件和 models/ 目录推断默认选中的模型。"""
    model_names = list_available_models()
    config = load_config()
    configured_path = Path(str(config.get("model", {}).get("model_path", "") or ""))
    configured_name = configured_path.name
    if configured_name in model_names:
        return configured_name
    return model_names[0] if model_names else ""


def refresh_model_choices() -> Any:
    """刷新模型下拉框选项。"""
    model_names = list_available_models()
    selected_model = get_default_model_selection()
    return gr.update(choices=model_names, value=selected_model)


def build_config_for_selected_model(selected_model: str) -> Dict[str, Any]:
    """用 UI 选择的本地模型覆盖配置文件中的模型路径。"""
    config = load_config()
    model_config = dict(config.get("model", {}))

    selected_model = str(selected_model or "").strip()
    if selected_model:
        model_path = MODELS_DIR / selected_model
        if not model_path.exists():
            raise FileNotFoundError("选择的模型目录不存在：{0}".format(model_path))
        model_config.update(
            {
                "model_name": selected_model,
                "model_type": "huggingface",
                "model_path": str(model_path),
                "tokenizer_path": str(model_path),
            }
        )

    config["model"] = model_config
    return config


def get_local_model(selected_model: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """按需加载并缓存所选本地待测模型，避免每次点击都重新加载权重。"""
    config_mtime = CONFIG_PATH.stat().st_mtime
    config = build_config_for_selected_model(selected_model)
    model_config = config.get("model", {})
    cache_key = "{0}|{1}|{2}".format(
        model_config.get("model_name", ""),
        model_config.get("model_path", ""),
        config_mtime,
    )

    if cache_key not in _LOCAL_MODEL_CACHE:
        model, tokenizer = load_model_and_tokenizer(config)
        _LOCAL_MODEL_CACHE[cache_key] = {
            "config": config,
            "model": model,
            "tokenizer": tokenizer,
        }
    cached = _LOCAL_MODEL_CACHE[cache_key]
    return cached["model"], cached["tokenizer"], cached["config"]


def split_keywords(keyword_text: str) -> List[str]:
    """将用户输入的关键词文本切分为关键词列表。"""
    separators = [",", "，", "、", "\n", ";", "；", " "]
    normalized_text = keyword_text or ""
    for separator in separators:
        normalized_text = normalized_text.replace(separator, ",")
    return [item.strip() for item in normalized_text.split(",") if item.strip()]


def normalize_api_url(api_url: str) -> str:
    """将 API 根地址自动补齐为 chat/completions 接口地址。"""
    normalized_url = str(api_url or "").strip().rstrip("/")
    if not normalized_url:
        return ""
    if normalized_url.endswith("/chat/completions") or normalized_url.endswith("/completions"):
        return normalized_url
    return "{0}/chat/completions".format(normalized_url)


def messages_to_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """将 messages 转为本地聊天模型可用 prompt。"""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts = ["{0}: {1}".format(item.get("role", "user"), item.get("content", "")) for item in messages]
    parts.append("assistant:")
    return "\n".join(parts)


def generate_huggingface_text(model: Any, tokenizer: Any, prompt: str, config: Dict[str, Any]) -> str:
    """调用本地 Hugging Face 模型生成文本，只解码新增 token。"""
    if tokenizer is None:
        raise ValueError("Hugging Face 生成需要 tokenizer。")

    import torch

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
    return clean_generated_text(tokenizer.decode(generated_ids, skip_special_tokens=True))


def call_openai_compatible_chat(
    messages: List[Dict[str, str]],
    api_url: str,
    api_key: str,
    model_name: str = "local-poem-model",
) -> str:
    """调用 OpenAI-compatible 待测模型 API。"""
    try:
        import requests
    except ImportError as exc:
        raise ImportError("缺少 requests，无法调用待测模型 API。请运行：pip install requests") from exc

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer {0}".format(api_key)

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    response = requests.post(normalize_api_url(api_url), headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    response_data = response.json()
    choices = response_data.get("choices", [])
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        if isinstance(message, dict) and message.get("content") is not None:
            return clean_generated_text(str(message["content"]))
        if choices[0].get("text") is not None:
            return clean_generated_text(str(choices[0]["text"]))
    return clean_generated_text(json.dumps(response_data, ensure_ascii=False))


def generate_with_test_model(
    task_type: str,
    sample: Dict[str, Any],
    selected_model: str,
) -> str:
    """根据 UI 选择调用真实本地待测模型生成诗词。"""
    if task_type == "theme_generation":
        model, tokenizer, config = get_local_model(selected_model)
        return generate_poem(model, tokenizer, sample, config)

    if task_type == "keyword_generation":
        messages = build_keyword_prompt(sample)
    elif task_type == "prefix_continuation":
        messages = build_continuation_prompt(sample)
    elif task_type == "style_control":
        messages = build_style_prompt(sample)
    else:
        raise ValueError("不支持的任务类型：{0}".format(task_type))

    model, tokenizer, config = get_local_model(selected_model)
    prompt = messages_to_prompt(tokenizer, messages)
    if hasattr(model, "generate_poem") and task_type != "prefix_continuation":
        return clean_generated_text(model.generate_poem(sample.get("theme", ""), sample.get("poem_type", "")))
    generated_text = generate_huggingface_text(model, tokenizer, prompt, config)
    if task_type == "prefix_continuation":
        return clean_continuation_text(generated_text, prefix=sample.get("prefix", ""))
    return generated_text


def resolve_judge_config(judge_url: str, judge_api_key: str) -> Dict[str, str]:
    """读取裁判模型配置，UI 输入优先，配置文件兜底。"""
    config = load_config()
    judge_config = config.get("judge", {})
    if not isinstance(judge_config, dict):
        judge_config = {}

    resolved_url = str(judge_url or "").strip() or str(judge_config.get("api_url", "") or "")
    resolved_key = str(judge_api_key or "").strip() or str(judge_config.get("api_key", "") or "")
    resolved_model = str(judge_config.get("model_name", "deepseek-chat") or "deepseek-chat")

    if resolved_key in ("YOUR_API_KEY", "your_api_key", "在这里填写你的真实 API Key"):
        resolved_key = ""

    return {
        "api_url": normalize_api_url(resolved_url),
        "api_key": resolved_key,
        "model_name": resolved_model,
    }


def evaluate_with_judge_if_needed(
    sample: Dict[str, Any],
    generated_poem: str,
    judge_url: str,
    judge_api_key: str,
) -> Tuple[Dict[str, Any], str]:
    """仅风格控制任务调用裁判大模型，其它任务返回跳过说明。"""
    if sample.get("task_type") != "style_control":
        return {
            "skipped": True,
            "reason": "为控制 API 成本，裁判大模型当前仅用于风格控制任务。",
        }, "非风格控制任务未调用裁判大模型；请查看规则指标。"

    judge_config = resolve_judge_config(judge_url, judge_api_key)
    if not judge_config["api_url"] or not judge_config["api_key"]:
        return {
            "skipped": True,
            "reason": "未配置裁判模型 URL 或 API Key，已跳过 LLM-as-a-Judge。",
        }, "未配置裁判模型 API，无法生成大模型评语。"

    raw_response = call_judge_api(
        messages=build_judge_prompt(sample, generated_poem),
        api_url=judge_config["api_url"],
        api_key=judge_config["api_key"],
        model_name=judge_config["model_name"],
    )
    judge_result = parse_judge_result(raw_response)
    comment = str(judge_result.get("brief_comment", "") or "")
    return judge_result, comment


def build_rule_metrics(task_type: str, sample: Dict[str, Any], generated_poem: str, reference: str = "") -> Dict[str, Any]:
    """按任务类型计算规则指标。"""
    format_result = check_format(generated_poem, sample.get("poem_type", ""))

    if task_type == "theme_generation":
        return {"format_check": format_result}

    if task_type == "keyword_generation":
        keyword_result = evaluate_keyword_coverage(
            generated_text=generated_poem,
            keywords=sample.get("keywords", []),
            include_repetition_rate=True,
        )
        return {
            "format_check": format_result,
            "keyword_coverage": keyword_result,
        }

    if task_type == "prefix_continuation":
        references = [reference.strip()] if reference and reference.strip() else []
        if references:
            overlap_scores = evaluate_bleu_rouge(generated_poem, references)
        else:
            overlap_scores = {"warning": "未填写参考答案，已跳过 BLEU/ROUGE。"}
        return {
            "selected_prefix": sample.get("prefix", ""),
            "selected_reference": references[0] if references else "",
            "overlap_metrics": overlap_scores,
        }

    if task_type == "style_control":
        return {"format_check": format_result}

    return {}


def yes_no(value: bool) -> str:
    """将布尔值转为适合界面展示的中文结果。"""
    return "通过" if value else "未通过"


def percent(value: Any) -> str:
    """将 0-1 或 0-100 数值格式化为百分比。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if number <= 1:
        number *= 100
    return "{0:.2f}%".format(number)


def format_char_counts(char_counts: List[int]) -> str:
    """格式化每句字数。"""
    if not char_counts:
        return "-"
    return " / ".join(str(item) for item in char_counts)


def format_errors(errors: List[str]) -> str:
    """格式化错误列表。"""
    if not errors:
        return "无"
    return "；".join(str(item) for item in errors)


def format_metrics_markdown(
    task_type: str,
    rule_metrics: Dict[str, Any],
    judge_scores: Dict[str, Any],
) -> str:
    """将内部评测结果整理成一目了然的 Markdown 表格。"""
    format_result = rule_metrics.get("format_check", {})
    lines = ["### 规则指标", "", "| 指标 | 结果 |", "|---|---:|"]

    if task_type != "prefix_continuation":
        lines.extend(
            [
                "| 格式检查 | {0} |".format(yes_no(bool(format_result.get("is_format_correct")))),
                "| 诗体 | {0} |".format(format_result.get("expected_type", "-")),
                "| 句数 | {0} |".format(format_result.get("num_sentences", "-")),
                "| 每句字数 | {0} |".format(format_char_counts(format_result.get("char_counts", []))),
                "| 格式错误 | {0} |".format(format_errors(format_result.get("errors", []))),
            ]
        )

    if task_type == "keyword_generation":
        keyword_result = rule_metrics.get("keyword_coverage", {})
        missed_keywords = keyword_result.get("missed_keywords", [])
        lines.extend(
            [
                "| 关键词命中 | {0}/{1} |".format(
                    keyword_result.get("hit_count", 0),
                    keyword_result.get("total_count", 0),
                ),
                "| 关键词命中率 | {0} |".format(percent(keyword_result.get("hit_rate", 0))),
                "| 缺失关键词 | {0} |".format("、".join(missed_keywords) if missed_keywords else "无"),
                "| 重复率 | {0} |".format(percent(keyword_result.get("repetition_rate", 0))),
            ]
        )

    if task_type == "prefix_continuation":
        overlap_metrics = rule_metrics.get("overlap_metrics", {})
        lines.extend(
            [
                "| 随机上句 | {0} |".format(rule_metrics.get("selected_prefix", "-")),
                "| 参考答案 | {0} |".format(rule_metrics.get("selected_reference", "-")),
            ]
        )
        if "warning" in overlap_metrics:
            lines.append("| BLEU/ROUGE | {0} |".format(overlap_metrics["warning"]))
        else:
            lines.extend(
                [
                    "| BLEU-4 | {0} |".format(percent(overlap_metrics.get("BLEU-4", 0))),
                    "| ROUGE-L | {0} |".format(percent(overlap_metrics.get("ROUGE-L", 0))),
                ]
            )

    if task_type == "style_control" and judge_scores and not judge_scores.get("skipped"):
        lines.extend(
            [
                "",
                "### 裁判大模型百分制评分",
                "",
                "| 维度 | 分数 |",
                "|---|---:|",
                "| 格式正确性 | {0:.2f} |".format(float(judge_scores.get("format_score", 0))),
                "| 主题相关性 | {0:.2f} |".format(float(judge_scores.get("theme_score", 0))),
                "| 语言流畅性 | {0:.2f} |".format(float(judge_scores.get("fluency_score", 0))),
                "| 古诗风格 | {0:.2f} |".format(float(judge_scores.get("style_score", 0))),
                "| 意境创造性 | {0:.2f} |".format(float(judge_scores.get("creativity_score", 0))),
                "| 综合得分 | **{0:.2f}** |".format(float(judge_scores.get("total_score", 0))),
            ]
        )
    elif task_type == "style_control":
        lines.extend(
            [
                "",
                "### 裁判大模型百分制评分",
                "",
                "未调用裁判大模型。请检查裁判模型 URL 和 API Key 是否已填写。",
            ]
        )

    return "\n".join(lines)


def real_evaluate_pipeline(
    task_type: str,
    selected_model: str,
    judge_url: str,
    judge_api_key: str,
    theme: str = "",
    keywords: str = "",
    prefix: str = "",
    reference: str = "",
    style: str = "",
    poem_type: str = "五言绝句",
) -> Tuple[str, str, str]:
    """真实评测流程：调用待测模型、规则评价，并按需调用裁判模型。"""
    keyword_list = split_keywords(keywords)
    sample = {
        "id": "WEB001",
        "task_type": task_type,
        "theme": theme.strip(),
        "keywords": keyword_list,
        "prefix": prefix.strip(),
        "style": style.strip(),
        "poem_type": poem_type,
        "references": [reference.strip()] if reference.strip() else [],
    }

    generated_poem = generate_with_test_model(task_type, sample, selected_model)
    rule_metrics = build_rule_metrics(task_type, sample, generated_poem, reference=reference)
    judge_scores, brief_comment = evaluate_with_judge_if_needed(
        sample=sample,
        generated_poem=generated_poem,
        judge_url=judge_url,
        judge_api_key=judge_api_key,
    )

    metrics_markdown = format_metrics_markdown(task_type, rule_metrics, judge_scores)
    return generated_poem, metrics_markdown, brief_comment


def format_batch_metrics(task_name: str, results_path: Path, metrics_path: Path, metrics: Dict[str, Any]) -> str:
    """将批量评测输出整理成简明 Markdown。"""
    lines = [
        "### {0} 批量评测完成".format(task_name),
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        "| 结果文件 | `{0}` |".format(results_path.relative_to(PROJECT_ROOT)),
        "| 指标文件 | `{0}` |".format(metrics_path.relative_to(PROJECT_ROOT)),
    ]

    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            continue
        display_key = str(key)
        if isinstance(value, float):
            display_value = "{0:.4f}".format(value)
        else:
            display_value = str(value)
        lines.append("| {0} | {1} |".format(display_key, display_value))

    return "\n".join(lines)


def run_batch_eval(task_type: str, selected_model: str, max_samples: int) -> str:
    """从 WebUI 直接调用 evaluation/run_*_eval.py 对应的本地批量评测函数。"""
    config = build_config_for_selected_model(selected_model)
    output_dir = "outputs"
    max_samples = int(max_samples or 0)

    if task_type == "theme_generation":
        results_path, metrics_path, metrics = evaluate_theme_generation(
            config=config,
            output_dir=output_dir,
            max_samples=max_samples,
        )
        return format_batch_metrics("主题词生成", results_path, metrics_path, metrics)

    if task_type == "keyword_generation":
        results_path, metrics_path, metrics = evaluate_keyword_generation(
            config=config,
            output_dir=output_dir,
            max_samples=max_samples,
        )
        return format_batch_metrics("关键词约束", results_path, metrics_path, metrics)

    if task_type == "prefix_continuation":
        results_path, metrics_path, metrics = evaluate_prefix_continuation(
            config=config,
            output_dir=output_dir,
            max_samples=max_samples,
            samples_file="",
            sample_source="builtin",
            hf_dataset="MatrixStudio/ChinesePoetry",
            hf_split="train",
            hf_scan_limit=0,
            shuffle=True,
            seed=random.randint(1, 999999),
        )
        return format_batch_metrics("上句续写", results_path, metrics_path, metrics)

    if task_type == "style_control":
        args = SimpleNamespace(judge_url="", judge_key="", judge_model="", require_judge_api=False)
        judge_config = resolve_style_judge_config(config, args)
        validate_judge_config(judge_config)
        results_path, metrics_path, metrics = evaluate_style_control(
            config=config,
            output_dir=output_dir,
            max_samples=max_samples,
            judge_url=judge_config["api_url"],
            judge_key=judge_config["api_key"],
            judge_model=judge_config["model_name"],
            require_judge_api=judge_config["require_api"],
        )
        return format_batch_metrics("风格控制", results_path, metrics_path, metrics)

    raise ValueError("不支持的批量评测任务：{0}".format(task_type))


def safe_run_batch(task_type: str, selected_model: str, max_samples: int) -> str:
    """包装批量评测，避免异常直接打断前端。"""
    try:
        return run_batch_eval(task_type, selected_model, max_samples)
    except Exception as exc:
        return "### 批量评测失败\n\n{0}".format(exc)


def safe_run(fn: Any, *args: Any) -> Tuple[str, str, str]:
    """包装 UI 任务，避免异常直接打断前端。"""
    try:
        return fn(*args)
    except Exception as exc:
        error_message = "运行失败：{0}".format(exc)
        return "", "### 运行失败\n\n{0}".format(error_message), error_message


def set_button_loading() -> Any:
    """点击后禁用按钮，并显示处理中状态。"""
    return gr.update(value="生成与评测中...", interactive=False)


def reset_button() -> Any:
    """任务结束后恢复按钮状态。"""
    return gr.update(value="🚀 开始测试", interactive=True)


def run_theme_task(
    selected_model: str,
    judge_url: str,
    judge_api_key: str,
    theme: str,
    poem_type: str,
) -> Tuple[str, str, str]:
    """主题词生成任务入口。"""
    return real_evaluate_pipeline(
        task_type="theme_generation",
        selected_model=selected_model,
        judge_url=judge_url,
        judge_api_key=judge_api_key,
        theme=theme,
        poem_type=poem_type,
    )


def run_keyword_task(
    selected_model: str,
    judge_url: str,
    judge_api_key: str,
    keywords: str,
    poem_type: str,
) -> Tuple[str, str, str]:
    """关键词约束生成任务入口。"""
    return real_evaluate_pipeline(
        task_type="keyword_generation",
        selected_model=selected_model,
        judge_url=judge_url,
        judge_api_key=judge_api_key,
        keywords=keywords,
        poem_type=poem_type,
    )


def run_prefix_task(
    selected_model: str,
    judge_url: str,
    judge_api_key: str,
) -> Tuple[str, str, str]:
    """上句续写任务入口：从内置样本池随机抽取上句和参考答案。"""
    sample = random.choice(build_prefix_continuation_samples())
    selected_reference = sample.get("references", [""])[0]
    return real_evaluate_pipeline(
        task_type="prefix_continuation",
        selected_model=selected_model,
        judge_url=judge_url,
        judge_api_key=judge_api_key,
        prefix=sample["prefix"],
        reference=selected_reference,
        poem_type=sample.get("poem_type", ""),
    )


def run_style_task(
    selected_model: str,
    judge_url: str,
    judge_api_key: str,
    theme: str,
    style: str,
    poem_type: str,
) -> Tuple[str, str, str]:
    """风格控制任务入口。"""
    return real_evaluate_pipeline(
        task_type="style_control",
        selected_model=selected_model,
        judge_url=judge_url,
        judge_api_key=judge_api_key,
        theme=theme,
        style=style,
        poem_type=poem_type,
    )


def bind_task_button(button: gr.Button, task_fn: Any, inputs: List[Any], outputs: List[Any]) -> None:
    """为每个任务按钮绑定统一的加载态和恢复态。"""
    button.click(fn=set_button_loading, inputs=None, outputs=button).then(
        fn=lambda *args: safe_run(task_fn, *args),
        inputs=inputs,
        outputs=outputs,
        show_progress="full",
    ).then(fn=reset_button, inputs=None, outputs=button)


def build_demo() -> gr.Blocks:
    """构建 Gradio Blocks 应用。"""
    css = """
    .app-title { text-align: center; margin-bottom: 0.5rem; }
    .app-subtitle { text-align: center; color: #666; margin-top: -0.5rem; margin-bottom: 1rem; }
    """

    with gr.Blocks(title="中文古诗词生成评测系统", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown(
            """
            <div class="app-title"><h1>📜 大模型中文古诗词生成评测系统</h1></div>
            <div class="app-subtitle">主题词生成 · 关键词约束 · 上句续写 · 风格控制</div>
            """
        )

        with gr.Accordion("模型选择", open=True):
            gr.Markdown("从 `models/` 文件夹中选择本地待测模型。新模型放入 `models/模型目录名/` 后，点击刷新即可出现在下拉框中。")
            with gr.Row():
                model_selector = gr.Dropdown(
                    label="待测模型",
                    choices=list_available_models(),
                    value=get_default_model_selection(),
                    interactive=True,
                )
                refresh_model_button = gr.Button("刷新模型列表")
                judge_url = gr.Textbox(label="裁判模型 URL", value="https://api.deepseek.com/chat/completions")
                judge_api_key = gr.Textbox(label="裁判模型 API Key", type="password", placeholder="风格控制任务使用")
            refresh_model_button.click(fn=refresh_model_choices, inputs=None, outputs=model_selector)

        with gr.Accordion("批量本地评测", open=False):
            gr.Markdown(
                "这里直接调用本地 `evaluation/run_theme_eval.py`、`run_keyword_eval.py`、"
                "`run_prefix_eval.py`、`run_style_eval.py` 对应的评测函数，输出仍保存到 `outputs/`。"
            )
            with gr.Row():
                batch_max_samples = gr.Number(label="批量样本数", value=10, precision=0)
                batch_theme_button = gr.Button("运行主题词生成")
                batch_keyword_button = gr.Button("运行关键词约束")
                batch_prefix_button = gr.Button("运行上句续写")
                batch_style_button = gr.Button("运行风格控制")
            batch_output = gr.Markdown(label="批量评测摘要")

            batch_theme_button.click(
                fn=lambda selected_model, max_samples: safe_run_batch("theme_generation", selected_model, max_samples),
                inputs=[model_selector, batch_max_samples],
                outputs=batch_output,
                show_progress="full",
            )
            batch_keyword_button.click(
                fn=lambda selected_model, max_samples: safe_run_batch("keyword_generation", selected_model, max_samples),
                inputs=[model_selector, batch_max_samples],
                outputs=batch_output,
                show_progress="full",
            )
            batch_prefix_button.click(
                fn=lambda selected_model, max_samples: safe_run_batch("prefix_continuation", selected_model, max_samples),
                inputs=[model_selector, batch_max_samples],
                outputs=batch_output,
                show_progress="full",
            )
            batch_style_button.click(
                fn=lambda selected_model, max_samples: safe_run_batch("style_control", selected_model, max_samples),
                inputs=[model_selector, batch_max_samples],
                outputs=batch_output,
                show_progress="full",
            )

        with gr.Tabs():
            with gr.Tab("主题词生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        theme_input = gr.Textbox(label="主题词", value="月夜", placeholder="例如：春天、边塞、思乡")
                        theme_poem_type = gr.Dropdown(label="诗体", choices=POEM_TYPES, value="五言绝句")
                        theme_button = gr.Button("🚀 开始测试", variant="primary")
                    with gr.Column(scale=2):
                        theme_poem_output = gr.Textbox(label="待测模型生成结果", lines=4)
                        theme_metrics_output = gr.Markdown(label="规则指标与评测结果")
                        theme_comment_output = gr.Textbox(label="评语 / 状态", lines=3)

                bind_task_button(
                    theme_button,
                    run_theme_task,
                    [model_selector, judge_url, judge_api_key, theme_input, theme_poem_type],
                    [theme_poem_output, theme_metrics_output, theme_comment_output],
                )

            with gr.Tab("关键词约束"):
                with gr.Row():
                    with gr.Column(scale=1):
                        keyword_input = gr.Textbox(
                            label="关键词",
                            value="明月，故乡，清风",
                            placeholder="多个关键词用逗号、顿号或换行分隔",
                            lines=4,
                        )
                        keyword_poem_type = gr.Dropdown(label="诗体", choices=POEM_TYPES, value="五言绝句")
                        keyword_button = gr.Button("🚀 开始测试", variant="primary")
                    with gr.Column(scale=2):
                        keyword_poem_output = gr.Textbox(label="待测模型生成结果", lines=4)
                        keyword_metrics_output = gr.Markdown(label="规则指标与评测结果")
                        keyword_comment_output = gr.Textbox(label="评语 / 状态", lines=3)

                bind_task_button(
                    keyword_button,
                    run_keyword_task,
                    [model_selector, judge_url, judge_api_key, keyword_input, keyword_poem_type],
                    [keyword_poem_output, keyword_metrics_output, keyword_comment_output],
                )

            with gr.Tab("上句续写"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "点击开始测试时，系统会从内置上句-答案样本池中随机抽取一条，"
                            "无需手动填写上句和参考答案。抽中的样本会显示在右侧指标表中。"
                        )
                        prefix_button = gr.Button("🚀 开始测试", variant="primary")
                    with gr.Column(scale=2):
                        prefix_poem_output = gr.Textbox(label="待测模型生成结果", lines=4)
                        prefix_metrics_output = gr.Markdown(label="规则指标与评测结果")
                        prefix_comment_output = gr.Textbox(label="评语 / 状态", lines=3)

                bind_task_button(
                    prefix_button,
                    run_prefix_task,
                    [model_selector, judge_url, judge_api_key],
                    [prefix_poem_output, prefix_metrics_output, prefix_comment_output],
                )

            with gr.Tab("风格控制"):
                with gr.Row():
                    with gr.Column(scale=1):
                        style_theme_input = gr.Textbox(label="主题词", value="月夜", placeholder="例如：江南、边塞、田园")
                        style_input = gr.Textbox(label="风格要求", value="思乡", placeholder="例如：豪迈、婉约、清新")
                        style_poem_type = gr.Dropdown(label="诗体", choices=POEM_TYPES, value="五言绝句")
                        style_button = gr.Button("🚀 开始测试", variant="primary")
                    with gr.Column(scale=2):
                        style_poem_output = gr.Textbox(label="待测模型生成结果", lines=4)
                        style_metrics_output = gr.Markdown(label="规则指标与 LLM-as-a-Judge 百分制评分")
                        style_comment_output = gr.Textbox(label="裁判模型简短评语", lines=3)

                bind_task_button(
                    style_button,
                    run_style_task,
                    [model_selector, judge_url, judge_api_key, style_theme_input, style_input, style_poem_type],
                    [style_poem_output, style_metrics_output, style_comment_output],
                )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue()
    app.launch()
