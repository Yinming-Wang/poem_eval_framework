"""style_control 任务评测入口：生成诗词并调用 LLM-as-a-Judge 打分。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
from tqdm import tqdm

from evaluation.llm_judge import call_judge_api, parse_judge_result
from inference.generate import clean_generated_text
from inference.model_loader import load_model_and_tokenizer
from inference.style_prompt import build_judge_prompt, build_style_prompt


STYLE_CASES: List[Tuple[str, str, str]] = [
    ("月夜", "思乡", "五言绝句"),
    ("江南", "婉约", "七言绝句"),
    ("边塞", "豪迈", "七言绝句"),
    ("田园", "闲适", "五言绝句"),
    ("离别", "伤感", "七言绝句"),
    ("春雨", "清新", "五言绝句"),
    ("秋风", "萧瑟", "七言绝句"),
    ("冬雪", "孤寂", "五言绝句"),
    ("中秋", "团圆", "七言绝句"),
    ("登高", "壮阔", "七言绝句"),
    ("古寺", "禅意", "五言绝句"),
    ("长安", "怀古", "七言绝句"),
    ("山水", "空灵", "五言绝句"),
    ("渔舟", "淡泊", "七言绝句"),
    ("美酒", "旷达", "五言绝句"),
    ("故人", "怀念", "七言绝句"),
    ("梅雪", "高洁", "五言绝句"),
    ("楼台", "华丽", "七言绝句"),
    ("夜雨", "忧思", "五言绝句"),
    ("远方", "浪漫", "七言绝句"),
]

STYLE_VARIANTS: List[str] = [
    "含蓄",
    "清雅",
    "沉郁",
    "明快",
    "苍凉",
]

DEFAULT_STYLE_SAMPLE_COUNT = 100


def build_style_control_samples(sample_count: int = DEFAULT_STYLE_SAMPLE_COUNT) -> List[Dict[str, Any]]:
    """构造风格控制测试样本，默认生成 100 条。"""
    samples: List[Dict[str, Any]] = []
    if sample_count <= 0:
        sample_count = len(STYLE_CASES)

    for index in range(sample_count):
        theme, base_style, base_poem_type = STYLE_CASES[index % len(STYLE_CASES)]
        variant = STYLE_VARIANTS[(index // len(STYLE_CASES)) % len(STYLE_VARIANTS)]
        style = base_style if index < len(STYLE_CASES) else "{0}、{1}".format(base_style, variant)
        poem_type = "五言绝句" if index % 2 == 0 else "七言绝句"
        if index < len(STYLE_CASES):
            poem_type = base_poem_type

        samples.append(
            {
                "id": "S{0:03d}".format(index + 1),
                "task_type": "style_control",
                "theme": theme,
                "keywords": [],
                "prefix": "",
                "style": style,
                "poem_type": poem_type,
            }
        )
    return samples


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行风格控制生成与 LLM-as-a-Judge 评测。")
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="模型、生成与判分 API 配置文件路径。相对路径默认基于项目根目录解析。",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="评测输出目录。相对路径默认基于项目根目录解析。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_STYLE_SAMPLE_COUNT,
        help="最多评测多少条样本；默认 100。0 表示只运行基础内置样本。",
    )
    parser.add_argument("--judge_url", default="", help="判分 API URL；不传则读取配置文件 judge.api_url。")
    parser.add_argument("--judge_key", default="", help="判分 API Key；不传则读取配置文件 judge.api_key。")
    parser.add_argument("--judge_model", default="", help="判分模型名称；不传则读取配置文件 judge.model_name。")
    parser.add_argument(
        "--require_judge_api",
        action="store_true",
        help="启用后未提供有效 judge 配置就报错；也可在配置文件 judge.require_api 中开启。",
    )
    return parser.parse_args()


def resolve_project_path(path: str) -> Path:
    """将相对路径解析到项目根目录，绝对路径保持不变。"""
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    return PROJECT_ROOT / resolved_path


def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    path = resolve_project_path(config_path)
    if not path.exists():
        raise FileNotFoundError("配置文件不存在: {0}".format(path))
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件内容必须是 YAML 字典。")
    return config


def resolve_judge_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """合并配置文件和命令行中的 judge 参数，命令行非空值优先。"""
    judge_config = config.get("judge", {})
    if not isinstance(judge_config, dict):
        judge_config = {}

    api_url = args.judge_url or str(judge_config.get("api_url", "") or "")
    api_key = args.judge_key or str(judge_config.get("api_key", "") or "")
    model_name = args.judge_model or str(judge_config.get("model_name", "deepseek-chat") or "deepseek-chat")
    require_api = bool(judge_config.get("require_api", False)) or bool(args.require_judge_api)

    if api_key.strip() in ("YOUR_API_KEY", "your_api_key", "sk-xxx", ""):
        api_key = ""

    return {
        "api_url": normalize_judge_api_url(api_url),
        "api_key": api_key.strip(),
        "model_name": model_name.strip(),
        "require_api": require_api,
    }


def normalize_judge_api_url(api_url: str) -> str:
    """
    规范化 OpenAI-compatible 判分 API URL。

    如果用户只填写了服务根地址，例如 https://api.deepseek.com，
    自动补齐为 https://api.deepseek.com/chat/completions。
    """
    normalized_url = str(api_url or "").strip().rstrip("/")
    if not normalized_url:
        return ""
    if normalized_url.endswith("/chat/completions") or normalized_url.endswith("/completions"):
        return normalized_url
    return "{0}/chat/completions".format(normalized_url)


def validate_judge_config(judge_config: Dict[str, Any]) -> None:
    """在加载测试模型前校验 judge 配置，避免跑完生成后才发现 API 参数错误。"""
    if not judge_config.get("require_api"):
        return
    if not judge_config.get("api_url"):
        raise ValueError("已启用 judge API，但 judge.api_url 为空。请在 configs/model_config.yaml 中填写。")
    if not judge_config.get("api_key"):
        raise ValueError("已启用 judge API，但 judge.api_key 为空。请在 configs/model_config.yaml 中填写真实 API Key。")
    if not judge_config.get("model_name"):
        raise ValueError("已启用 judge API，但 judge.model_name 为空。请在 configs/model_config.yaml 中填写。")


def ensure_output_dirs(output_dir: str) -> Tuple[Path, Path]:
    """创建输出目录，返回 generations 和 metrics 目录。"""
    base_dir = resolve_project_path(output_dir)
    generations_dir = base_dir / "generations"
    metrics_dir = base_dir / "metrics"
    generations_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return generations_dir, metrics_dir


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


def generate_style_poem(model: Any, tokenizer: Any, sample: Dict[str, Any], config: Dict[str, Any]) -> str:
    """根据风格控制样本调用 mock 或 Hugging Face 模型生成诗词。"""
    if hasattr(model, "generate_poem"):
        return clean_generated_text(model.generate_poem(sample["theme"], sample["poem_type"]))

    prompt = messages_to_prompt(tokenizer, build_style_prompt(sample))
    decoded = generate_huggingface_text(model, tokenizer, prompt, config)
    return clean_generated_text(decoded)


def generate_huggingface_text(model: Any, tokenizer: Any, prompt: str, config: Dict[str, Any]) -> str:
    """使用 Hugging Face generate 接口生成文本，只解码新增 token。"""
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
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def judge_generated_poem(
    sample: Dict[str, Any],
    generated_poem: str,
    judge_url: str,
    judge_key: str,
    judge_model: str,
    require_judge_api: bool,
) -> Tuple[str, Dict[str, Any]]:
    """调用判分 API 并解析结果；未提供 API 时可使用 mock 判分自检流程。"""
    judge_messages = build_judge_prompt(sample, generated_poem)

    if judge_url:
        if require_judge_api and not judge_key:
            raise ValueError("已启用 judge API，但 api_key 为空。请在 configs/model_config.yaml 的 judge.api_key 中填写，或传 --judge_key。")
        raw_response = call_judge_api(
            messages=judge_messages,
            api_url=judge_url,
            api_key=judge_key,
            model_name=judge_model,
        )
    elif require_judge_api:
        raise ValueError("已启用 judge API，但未提供 judge_url。请在 configs/model_config.yaml 的 judge.api_url 中填写，或传 --judge_url。")
    else:
        raw_response = _mock_judge_response(generated_poem)

    return raw_response, parse_judge_result(raw_response)


def _mock_judge_response(generated_poem: str) -> str:
    """无 API 时用于自检流程的 mock 判分响应。"""
    if generated_poem.strip():
        return json.dumps(
            {
                "format_score": 60,
                "theme_score": 60,
                "fluency_score": 60,
                "style_score": 60,
                "creativity_score": 60,
                "total_score": 999,
                "brief_comment": "mock 判分：仅用于流程自检，请接入真实 judge API 获取有效评价。",
                "failure_type": "none",
            },
            ensure_ascii=False,
        )
    return "not json"


def evaluate_style_control(
    config: Dict[str, Any],
    output_dir: str,
    max_samples: int,
    judge_url: str,
    judge_key: str,
    judge_model: str,
    require_judge_api: bool,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """执行风格控制生成与 LLM-as-a-Judge 评测流程。"""
    generations_dir, metrics_dir = ensure_output_dirs(output_dir)
    all_samples = build_style_control_samples(max_samples if max_samples > 0 else len(STYLE_CASES))
    samples = all_samples
    model, tokenizer = load_model_and_tokenizer(config)

    records: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="Style control", unit="sample"):
        style_messages = build_style_prompt(sample)
        generation_prompt = messages_to_prompt(tokenizer, style_messages)
        try:
            generated_poem = generate_style_poem(model, tokenizer, sample, config)
            raw_judge_response, judge_result = judge_generated_poem(
                sample=sample,
                generated_poem=generated_poem,
                judge_url=judge_url,
                judge_key=judge_key,
                judge_model=judge_model,
                require_judge_api=require_judge_api,
            )
            error = ""
        except Exception as exc:
            generated_poem = ""
            raw_judge_response = ""
            judge_result = parse_judge_result("")
            error = str(exc)

        records.append(
            {
                "id": sample["id"],
                "task_type": sample["task_type"],
                "theme": sample["theme"],
                "style": sample["style"],
                "poem_type": sample["poem_type"],
                "generation_prompt": generation_prompt,
                "generated_poem": generated_poem,
                "format_score": judge_result["format_score"],
                "theme_score": judge_result["theme_score"],
                "fluency_score": judge_result["fluency_score"],
                "style_score": judge_result["style_score"],
                "creativity_score": judge_result["creativity_score"],
                "total_score": judge_result["total_score"],
                "brief_comment": judge_result["brief_comment"],
                "failure_type": judge_result["failure_type"],
                "raw_judge_response": raw_judge_response,
                "error": error,
            }
        )

    metrics = calculate_metrics(records)
    results_path = generations_dir / "style_control_results.csv"
    metrics_path = metrics_dir / "style_control_metrics.json"
    pd.DataFrame(records).to_csv(results_path, index=False, encoding="utf-8-sig")
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    return results_path, metrics_path, metrics


def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算风格控制 LLM 判分汇总指标。"""
    total_samples = len(records)
    score_keys = [
        "format_score",
        "theme_score",
        "fluency_score",
        "style_score",
        "creativity_score",
        "total_score",
    ]
    metrics: Dict[str, Any] = {
        "task_type": "style_control",
        "total_samples": total_samples,
    }
    for key in score_keys:
        avg_value = sum(float(item[key]) for item in records) / total_samples if total_samples else 0.0
        metrics["average_{0}".format(key)] = round(avg_value, 4)

    failure_counts: Dict[str, int] = {}
    for item in records:
        failure_type = str(item.get("failure_type", "none"))
        failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
    metrics["failure_type_counts"] = failure_counts
    return metrics


def format_output_path(path: Path) -> str:
    """优先展示相对项目根目录的输出路径。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_summary(metrics: Dict[str, Any], results_path: Path, metrics_path: Path) -> None:
    """打印评测摘要。"""
    print("Style Control Evaluation Finished.")
    print("Total samples: {0}".format(metrics["total_samples"]))
    print("Average total score: {0:.4f}".format(metrics["average_total_score"]))
    print("Average format score: {0:.4f}".format(metrics["average_format_score"]))
    print("Average theme score: {0:.4f}".format(metrics["average_theme_score"]))
    print("Average fluency score: {0:.4f}".format(metrics["average_fluency_score"]))
    print("Average style score: {0:.4f}".format(metrics["average_style_score"]))
    print("Average creativity score: {0:.4f}".format(metrics["average_creativity_score"]))
    print("Failure type counts: {0}".format(json.dumps(metrics["failure_type_counts"], ensure_ascii=False)))
    print("Results saved to {0}".format(format_output_path(results_path)))
    print("Metrics saved to {0}".format(format_output_path(metrics_path)))


def main() -> int:
    """程序入口。"""
    args = parse_args()
    try:
        config = load_config(args.config)
        judge_config = resolve_judge_config(config, args)
        validate_judge_config(judge_config)
        results_path, metrics_path, metrics = evaluate_style_control(
            config=config,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            judge_url=judge_config["api_url"],
            judge_key=judge_config["api_key"],
            judge_model=judge_config["model_name"],
            require_judge_api=judge_config["require_api"],
        )
        print_summary(metrics, results_path, metrics_path)
    except Exception as exc:
        print("风格控制评测运行失败：{0}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
