"""theme_generation 任务评测入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
from tqdm import tqdm

from data.data_builder import build_theme_generation_samples
from evaluation.format_check import check_format
from inference.generate import build_prompt, generate_poem
from inference.model_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行主题词生成格式正确率评测。")
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="模型与生成配置文件路径。相对路径默认基于项目根目录解析。",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="评测输出目录。相对路径默认基于项目根目录解析。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="最多评测多少条样本；默认 1000，0 表示只使用原始主题池长度。",
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
        raise FileNotFoundError(f"配置文件不存在: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"配置文件 YAML 解析失败: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("配置文件内容必须是 YAML 字典。")
    return config


def ensure_output_dirs(output_dir: str) -> Tuple[Path, Path]:
    """创建输出目录，返回 generations 和 metrics 目录。"""
    base_dir = resolve_project_path(output_dir)
    generations_dir = base_dir / "generations"
    metrics_dir = base_dir / "metrics"
    generations_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return generations_dir, metrics_dir


def evaluate_theme_generation(config: Dict[str, Any], output_dir: str, max_samples: int) -> Tuple[Path, Path, Dict[str, Any]]:
    """执行 theme_generation 的完整生成与格式评价流程。"""
    generations_dir, metrics_dir = ensure_output_dirs(output_dir)
    samples = build_theme_generation_samples(num_samples=max_samples)
    model, tokenizer = load_model_and_tokenizer(config)

    records: list[dict] = []
    for sample in tqdm(samples, desc="Theme generation", unit="sample"):
        prompt = build_prompt(sample, config)
        try:
            generated_poem = generate_poem(model, tokenizer, sample, config)
        except Exception as exc:
            generated_poem = ""
            format_result = {
                "is_format_correct": False,
                "num_sentences": 0,
                "char_counts": [],
                "errors": [f"生成失败：{exc}"],
            }
        else:
            format_result = check_format(generated_poem, sample["poem_type"])

        records.append(
            {
                "id": sample["id"],
                "task_type": sample["task_type"],
                "theme": sample["theme"],
                "poem_type": sample["poem_type"],
                "prompt": prompt,
                "generated_poem": generated_poem,
                "is_format_correct": format_result["is_format_correct"],
                "num_sentences": format_result["num_sentences"],
                "char_counts": json.dumps(format_result["char_counts"], ensure_ascii=False),
                "errors": json.dumps(format_result["errors"], ensure_ascii=False),
            }
        )

    metrics = calculate_metrics(records)
    results_path = generations_dir / "theme_generation_results.csv"
    metrics_path = metrics_dir / "theme_generation_metrics.json"

    save_results(records, results_path)
    save_metrics(metrics, metrics_path)
    return results_path, metrics_path, metrics


def calculate_metrics(records: list[dict]) -> Dict[str, Any]:
    """计算整体和分诗体格式正确率。"""
    total = len(records)
    correct = sum(1 for item in records if item["is_format_correct"])
    breakdown: dict[str, dict] = {}

    for record in records:
        poem_type = record["poem_type"]
        stats = breakdown.setdefault(poem_type, {"total": 0, "correct": 0, "accuracy": 0.0})
        stats["total"] += 1
        if record["is_format_correct"]:
            stats["correct"] += 1

    for stats in breakdown.values():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0

    return {
        "task_type": "theme_generation",
        "total_samples": total,
        "format_correct_samples": correct,
        "format_accuracy": round(correct / total, 4) if total else 0.0,
        "poem_type_breakdown": breakdown,
    }


def save_results(records: list[dict], results_path: Path) -> None:
    """保存逐条生成结果为 CSV。"""
    try:
        pd.DataFrame(records).to_csv(results_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        raise IOError(f"保存逐条生成结果失败: {results_path}, 错误: {exc}") from exc


def save_metrics(metrics: dict, metrics_path: Path) -> None:
    """保存指标汇总为 JSON。"""
    try:
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise IOError(f"保存指标文件失败: {metrics_path}, 错误: {exc}") from exc


def format_output_path(path: Path) -> str:
    """优先展示相对项目根目录的输出路径。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_summary(metrics: dict, results_path: Path, metrics_path: Path) -> None:
    """在终端打印评测摘要。"""
    print("Theme Generation Evaluation Finished.")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Format correct samples: {metrics['format_correct_samples']}")
    print(f"Format accuracy: {metrics['format_accuracy']:.4f}")
    for poem_type, stats in metrics["poem_type_breakdown"].items():
        print(f"{poem_type}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")
    print(f"Results saved to {format_output_path(results_path)}")
    print(f"Metrics saved to {format_output_path(metrics_path)}")


def main() -> int:
    """程序入口。"""
    args = parse_args()
    try:
        config = load_config(args.config)
        results_path, metrics_path, metrics = evaluate_theme_generation(
            config=config,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
        print_summary(metrics, results_path, metrics_path)
    except Exception as exc:
        print(f"评测运行失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
