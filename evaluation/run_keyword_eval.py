"""keyword_generation 任务评测入口。"""

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

from evaluation.keyword_coverage import evaluate_keyword_coverage
from inference.generate import clean_generated_text, generate_from_chat_messages
from inference.keyword_prompt import build_keyword_prompt
from inference.model_loader import load_model_and_tokenizer


KEYWORD_CASES: List[Tuple[str, List[str]]] = [
    ("月夜", ["明月", "故乡", "清风"]),
    ("江南", ["春雨", "杨柳", "归舟"]),
    ("边塞", ["黄沙", "孤城", "战马"]),
    ("田园", ["青山", "流水", "白云"]),
    ("离别", ["长亭", "落日", "故人"]),
    ("春天", ["春风", "花开", "燕子"]),
    ("夏日", ["荷花", "蝉声", "绿阴"]),
    ("秋思", ["秋风", "归雁", "寒灯"]),
    ("冬雪", ["白雪", "寒梅", "孤村"]),
    ("清明", ["细雨", "行人", "杏花"]),
    ("中秋", ["圆月", "桂影", "家书"]),
    ("新年", ["爆竹", "东风", "桃符"]),
    ("山水", ["青山", "碧水", "云烟"]),
    ("江河", ["江流", "渔火", "客船"]),
    ("大海", ["沧海", "潮声", "孤帆"]),
    ("白云", ["白云", "远岫", "松声"]),
    ("落日", ["落日", "长河", "孤烟"]),
    ("晚霞", ["晚霞", "飞鸟", "归林"]),
    ("青山", ["青山", "古道", "松风"]),
    ("流水", ["流水", "小桥", "人家"]),
    ("花开", ["桃花", "春水", "东风"]),
    ("落花", ["落花", "流水", "残香"]),
    ("杨柳", ["杨柳", "春烟", "离亭"]),
    ("孤舟", ["孤舟", "夜泊", "江月"]),
    ("飞鸟", ["飞鸟", "长空", "夕阳"]),
    ("寒梅", ["寒梅", "疏影", "雪月"]),
    ("竹林", ["修竹", "清泉", "幽径"]),
    ("松柏", ["松柏", "石径", "寒云"]),
    ("荷花", ["荷花", "清露", "画船"]),
    ("枫叶", ["枫叶", "秋水", "寒山"]),
    ("思乡", ["故乡", "明月", "归梦"]),
    ("相思", ["红豆", "春山", "锦书"]),
    ("孤独", ["孤灯", "夜雨", "空庭"]),
    ("怀古", ["古台", "残碑", "夕照"]),
    ("忧愁", ["寒灯", "秋雨", "长夜"]),
    ("喜悦", ["春风", "新燕", "晴光"]),
    ("闲适", ["柴门", "白云", "清茶"]),
    ("感怀", ["浮生", "旧梦", "斜阳"]),
    ("伤春", ["落花", "杜鹃", "春尽"]),
    ("人生", ["浮云", "沧海", "归舟"]),
    ("时光", ["流年", "残照", "旧尘"]),
    ("梦想", ["长风", "远路", "星河"]),
    ("远方", ["关山", "孤雁", "长亭"]),
    ("归隐", ["柴扉", "松月", "渔樵"]),
    ("漂泊", ["孤帆", "客路", "风尘"]),
    ("壮志", ["长剑", "关河", "凌云"]),
    ("知己", ["美酒", "高山", "流水"]),
    ("故人", ["故人", "酒盏", "西窗"]),
    ("岁月", ["青丝", "白发", "流光"]),
    ("边塞", ["关山", "胡笳", "铁衣"]),
    ("田园", ["桑麻", "鸡犬", "夕烟"]),
    ("江南", ["烟雨", "小桥", "画舫"]),
    ("长安", ["长安", "宫阙", "春灯"]),
    ("古寺", ["古寺", "钟声", "松月"]),
    ("夜雨", ["夜雨", "寒窗", "孤灯"]),
    ("客栈", ["客栈", "青灯", "远客"]),
    ("渡口", ["渡口", "潮声", "归帆"]),
    ("山村", ["山村", "柴门", "犬吠"]),
    ("渔家", ["渔火", "蓑衣", "晚潮"]),
    ("美酒", ["美酒", "金樽", "月色"]),
    ("琴声", ["琴声", "竹影", "清夜"]),
    ("书卷", ["书卷", "灯火", "寒窗"]),
    ("灯火", ["灯火", "夜市", "归人"]),
    ("诗酒", ["诗酒", "江月", "青衫"]),
    ("征人", ["征人", "关月", "寒衣"]),
    ("游子", ["游子", "乡书", "客梦"]),
    ("故乡", ["故乡", "桑梓", "归云"]),
    ("关山", ["关山", "塞月", "边风"]),
    ("楼台", ["楼台", "烟柳", "斜阳"]),
    ("晨景", ["朝霞", "晓钟", "新露"]),
    ("暮景", ["暮云", "归鸟", "晚钟"]),
    ("湖上", ["湖光", "莲叶", "画桥"]),
    ("溪边", ["溪声", "苔石", "野花"]),
    ("山寺", ["山寺", "钟磬", "松阴"]),
    ("秋夜", ["秋月", "寒砧", "疏星"]),
    ("春游", ["芳草", "柳色", "莺啼"]),
    ("送别", ["离歌", "酒旗", "长路"]),
    ("登高", ["高台", "远树", "秋江"]),
    ("望月", ["明月", "天涯", "乡心"]),
    ("听雨", ["雨声", "竹窗", "孤枕"]),
    ("赏花", ["花枝", "香径", "春衫"]),
    ("观潮", ["潮头", "海门", "白浪"]),
    ("塞外", ["大漠", "孤烟", "羌笛"]),
    ("军旅", ["铁骑", "旌旗", "鼓角"]),
    ("宫怨", ["宫花", "玉阶", "秋扇"]),
    ("闺思", ["罗衣", "银烛", "远书"]),
    ("农家", ["麦浪", "桑田", "村酒"]),
    ("渔舟", ["渔舟", "江雪", "蓑笠"]),
    ("山居", ["山居", "石泉", "松门"]),
    ("旅夜", ["旅馆", "寒灯", "乡梦"]),
    ("春雨", ["春雨", "梨花", "燕泥"]),
    ("秋风", ["秋风", "黄叶", "孤雁"]),
    ("冬夜", ["冬夜", "寒炉", "雪窗"]),
    ("夏雨", ["夏雨", "荷声", "雷云"]),
    ("梅雪", ["梅花", "白雪", "清香"]),
    ("兰亭", ["兰亭", "曲水", "流觞"]),
    ("边月", ["边月", "胡尘", "戍楼"]),
    ("江雪", ["江雪", "孤舟", "寒蓑"]),
    ("云梦", ["云梦", "烟波", "楚天"]),
    ("雨巷", ["雨巷", "油伞", "丁香"]),
]


def build_keyword_generation_samples() -> List[Dict[str, Any]]:
    """构造 100 条关键词约束生成样本。"""
    samples: List[Dict[str, Any]] = []
    for index, (theme, keywords) in enumerate(KEYWORD_CASES):
        poem_type = "五言绝句" if index % 2 == 0 else "七言绝句"
        samples.append(
            {
                "id": "K{0:03d}".format(index + 1),
                "task_type": "keyword_generation",
                "theme": theme,
                "keywords": keywords,
                "prefix": "",
                "style": "",
                "poem_type": poem_type,
            }
        )
    return samples


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行关键词约束生成评测。")
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
        default=0,
        help="最多评测多少条样本；0 表示使用全部 100 条内置关键词样本。",
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


def ensure_output_dirs(output_dir: str) -> Tuple[Path, Path]:
    """创建输出目录，返回 generations 和 metrics 目录。"""
    base_dir = resolve_project_path(output_dir)
    generations_dir = base_dir / "generations"
    metrics_dir = base_dir / "metrics"
    generations_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return generations_dir, metrics_dir


def generate_keyword_poem(model: Any, tokenizer: Any, sample: Dict[str, Any], config: Dict[str, Any]) -> str:
    """根据关键词样本调用 mock 或 Hugging Face 模型生成诗词。"""
    messages = build_keyword_prompt(sample)
    poem_type = str(sample.get("poem_type") or "五言绝句")

    if hasattr(model, "generate_poem"):
        keywords = sample.get("keywords", [])
        return clean_generated_text(_mock_keyword_poem(keywords, poem_type), poem_type=poem_type)

    return generate_from_chat_messages(
        model,
        tokenizer,
        messages,
        config,
        poem_type=poem_type,
        continuation_prefix="",
    )


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


def _mock_keyword_poem(keywords: List[str], poem_type: str) -> str:
    """为 mock 模型构造包含关键词的简单诗句。"""
    target_len = 5 if poem_type == "五言绝句" else 7
    clean_keywords = [str(item).strip() for item in keywords if str(item).strip()]
    base_text = "".join(clean_keywords) + "清风明月山河花雨云"
    chars = [char for char in base_text if "\u4e00" <= char <= "\u9fff"]
    while len(chars) < target_len * 4:
        chars.extend(list("清风明月山河花雨云"))
    lines = ["".join(chars[index:index + target_len]) for index in range(0, target_len * 4, target_len)]
    return "\n".join(
        "{0}{1}".format(line, "，" if index in (0, 2) else "。")
        for index, line in enumerate(lines)
    )


def evaluate_keyword_generation(config: Dict[str, Any], output_dir: str, max_samples: int) -> Tuple[Path, Path, Dict[str, Any]]:
    """执行关键词约束生成评测流程。"""
    generations_dir, metrics_dir = ensure_output_dirs(output_dir)
    all_samples = build_keyword_generation_samples()
    samples = all_samples[:max_samples] if max_samples > 0 else all_samples
    model, tokenizer = load_model_and_tokenizer(config)

    records: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="Keyword generation", unit="sample"):
        messages = build_keyword_prompt(sample)
        prompt_text = messages_to_prompt(tokenizer, messages)
        try:
            generated_poem = generate_keyword_poem(model, tokenizer, sample, config)
            eval_result = evaluate_keyword_coverage(
                generated_poem,
                sample["keywords"],
                include_repetition_rate=True,
            )
        except Exception as exc:
            generated_poem = ""
            eval_result = {
                "hit_count": 0,
                "total_count": len(sample.get("keywords", [])),
                "hit_rate": 0.0,
                "missed_keywords": sample.get("keywords", []),
                "repetition_rate": 0.0,
                "repeated_fragment_count": 0,
                "total_fragment_count": 0,
                "repeated_fragments": {},
                "fragment_size": 2,
                "error": str(exc),
            }

        records.append(
            {
                "id": sample["id"],
                "task_type": sample["task_type"],
                "theme": sample["theme"],
                "poem_type": sample["poem_type"],
                "keywords": json.dumps(sample["keywords"], ensure_ascii=False),
                "prompt": prompt_text,
                "generated_poem": generated_poem,
                "hit_count": eval_result["hit_count"],
                "total_count": eval_result["total_count"],
                "hit_rate": eval_result["hit_rate"],
                "missed_keywords": json.dumps(eval_result["missed_keywords"], ensure_ascii=False),
                "repetition_rate": eval_result.get("repetition_rate", 0.0),
                "repeated_fragments": json.dumps(eval_result.get("repeated_fragments", {}), ensure_ascii=False),
                "error": eval_result.get("error", ""),
            }
        )

    metrics = calculate_metrics(records)
    results_path = generations_dir / "keyword_generation_results.csv"
    metrics_path = metrics_dir / "keyword_generation_metrics.json"
    pd.DataFrame(records).to_csv(results_path, index=False, encoding="utf-8-sig")
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    return results_path, metrics_path, metrics


def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算关键词约束生成整体指标。"""
    total_samples = len(records)
    avg_hit_rate = sum(float(item["hit_rate"]) for item in records) / total_samples if total_samples else 0.0
    all_hit_samples = sum(1 for item in records if float(item["hit_rate"]) == 1.0)
    avg_repetition_rate = (
        sum(float(item["repetition_rate"]) for item in records) / total_samples
        if total_samples
        else 0.0
    )
    return {
        "task_type": "keyword_generation",
        "total_samples": total_samples,
        "all_keywords_hit_samples": all_hit_samples,
        "all_keywords_hit_accuracy": round(all_hit_samples / total_samples, 4) if total_samples else 0.0,
        "average_hit_rate": round(avg_hit_rate, 4),
        "average_repetition_rate": round(avg_repetition_rate, 4),
    }


def format_output_path(path: Path) -> str:
    """优先展示相对项目根目录的输出路径。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_summary(metrics: Dict[str, Any], results_path: Path, metrics_path: Path) -> None:
    """打印评测摘要。"""
    print("Keyword Generation Evaluation Finished.")
    print("Total samples: {0}".format(metrics["total_samples"]))
    print("All keywords hit samples: {0}".format(metrics["all_keywords_hit_samples"]))
    print("All keywords hit accuracy: {0:.4f}".format(metrics["all_keywords_hit_accuracy"]))
    print("Average hit rate: {0:.4f}".format(metrics["average_hit_rate"]))
    print("Average repetition rate: {0:.4f}".format(metrics["average_repetition_rate"]))
    print("Results saved to {0}".format(format_output_path(results_path)))
    print("Metrics saved to {0}".format(format_output_path(metrics_path)))


def main() -> int:
    """程序入口。"""
    args = parse_args()
    try:
        config = load_config(args.config)
        results_path, metrics_path, metrics = evaluate_keyword_generation(
            config=config,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
        print_summary(metrics, results_path, metrics_path)
    except Exception as exc:
        print("关键词评测运行失败：{0}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
