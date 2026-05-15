"""prefix_continuation 任务评测入口。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
from tqdm import tqdm

from evaluation.bleu_rouge import clean_to_spaced_chars, evaluate_bleu_rouge
from inference.continuation_prompt import build_continuation_prompt
from inference.generate import clean_continuation_text
from inference.model_loader import load_model_and_tokenizer


PREFIX_CASES: List[Tuple[str, str, str, List[str]]] = [
    ("举头望明月", "低头思故乡。", "思乡", ["低头思故乡。"]),
    ("春眠不觉晓", "处处闻啼鸟。", "春天", ["处处闻啼鸟。"]),
    ("白日依山尽", "黄河入海流。", "登高", ["黄河入海流。"]),
    ("海内存知己", "天涯若比邻。", "友情", ["天涯若比邻。"]),
    ("野火烧不尽", "春风吹又生。", "草原", ["春风吹又生。"]),
    ("明月松间照", "清泉石上流。", "山居", ["清泉石上流。"]),
    ("大漠孤烟直", "长河落日圆。", "边塞", ["长河落日圆。"]),
    ("会当凌绝顶", "一览众山小。", "壮志", ["一览众山小。"]),
    ("山重水复疑无路", "柳暗花明又一村。", "山村", ["柳暗花明又一村。"]),
    ("洛阳亲友如相问", "一片冰心在玉壶。", "送别", ["一片冰心在玉壶。"]),
    ("劝君更尽一杯酒", "西出阳关无故人。", "离别", ["西出阳关无故人。"]),
    ("两岸猿声啼不住", "轻舟已过万重山。", "江行", ["轻舟已过万重山。"]),
    ("孤帆远影碧空尽", "唯见长江天际流。", "送别", ["唯见长江天际流。"]),
    ("接天莲叶无穷碧", "映日荷花别样红。", "荷花", ["映日荷花别样红。"]),
    ("停车坐爱枫林晚", "霜叶红于二月花。", "秋景", ["霜叶红于二月花。"]),
    ("忽如一夜春风来", "千树万树梨花开。", "冬雪", ["千树万树梨花开。"]),
    ("但愿人长久", "千里共婵娟。", "中秋", ["千里共婵娟。"]),
    ("人生自古谁无死", "留取丹心照汗青。", "壮志", ["留取丹心照汗青。"]),
    ("天街小雨润如酥", "草色遥看近却无。", "春雨", ["草色遥看近却无。"]),
    ("小荷才露尖尖角", "早有蜻蜓立上头。", "夏日", ["早有蜻蜓立上头。"]),
    ("柴门闻犬吠", "风雪夜归人。", "冬夜", ["风雪夜归人。"]),
    ("空山新雨后", "天气晚来秋。", "秋山", ["天气晚来秋。"]),
    ("月落乌啼霜满天", "江枫渔火对愁眠。", "夜泊", ["江枫渔火对愁眠。"]),
    ("葡萄美酒夜光杯", "欲饮琵琶马上催。", "边塞", ["欲饮琵琶马上催。"]),
    ("黄河远上白云间", "一片孤城万仞山。", "边塞", ["一片孤城万仞山。"]),
    ("日照香炉生紫烟", "遥看瀑布挂前川。", "山水", ["遥看瀑布挂前川。"]),
    ("朝辞白帝彩云间", "千里江陵一日还。", "江行", ["千里江陵一日还。"]),
    ("千山鸟飞绝", "万径人踪灭。", "江雪", ["万径人踪灭。"]),
    ("采菊东篱下", "悠然见南山。", "田园", ["悠然见南山。"]),
    ("夕阳无限好", "只是近黄昏。", "感怀", ["只是近黄昏。"]),
]


def build_prefix_continuation_samples() -> List[Dict[str, Any]]:
    """构造上句续写测试样本。"""
    samples: List[Dict[str, Any]] = []
    for index, (prefix, reference, theme, references) in enumerate(_build_large_prefix_cases()):
        poem_type = "下一句续写"
        samples.append(
            {
                "id": "P{0:03d}".format(index + 1),
                "task_type": "prefix_continuation",
                "theme": theme,
                "keywords": [],
                "prefix": prefix,
                "style": "",
                "poem_type": poem_type,
                "references": references or [reference],
            }
        )
    return samples


def _build_large_prefix_cases() -> List[Tuple[str, str, str, List[str]]]:
    """构造较大的内置上句-下句样本池。"""
    extra_cases: List[Tuple[str, str, str, List[str]]] = [
        ("床前明月光", "疑是地上霜。", "月夜", ["疑是地上霜。"]),
        ("疑是地上霜", "举头望明月。", "月夜", ["举头望明月。"]),
        ("移舟泊烟渚", "日暮客愁新。", "旅夜", ["日暮客愁新。"]),
        ("日暮客愁新", "野旷天低树。", "旅夜", ["野旷天低树。"]),
        ("野旷天低树", "江清月近人。", "江夜", ["江清月近人。"]),
        ("红豆生南国", "春来发几枝。", "相思", ["春来发几枝。"]),
        ("春来发几枝", "愿君多采撷。", "相思", ["愿君多采撷。"]),
        ("愿君多采撷", "此物最相思。", "相思", ["此物最相思。"]),
        ("松下问童子", "言师采药去。", "寻隐", ["言师采药去。"]),
        ("言师采药去", "只在此山中。", "寻隐", ["只在此山中。"]),
        ("只在此山中", "云深不知处。", "寻隐", ["云深不知处。"]),
        ("独坐幽篁里", "弹琴复长啸。", "竹林", ["弹琴复长啸。"]),
        ("弹琴复长啸", "深林人不知。", "竹林", ["深林人不知。"]),
        ("深林人不知", "明月来相照。", "竹林", ["明月来相照。"]),
        ("人闲桂花落", "夜静春山空。", "春山", ["夜静春山空。"]),
        ("夜静春山空", "月出惊山鸟。", "春山", ["月出惊山鸟。"]),
        ("月出惊山鸟", "时鸣春涧中。", "春山", ["时鸣春涧中。"]),
        ("返景入深林", "复照青苔上。", "山林", ["复照青苔上。"]),
        ("空山不见人", "但闻人语响。", "山林", ["但闻人语响。"]),
        ("但闻人语响", "返景入深林。", "山林", ["返景入深林。"]),
        ("功盖三分国", "名成八阵图。", "怀古", ["名成八阵图。"]),
        ("名成八阵图", "江流石不转。", "怀古", ["江流石不转。"]),
        ("江流石不转", "遗恨失吞吴。", "怀古", ["遗恨失吞吴。"]),
        ("迟日江山丽", "春风花草香。", "春景", ["春风花草香。"]),
        ("春风花草香", "泥融飞燕子。", "春景", ["泥融飞燕子。"]),
        ("泥融飞燕子", "沙暖睡鸳鸯。", "春景", ["沙暖睡鸳鸯。"]),
        ("江碧鸟逾白", "山青花欲燃。", "春望", ["山青花欲燃。"]),
        ("山青花欲燃", "今春看又过。", "春望", ["今春看又过。"]),
        ("今春看又过", "何日是归年。", "思归", ["何日是归年。"]),
        ("两个黄鹂鸣翠柳", "一行白鹭上青天。", "春景", ["一行白鹭上青天。"]),
        ("一行白鹭上青天", "窗含西岭千秋雪。", "春景", ["窗含西岭千秋雪。"]),
        ("窗含西岭千秋雪", "门泊东吴万里船。", "春景", ["门泊东吴万里船。"]),
        ("岐王宅里寻常见", "崔九堂前几度闻。", "怀旧", ["崔九堂前几度闻。"]),
        ("崔九堂前几度闻", "正是江南好风景。", "怀旧", ["正是江南好风景。"]),
        ("正是江南好风景", "落花时节又逢君。", "怀旧", ["落花时节又逢君。"]),
        ("故人西辞黄鹤楼", "烟花三月下扬州。", "送别", ["烟花三月下扬州。"]),
        ("烟花三月下扬州", "孤帆远影碧空尽。", "送别", ["孤帆远影碧空尽。"]),
        ("千里黄云白日曛", "北风吹雁雪纷纷。", "送别", ["北风吹雁雪纷纷。"]),
        ("北风吹雁雪纷纷", "莫愁前路无知己。", "送别", ["莫愁前路无知己。"]),
        ("莫愁前路无知己", "天下谁人不识君。", "送别", ["天下谁人不识君。"]),
        ("渭城朝雨浥轻尘", "客舍青青柳色新。", "送别", ["客舍青青柳色新。"]),
        ("客舍青青柳色新", "劝君更尽一杯酒。", "送别", ["劝君更尽一杯酒。"]),
        ("秦时明月汉时关", "万里长征人未还。", "边塞", ["万里长征人未还。"]),
        ("万里长征人未还", "但使龙城飞将在。", "边塞", ["但使龙城飞将在。"]),
        ("但使龙城飞将在", "不教胡马度阴山。", "边塞", ["不教胡马度阴山。"]),
        ("青海长云暗雪山", "孤城遥望玉门关。", "边塞", ["孤城遥望玉门关。"]),
        ("孤城遥望玉门关", "黄沙百战穿金甲。", "边塞", ["黄沙百战穿金甲。"]),
        ("黄沙百战穿金甲", "不破楼兰终不还。", "边塞", ["不破楼兰终不还。"]),
        ("寒雨连江夜入吴", "平明送客楚山孤。", "送别", ["平明送客楚山孤。"]),
        ("平明送客楚山孤", "洛阳亲友如相问。", "送别", ["洛阳亲友如相问。"]),
        ("千里莺啼绿映红", "水村山郭酒旗风。", "江南", ["水村山郭酒旗风。"]),
        ("水村山郭酒旗风", "南朝四百八十寺。", "江南", ["南朝四百八十寺。"]),
        ("南朝四百八十寺", "多少楼台烟雨中。", "江南", ["多少楼台烟雨中。"]),
        ("远上寒山石径斜", "白云生处有人家。", "秋山", ["白云生处有人家。"]),
        ("白云生处有人家", "停车坐爱枫林晚。", "秋山", ["停车坐爱枫林晚。"]),
        ("银烛秋光冷画屏", "轻罗小扇扑流萤。", "秋夕", ["轻罗小扇扑流萤。"]),
        ("轻罗小扇扑流萤", "天阶夜色凉如水。", "秋夕", ["天阶夜色凉如水。"]),
        ("天阶夜色凉如水", "卧看牵牛织女星。", "秋夕", ["卧看牵牛织女星。"]),
        ("少小离家老大回", "乡音无改鬓毛衰。", "回乡", ["乡音无改鬓毛衰。"]),
        ("乡音无改鬓毛衰", "儿童相见不相识。", "回乡", ["儿童相见不相识。"]),
        ("儿童相见不相识", "笑问客从何处来。", "回乡", ["笑问客从何处来。"]),
        ("爆竹声中一岁除", "春风送暖入屠苏。", "新年", ["春风送暖入屠苏。"]),
        ("春风送暖入屠苏", "千门万户曈曈日。", "新年", ["千门万户曈曈日。"]),
        ("千门万户曈曈日", "总把新桃换旧符。", "新年", ["总把新桃换旧符。"]),
        ("横看成岭侧成峰", "远近高低各不同。", "山水", ["远近高低各不同。"]),
        ("远近高低各不同", "不识庐山真面目。", "山水", ["不识庐山真面目。"]),
        ("不识庐山真面目", "只缘身在此山中。", "山水", ["只缘身在此山中。"]),
        ("京口瓜洲一水间", "钟山只隔数重山。", "思乡", ["钟山只隔数重山。"]),
        ("钟山只隔数重山", "春风又绿江南岸。", "思乡", ["春风又绿江南岸。"]),
        ("春风又绿江南岸", "明月何时照我还。", "思乡", ["明月何时照我还。"]),
        ("死去元知万事空", "但悲不见九州同。", "忧国", ["但悲不见九州同。"]),
        ("但悲不见九州同", "王师北定中原日。", "忧国", ["王师北定中原日。"]),
        ("王师北定中原日", "家祭无忘告乃翁。", "忧国", ["家祭无忘告乃翁。"]),
        ("水光潋滟晴方好", "山色空蒙雨亦奇。", "西湖", ["山色空蒙雨亦奇。"]),
        ("山色空蒙雨亦奇", "欲把西湖比西子。", "西湖", ["欲把西湖比西子。"]),
        ("欲把西湖比西子", "淡妆浓抹总相宜。", "西湖", ["淡妆浓抹总相宜。"]),
    ]
    return PREFIX_CASES + extra_cases


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行上句续写 BLEU/ROUGE 评测。")
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
        help="最多评测多少条样本；默认从 Hugging Face 数据集中随机抽取 1000 条，0 表示使用全部可用样本。",
    )
    parser.add_argument(
        "--samples_file",
        default="",
        help="可选外部样本文件，支持 json/jsonl；可直接使用从 Chinese poetry 数据导出的样本。",
    )
    parser.add_argument(
        "--sample_source",
        choices=["hf", "builtin"],
        default="hf",
        help="样本来源；默认 hf，表示从 Hugging Face Chinese poetry 数据集随机抽取。",
    )
    parser.add_argument(
        "--hf_dataset",
        default="MatrixStudio/ChinesePoetry",
        help="Hugging Face 数据集名称。",
    )
    parser.add_argument(
        "--hf_split",
        default="train",
        help="Hugging Face 数据集 split。",
    )
    parser.add_argument(
        "--hf_scan_limit",
        type=int,
        default=0,
        help="最多扫描多少条 HF 原始记录；0 表示不限制。",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="是否在评测前随机打乱样本，适合从大数据集中抽样。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机抽样种子。",
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


def load_prefix_samples(
    samples_file: str,
    sample_source: str,
    hf_dataset: str,
    hf_split: str,
    hf_scan_limit: int,
    target_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """加载上句续写样本；默认从 Hugging Face Chinese poetry 数据集抽样。"""
    if not samples_file:
        if sample_source == "hf":
            try:
                samples = build_huggingface_prefix_samples(
                    dataset_name=hf_dataset,
                    split=hf_split,
                    scan_limit=hf_scan_limit,
                    target_samples=target_samples,
                    seed=seed,
                )
                if samples:
                    return samples
                print("提示：Hugging Face 数据集未抽取到有效样本，已回退到内置样本池。")
            except Exception as exc:
                print("提示：无法从 Hugging Face 数据集加载样本，已回退到内置样本池。原因：{0}".format(exc))
        return build_prefix_continuation_samples()

    path = resolve_project_path(samples_file)
    if not path.exists():
        raise FileNotFoundError("样本文件不存在: {0}".format(path))

    raw_items: List[Any] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    raw_items.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        raw_items = data if isinstance(data, list) else [data]

    samples: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        samples.extend(_normalize_external_prefix_item(item, len(samples)))
    return samples


def build_huggingface_prefix_samples(
    dataset_name: str,
    split: str,
    scan_limit: int,
    target_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    从 Hugging Face Chinese poetry 数据集中随机抽取上句续写样本。

    默认数据集 MatrixStudio/ChinesePoetry 包含 full_poem 字段，脚本会从诗句中
    切分相邻上下句，构造 prefix -> reference 的评测样本。
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("缺少 datasets，无法加载 Hugging Face 数据集。请运行：pip install datasets") from exc

    dataset = load_dataset(dataset_name, split=split)
    candidates: List[Dict[str, Any]] = []
    for row_index, item in enumerate(dataset):
        if scan_limit > 0 and row_index >= scan_limit:
            break
        if not isinstance(item, dict):
            continue
        candidates.extend(_samples_from_hf_item(item))

    if not candidates:
        return []

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates if target_samples <= 0 else candidates[:target_samples]
    for index, sample in enumerate(selected):
        sample["id"] = "P{0:04d}".format(index + 1)
    return selected


def _samples_from_hf_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从单条 Hugging Face 数据记录中抽取相邻上下句样本。"""
    text = _extract_poem_text(item)
    sentences = _split_poem_sentences(text)
    title = str(item.get("title", item.get("theme", "")) or "")

    samples: List[Dict[str, Any]] = []
    for index in range(len(sentences) - 1):
        prefix = sentences[index]
        reference = sentences[index + 1]
        if not _is_good_poem_line(prefix) or not _is_good_poem_line(reference):
            continue
        samples.append(
            {
                "id": "",
                "task_type": "prefix_continuation",
                "theme": title,
                "keywords": [],
                "prefix": prefix,
                "style": "",
                "poem_type": _infer_poem_type(prefix),
                "references": [reference],
            }
        )
    return samples


def _extract_poem_text(item: Dict[str, Any]) -> str:
    """兼容多种 Hugging Face Chinese poetry 字段名提取诗文正文。"""
    for key in ("full_poem", "content", "text", "poem"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value

    paragraphs = item.get("paragraphs", item.get("paragraph", []))
    if isinstance(paragraphs, str):
        return paragraphs
    if isinstance(paragraphs, list):
        return " ".join(str(paragraph) for paragraph in paragraphs)
    return ""


def _is_good_poem_line(line: str) -> bool:
    """过滤过短、过长或中文含量过低的句子。"""
    chinese_chars = [char for char in line if "\u4e00" <= char <= "\u9fff"]
    if len(chinese_chars) < 4 or len(chinese_chars) > 12:
        return False
    return len(chinese_chars) / max(len(line), 1) >= 0.8


def _normalize_external_prefix_item(item: Dict[str, Any], start_index: int) -> List[Dict[str, Any]]:
    """
    将外部样本规范化为 prefix_continuation 格式。

    支持两类输入：
    1. 已经包含 prefix/references 的标准样本；
    2. Chinese poetry 常见的 paragraphs/paragraph 字段，从相邻句子自动抽取上下句。
    """
    if item.get("prefix") and item.get("references"):
        return [
            {
                "id": str(item.get("id") or "P{0:03d}".format(start_index + 1)),
                "task_type": "prefix_continuation",
                "theme": str(item.get("theme", "")),
                "keywords": item.get("keywords", []),
                "prefix": str(item["prefix"]),
                "style": str(item.get("style", "")),
                "poem_type": str(item.get("poem_type", _infer_poem_type(str(item["prefix"])))),
                "references": [str(ref) for ref in item["references"]],
            }
        ]

    paragraphs = item.get("paragraphs", item.get("paragraph", []))
    if isinstance(paragraphs, str):
        sentences = _split_poem_sentences(paragraphs)
    elif isinstance(paragraphs, list):
        sentences = []
        for paragraph in paragraphs:
            sentences.extend(_split_poem_sentences(str(paragraph)))
    else:
        sentences = []

    samples: List[Dict[str, Any]] = []
    for index in range(len(sentences) - 1):
        prefix = sentences[index]
        reference = sentences[index + 1]
        samples.append(
            {
                "id": "P{0:03d}".format(start_index + len(samples) + 1),
                "task_type": "prefix_continuation",
                "theme": str(item.get("title", item.get("theme", ""))),
                "keywords": [],
                "prefix": prefix,
                "style": "",
                "poem_type": _infer_poem_type(prefix),
                "references": [reference],
            }
        )
    return samples


def _split_poem_sentences(text: str) -> List[str]:
    """按常见中文诗句标点切分句子。"""
    import re

    return [part.strip() for part in re.split(r"[，。！？；、,.!?;\n]+", text) if part.strip()]


def _infer_poem_type(prefix: str) -> str:
    """上句续写不强行绑定绝句诗体，只记录任务形式。"""
    return "下一句续写"


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


def generate_prefix_continuation(model: Any, tokenizer: Any, sample: Dict[str, Any], config: Dict[str, Any]) -> str:
    """根据上句续写样本调用 mock 或 Hugging Face 模型生成续写。"""
    if hasattr(model, "generate_poem"):
        return clean_continuation_text(sample["references"][0], prefix=sample.get("prefix", ""))

    prompt = messages_to_prompt(tokenizer, build_continuation_prompt(sample))
    decoded = generate_huggingface_text(model, tokenizer, prompt, config)
    return clean_continuation_text(decoded, prefix=sample.get("prefix", ""))


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


def evaluate_prefix_continuation(
    config: Dict[str, Any],
    output_dir: str,
    max_samples: int,
    samples_file: str,
    sample_source: str,
    hf_dataset: str,
    hf_split: str,
    hf_scan_limit: int,
    shuffle: bool,
    seed: int,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """执行上句续写评测流程。"""
    generations_dir, metrics_dir = ensure_output_dirs(output_dir)
    all_samples = load_prefix_samples(
        samples_file=samples_file,
        sample_source=sample_source,
        hf_dataset=hf_dataset,
        hf_split=hf_split,
        hf_scan_limit=hf_scan_limit,
        target_samples=max_samples,
        seed=seed,
    )
    if shuffle:
        random.Random(seed).shuffle(all_samples)
    samples = all_samples[:max_samples] if max_samples > 0 else all_samples
    model, tokenizer = load_model_and_tokenizer(config)

    records: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="Prefix continuation", unit="sample"):
        messages = build_continuation_prompt(sample)
        prompt_text = messages_to_prompt(tokenizer, messages)
        try:
            generated_text = generate_prefix_continuation(model, tokenizer, sample, config)
            scores = evaluate_bleu_rouge(generated_text, sample["references"])
            error = ""
        except Exception as exc:
            generated_text = ""
            scores = {
                "BLEU-4": 0.0,
                "ROUGE-L": 0.0,
                "BLEU-4-%": 0.0,
                "ROUGE-L-%": 0.0,
            }
            error = str(exc)

        records.append(
            {
                "id": sample["id"],
                "task_type": sample["task_type"],
                "theme": sample["theme"],
                "poem_type": sample["poem_type"],
                "prefix": sample["prefix"],
                "references": json.dumps(sample["references"], ensure_ascii=False),
                "prompt": prompt_text,
                "generated_text": generated_text,
                "generated_char_level": clean_to_spaced_chars(generated_text),
                "references_char_level": json.dumps(
                    [clean_to_spaced_chars(item) for item in sample["references"]],
                    ensure_ascii=False,
                ),
                "BLEU-4": scores["BLEU-4"],
                "ROUGE-L": scores["ROUGE-L"],
                "BLEU-4-%": scores.get("BLEU-4-%", round(scores["BLEU-4"] * 100, 2)),
                "ROUGE-L-%": scores.get("ROUGE-L-%", round(scores["ROUGE-L"] * 100, 2)),
                "error": error,
            }
        )

    metrics = calculate_metrics(records)
    results_path = generations_dir / "prefix_continuation_results.csv"
    metrics_path = metrics_dir / "prefix_continuation_metrics.json"
    pd.DataFrame(records).to_csv(results_path, index=False, encoding="utf-8-sig")
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    return results_path, metrics_path, metrics


def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算上句续写整体指标。"""
    total_samples = len(records)
    avg_bleu = sum(float(item["BLEU-4"]) for item in records) / total_samples if total_samples else 0.0
    avg_rouge = sum(float(item["ROUGE-L"]) for item in records) / total_samples if total_samples else 0.0
    avg_bleu_percent = sum(float(item["BLEU-4-%"]) for item in records) / total_samples if total_samples else 0.0
    avg_rouge_percent = sum(float(item["ROUGE-L-%"]) for item in records) / total_samples if total_samples else 0.0
    return {
        "task_type": "prefix_continuation",
        "total_samples": total_samples,
        "average_BLEU-4": round(avg_bleu, 4),
        "average_ROUGE-L": round(avg_rouge, 4),
        "average_BLEU-4-%": round(avg_bleu_percent, 2),
        "average_ROUGE-L-%": round(avg_rouge_percent, 2),
    }


def format_output_path(path: Path) -> str:
    """优先展示相对项目根目录的输出路径。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_summary(metrics: Dict[str, Any], results_path: Path, metrics_path: Path) -> None:
    """打印评测摘要。"""
    print("Prefix Continuation Evaluation Finished.")
    print("Total samples: {0}".format(metrics["total_samples"]))
    print("Average BLEU-4: {0:.4f}".format(metrics["average_BLEU-4"]))
    print("Average ROUGE-L: {0:.4f}".format(metrics["average_ROUGE-L"]))
    print("Average BLEU-4-%: {0:.2f}".format(metrics["average_BLEU-4-%"]))
    print("Average ROUGE-L-%: {0:.2f}".format(metrics["average_ROUGE-L-%"]))
    print("Results saved to {0}".format(format_output_path(results_path)))
    print("Metrics saved to {0}".format(format_output_path(metrics_path)))


def main() -> int:
    """程序入口。"""
    args = parse_args()
    try:
        config = load_config(args.config)
        results_path, metrics_path, metrics = evaluate_prefix_continuation(
            config=config,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            samples_file=args.samples_file,
            sample_source=args.sample_source,
            hf_dataset=args.hf_dataset,
            hf_split=args.hf_split,
            hf_scan_limit=args.hf_scan_limit,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        print_summary(metrics, results_path, metrics_path)
    except Exception as exc:
        print("上句续写评测运行失败：{0}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
