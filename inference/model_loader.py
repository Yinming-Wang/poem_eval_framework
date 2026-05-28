"""统一模型加载接口。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_peft_adapter_dir(model_path: Path) -> bool:
    return (model_path / "adapter_config.json").is_file()


def _read_peft_base_model(adapter_dir: Path) -> str:
    cfg_path = adapter_dir / "adapter_config.json"
    with cfg_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    base_id = data.get("base_model_name_or_path")
    if not base_id:
        raise ValueError("{0} 缺少 base_model_name_or_path。".format(cfg_path))
    return str(base_id).strip()


@dataclass
class MockPoemModel:
    """用于无真实模型时测试完整流程的 mock 模型。"""

    model_type: str = "mock"

    def generate_poem(self, theme: str, poem_type: str) -> str:
        """根据主题和诗体生成固定格式诗句。"""
        target_len = 5 if poem_type == "五言绝句" else 7
        lines = _build_mock_lines(theme, target_len)
        return "\n".join(lines)


def _build_mock_lines(theme: str, target_len: int) -> List[str]:
    """构造每句指定汉字数的 mock 诗句。"""
    clean_theme = "".join(ch for ch in theme if "\u4e00" <= ch <= "\u9fff") or "诗情"
    first = _fit_chinese_line(clean_theme + "入诗心", target_len)

    if target_len == 5:
        candidates = [
            first,
            "清风过远山",
            "明月照归舟",
            "花影落庭深",
        ]
    else:
        candidates = [
            first,
            "清风万里过青山",
            "明月一轮照客船",
            "花影满庭入梦来",
        ]
    return [_with_punctuation(line, idx) for idx, line in enumerate(candidates)]


def _fit_chinese_line(text: str, target_len: int) -> str:
    """截断或补足汉字，使句子满足目标字数。"""
    chinese_text = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    padding = "清风明月山河花雨云"
    while len(chinese_text) < target_len:
        chinese_text += padding
    return chinese_text[:target_len]


def _with_punctuation(line: str, index: int) -> str:
    """为 mock 诗句添加常见中文标点。"""
    return f"{line}{'，' if index in (0, 2) else '。'}"


def load_model_and_tokenizer(config: dict):
    """
    根据配置加载模型和 tokenizer。

    当前支持 mock、Hugging Face 自回归语言模型、以及本地字符级 LSTM（models/ 子目录下 vocab.json
    含 char2idx/idx2char + .pt 权重）。
    """
    model_config = config.get("model", {})
    model_type = str(model_config.get("model_type", "mock")).strip().lower()

    if model_type == "mock":
        return MockPoemModel(), None

    if model_type == "lstm":
        return _load_lstm_model(model_config)

    if model_type == "huggingface":
        return _load_huggingface_model(model_config)

    raise ValueError(
        f"不支持的 model_type: {model_type}。当前支持: mock, huggingface, lstm。"
    )


def _load_glm_peft_model(
    adapter_dir: Path,
    tokenizer_path: Path,
) -> Tuple[Any, Any]:
    """加载 glm-4-9b-chat 基座 + 本地 LoRA（对齐 hw/poetry/run_eval.py）。"""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "加载 GLM LoRA 需要 transformers、torch 与 peft。请安装：pip install peft"
        ) from exc

    from inference.glm_compat import (
        cleanup_generation_config,
        get_input_device,
        patch_config_for_legacy_glm4,
        patch_legacy_glm4_compat,
        patch_model_tied_weights_keys,
    )

    base_id = _read_peft_base_model(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    patch_legacy_glm4_compat()

    config = AutoConfig.from_pretrained(base_id, trust_remote_code=True)
    patch_config_for_legacy_glm4(config)

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cuda:0" if use_cuda else None,
    )
    if not use_cuda:
        model.to("cpu")

    patch_model_tied_weights_keys(model)
    cleanup_generation_config(model, tokenizer)

    model = PeftModel.from_pretrained(model, str(adapter_dir))
    patch_model_tied_weights_keys(model)
    cleanup_generation_config(model, tokenizer)

    model.eval()
    setattr(model, "model_type", "huggingface")
    setattr(model, "eval_device", get_input_device(model))
    setattr(model, "_poem_eval_legacy_glm_generate", True)
    return model, tokenizer


def _load_generic_peft_model(
    adapter_dir: Path,
    tokenizer_path: Path,
    requested_device: str,
) -> Tuple[Any, Any]:
    """非 GLM 的 LoRA：基座来自 adapter_config.base_model_name_or_path。"""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "检测到 LoRA 目录，需要 peft。请安装：pip install peft"
        ) from exc

    base_id = _read_peft_base_model(adapter_dir)
    actual_device = requested_device
    if actual_device.startswith("cuda") and not torch.cuda.is_available():
        _print_cuda_fallback_diagnostics(torch)
        actual_device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_id, trust_remote_code=True)
    model.to(actual_device)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(actual_device)
    model.eval()

    setattr(model, "model_type", "huggingface")
    setattr(model, "eval_device", torch.device(actual_device))
    setattr(model, "_poem_eval_legacy_glm_generate", False)
    return model, tokenizer


def _load_huggingface_model(model_config: dict):
    """加载 Hugging Face 风格模型，并处理路径和设备异常。"""
    model_path = _resolve_project_path(model_config.get("model_path"))
    tokenizer_path = _resolve_project_path(model_config.get("tokenizer_path") or model_config.get("model_path"))
    requested_device = str(model_config.get("device", "cpu")).strip().lower()

    if not model_path:
        raise ValueError("配置错误：model.model_path 不能为空。")
    if not tokenizer_path:
        raise ValueError("配置错误：model.tokenizer_path 不能为空。")
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer 路径不存在: {tokenizer_path}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "缺少 Hugging Face 依赖。请先安装 transformers 和 torch，"
            "或将 configs/model_config.yaml 中的 model_type 改为 mock。"
        ) from exc

    if _is_peft_adapter_dir(model_path):
        base_id_lower = _read_peft_base_model(model_path).lower()
        is_glm = "glm-4" in base_id_lower or "chatglm" in base_id_lower
        try:
            if is_glm:
                model, tokenizer = _load_glm_peft_model(model_path, tokenizer_path)
            else:
                model, tokenizer = _load_generic_peft_model(
                    model_path, tokenizer_path, requested_device
                )
        except Exception as exc:
            raise RuntimeError(f"LoRA / Hugging Face 模型加载失败: {exc}") from exc
        return model, tokenizer

    device = requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        _print_cuda_fallback_diagnostics(torch)
        device = "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"Hugging Face 模型加载失败: {exc}") from exc

    setattr(model, "model_type", "huggingface")
    setattr(model, "eval_device", device)
    setattr(model, "_poem_eval_legacy_glm_generate", False)
    return model, tokenizer


def _load_lstm_model(model_config: dict):
    """加载 poem_eval_framework 约定的 LSTM 目录（char 词表 + checkpoint）。"""
    from inference.lstm_backend import load_lstm_checkpoint_bundle

    model_path = _resolve_project_path(model_config.get("model_path"))
    requested_device = str(model_config.get("device", "cpu")).strip().lower()

    if not model_path:
        raise ValueError("配置错误：model.model_path 不能为空。")
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    bundle = load_lstm_checkpoint_bundle(model_path, requested_device)
    return bundle, None


def _resolve_project_path(path_value: Any) -> Optional[Path]:
    """将配置中的相对路径按项目根目录解析，绝对路径保持不变。"""
    if path_value is None:
        return None
    path = Path(str(path_value).strip())
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _print_cuda_fallback_diagnostics(torch_module: Any) -> None:
    """打印 CUDA 回退 CPU 时的诊断信息，帮助定位环境问题。"""
    print("提示：配置使用 cuda，但当前 PyTorch 未检测到可用 GPU，已自动退回 CPU。")
    print("CUDA 诊断信息：")
    print(f"  Python: {sys.executable}")
    print(f"  torch: {getattr(torch_module, '__version__', 'unknown')}")
    print(f"  torch.version.cuda: {getattr(torch_module.version, 'cuda', None)}")
    print(f"  torch.cuda.is_available(): {torch_module.cuda.is_available()}")
    print(f"  torch.cuda.device_count(): {torch_module.cuda.device_count()}")
    print("如果 nvidia-smi 能看到显卡，但这里 is_available=False，通常是当前环境安装了 CPU 版 PyTorch，")
    print("或当前 Python/conda 环境不是你安装 CUDA 版 PyTorch 的那个环境。")
