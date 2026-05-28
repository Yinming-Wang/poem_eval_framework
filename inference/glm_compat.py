"""兼容 zai-org/glm-4-9b-chat 旧版 remote code 与新版 transformers。

逻辑对齐 hw/poetry/run_eval.py，供 WebUI 与评测脚本加载/生成时复用。
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import GenerationConfig, PreTrainedModel


def patch_legacy_glm4_compat() -> None:
    """
    新版 accelerate 在 device_map=\"auto\" 时会访问 model.all_tied_weights_keys；
    旧版 GLM remote code 可能缺失该属性，需补默认 dict。
    """
    current = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if not isinstance(current, dict):
        PreTrainedModel.all_tied_weights_keys = {}  # type: ignore[attr-defined]


def patch_config_for_legacy_glm4(config: Any) -> None:
    """旧版 modeling_chatglm 初始化需要 config.max_length，新版 config 常只有 seq_length。"""
    if not hasattr(config, "max_length"):
        if hasattr(config, "seq_length"):
            config.max_length = config.seq_length
        elif hasattr(config, "max_position_embeddings"):
            config.max_length = config.max_position_embeddings
        else:
            config.max_length = 131072

    if not hasattr(config, "seq_length") and hasattr(config, "max_length"):
        config.seq_length = config.max_length


def remove_max_length_from_config(config: Any) -> None:
    """删除为初始化临时写入的 max_length，避免新版 generate() 报错。"""
    if config is None:
        return
    try:
        if "max_length" in getattr(config, "__dict__", {}):
            delattr(config, "max_length")
    except Exception:
        try:
            config.__dict__.pop("max_length", None)
        except Exception:
            pass


def cleanup_generation_config(model: torch.nn.Module, tokenizer: Any) -> None:
    """
    加载后清理：删除 config.max_length，并设置 generation_config（含 use_cache=False）。
    """
    default_gen_config = GenerationConfig()
    default_max_length = default_gen_config.max_length

    candidates: list[Any] = [model]

    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        inner_model = getattr(base_model, "model", None)
        if inner_model is not None:
            candidates.append(inner_model)

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            real_base = get_base_model()
            if real_base is not None:
                candidates.append(real_base)
        except Exception:
            pass

    seen_ids: set[int] = set()

    for m in candidates:
        if m is None or id(m) in seen_ids:
            continue
        seen_ids.add(id(m))

        cfg = getattr(m, "config", None)
        remove_max_length_from_config(cfg)

        try:
            gen_cfg = GenerationConfig.from_model_config(cfg) if cfg is not None else GenerationConfig()
        except Exception:
            gen_cfg = GenerationConfig()

        gen_cfg.max_length = default_max_length
        gen_cfg.max_new_tokens = None
        gen_cfg.eos_token_id = tokenizer.eos_token_id
        gen_cfg.pad_token_id = tokenizer.pad_token_id
        gen_cfg.use_cache = False

        try:
            m.generation_config = gen_cfg
        except Exception:
            pass


def patch_model_tied_weights_keys(model: torch.nn.Module) -> None:
    """给模型实例补 all_tied_weights_keys。"""
    current = getattr(model, "all_tied_weights_keys", None)
    if isinstance(current, dict):
        return

    tied = getattr(model, "_tied_weights_keys", None)
    if isinstance(tied, dict):
        model.all_tied_weights_keys = tied  # type: ignore[attr-defined]
    else:
        model.all_tied_weights_keys = {}  # type: ignore[attr-defined]


def get_input_device(model: torch.nn.Module) -> torch.device:
    """device_map='auto' 时，将输入放到模型第一个参数所在设备。"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_glm_generation_config(tokenizer: Any, generation_dict: dict[str, Any]) -> GenerationConfig:
    """与 run_eval.build_generation_config 对齐：禁用 KV cache，避免 DynamicCache 与旧版 GLM 不兼容。"""
    temperature = generation_dict.get("temperature", 0.8)
    do_sample = bool(generation_dict.get("do_sample", True)) and float(temperature or 0) > 0

    pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None and getattr(tokenizer, "pad_token_id", None) is not None:
        pad_token_id = tokenizer.pad_token_id

    generation_config = GenerationConfig(
        max_new_tokens=int(generation_dict.get("max_new_tokens", 128)),
        do_sample=do_sample,
        top_p=float(generation_dict.get("top_p", 0.95)),
        repetition_penalty=float(generation_dict.get("repetition_penalty", 1.1)),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=False,
    )

    if do_sample:
        generation_config.temperature = max(float(temperature), 1e-5)
        top_k = generation_dict.get("top_k", 50)
        if top_k and int(top_k) > 0:
            generation_config.top_k = int(top_k)

    return generation_config
