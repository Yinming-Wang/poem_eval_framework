"""字符级 LSTM 诗词生成后端（与 training_and_infer_for_lstm 对话格式一致）。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
SPECIALS = [PAD, UNK, BOS, EOS]

SYSTEM_MARK = "<|system|>\n"
USER_MARK = "<|user|>\n"
ASSISTANT_MARK = "<|assistant|>\n"
END_MARK = "<|end|>\n"

SYSTEM_PROMPT = (
    "你是一位精通中国古典诗词的诗人。"
    "请严格按照用户指定的诗体和格式要求创作，只输出诗句，不要添加标题、作者、解释或任何额外内容。"
)


def build_conversation_prompt(system: str, instruction: str) -> str:
    return f"{SYSTEM_MARK}{system}{USER_MARK}{instruction}{ASSISTANT_MARK}"


def build_conversation_full(system: str, instruction: str, output: str) -> str:
    return build_conversation_prompt(system, instruction) + output + END_MARK


class Vocab:
    def __init__(self, char2idx: dict[str, int], idx2char: list[str]):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.pad_id = char2idx[PAD]
        self.unk_id = char2idx[UNK]
        self.bos_id = char2idx[BOS]
        self.eos_id = char2idx[EOS]

    def __len__(self) -> int:
        return len(self.char2idx)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.char2idx.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            ch = self.idx2char[i]
            if skip_special and ch in SPECIALS:
                continue
            out.append(ch)
        return "".join(out)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["char2idx"], data["idx2char"])


class CharLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.drop(self.embed(x))
        out, hidden = self.lstm(emb, hidden)
        logits = self.head(self.drop(out))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


@torch.no_grad()
def char_lstm_generate(
    model: CharLSTM,
    vocab: Vocab,
    prompt: str = "",
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    device: Optional[torch.device] = None,
) -> str:
    model.eval()
    device = device or next(model.parameters()).device
    ids = vocab.encode(prompt, add_bos=True, add_eos=False)
    if not ids:
        ids = [vocab.bos_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = model.init_hidden(1, device)

    if len(ids) > 1:
        logits, hidden = model(x[:, :-1], hidden)
        x = x[:, -1:]

    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits, hidden = model(x, hidden)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        logits[:, vocab.pad_id] = -1e9
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -1e9
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        token = int(nxt.item())
        if token == vocab.eos_id:
            break
        generated.append(token)
        x = nxt

    return vocab.decode(generated)


def is_lstm_vocab_file(path: Path) -> bool:
    """LSTM 词表为 char2idx + idx2char；与 HF BPE vocab.json 区分。"""
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict):
        return False
    return "char2idx" in data and "idx2char" in data


def pick_lstm_checkpoint(model_dir: Path) -> Optional[Path]:
    for name in ("best.pt", "final.pt"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    pts = sorted(model_dir.glob("*.pt"))
    if not pts:
        return None
    return pts[0]


def load_lstm_checkpoint_bundle(model_dir: Path, requested_device: str) -> "LSTMPoemBundle":
    """从目录加载 LSTMPoemBundle：vocab.json + best.pt / final.pt / 任一 .pt。"""
    ckpt_path = pick_lstm_checkpoint(model_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            "未在 {0} 中找到 LSTM 检查点（需要 best.pt、final.pt 或任意 .pt 文件）。".format(model_dir)
        )

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("提示：配置使用 cuda，但当前未检测到可用 GPU，LSTM 已退回 CPU。")
        device = torch.device("cpu")
    elif requested_device.startswith("cuda"):
        device = torch.device(requested_device)
    else:
        device = torch.device("cpu")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ckpt_dir = str(ckpt_path.parent)
    rel_vocab = ckpt.get("vocab_path", "vocab.json")
    vocab_path = os.path.join(ckpt_dir, rel_vocab)
    vocab = Vocab.load(vocab_path)

    cfg = ckpt.get("config", {}) or {}
    model = CharLSTM(
        len(vocab),
        embed_dim=int(cfg.get("embed_dim", 256)),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
        pad_id=vocab.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return LSTMPoemBundle(
        char_model=model,
        vocab=vocab,
        eval_device=device,
        checkpoint_path=ckpt_path,
    )


@dataclass
class LSTMPoemBundle:
    """与 HuggingFace 模型并列的待测模型封装。"""

    char_model: CharLSTM
    vocab: Vocab
    eval_device: torch.device
    checkpoint_path: Path

    model_type: str = "lstm"

    def generate_chars(self, prompt: str, config: Dict[str, Any]) -> str:
        generation = config.get("generation", {}) if isinstance(config, dict) else {}
        return char_lstm_generate(
            self.char_model,
            self.vocab,
            prompt=prompt,
            max_new_tokens=int(generation.get("max_new_tokens", 128)),
            temperature=float(generation.get("temperature", 0.8)),
            top_k=int(generation.get("top_k", 50)),
            device=self.eval_device,
        )


def messages_to_lstm_prompt(messages: List[Dict[str, str]]) -> str:
    """将 Chat messages 转为 LSTM 训练同款前缀（至 <|assistant|> ）。"""
    system_text = ""
    user_texts: List[str] = []
    for message in messages or []:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if role == "system":
            system_text = content
        elif role == "user":
            user_texts.append(content)
    system = system_text.strip() if system_text.strip() else SYSTEM_PROMPT
    user = "\n".join(user_texts).strip()
    return build_conversation_prompt(system, user)
