"""高兼容性 LLM-as-a-Judge 调用与解析模块。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.style_prompt import build_judge_prompt, build_style_prompt


SCORE_KEYS = [
    "format_score",
    "theme_score",
    "fluency_score",
    "style_score",
    "creativity_score",
]
SCORE_WEIGHTS = {
    "format_score": 0.25,
    "theme_score": 0.25,
    "fluency_score": 0.20,
    "style_score": 0.20,
    "creativity_score": 0.10,
}


def call_judge_api(
    messages: List[Dict[str, str]],
    api_url: str,
    api_key: str,
    model_name: str,
) -> str:
    """
    调用兼容 OpenAI Chat Completions 格式的大模型判分接口。

    Payload: {"model": model_name, "messages": messages, "temperature": 0.1}
    返回模型响应中的纯文本内容。
    """
    if not api_url:
        raise ValueError("api_url 不能为空。")
    if not model_name:
        raise ValueError("model_name 不能为空。")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空 List[Dict]。")

    try:
        import requests
    except ImportError as exc:
        raise ImportError("缺少 requests，无法调用判分 API。请运行：pip install requests") from exc

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer {0}".format(api_key)

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError("判分 API 调用失败：{0}".format(exc)) from exc

    try:
        response_data = response.json()
    except ValueError:
        return response.text.strip()

    return _extract_openai_compatible_text(response_data)


def _extract_openai_compatible_text(response_data: Dict[str, Any]) -> str:
    """从 OpenAI-compatible 响应中提取 assistant 文本，兼容常见变体。"""
    choices = response_data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if content is not None:
                    return str(content).strip()
            text = first_choice.get("text")
            if text is not None:
                return str(text).strip()

    for key in ("output_text", "content", "text"):
        value = response_data.get(key)
        if value is not None:
            return str(value).strip()

    return json.dumps(response_data, ensure_ascii=False)


def parse_judge_result(raw_response: str) -> Dict[str, Any]:
    """
    强健解析判分模型返回结果，并在代码层面重算 total_score。

    解析失败时返回兜底结果：所有分数为 0，总分为 0.0，failure_type=parse_error。
    """
    try:
        json_text = _extract_json_text(raw_response)
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError("判分结果 JSON 不是对象。")
    except Exception:
        return _fallback_parse_error()

    result = dict(parsed)
    for key in SCORE_KEYS:
        result[key] = _normalize_score(result.get(key, 0))

    result["total_score"] = _calculate_total_score(result)
    result["brief_comment"] = str(result.get("brief_comment", "") or "")
    result["failure_type"] = str(result.get("failure_type", "none") or "none")
    return result


def _extract_json_text(raw_response: str) -> str:
    """
    从模型原始输出中提取 JSON 字符串。

    优先提取 ```json ... ``` 或 ``` ... ``` 代码块；如果没有 markdown 标记，
    则截取首个 { 到最后一个 } 之间的内容。
    """
    if raw_response is None:
        raise ValueError("raw_response 不能为空。")

    text = str(raw_response).strip()
    if not text:
        raise ValueError("raw_response 不能为空。")

    fenced_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
    fenced_matches = fenced_pattern.findall(text)
    for candidate in fenced_matches:
        candidate = candidate.strip()
        try:
            json.loads(candidate)
        except Exception:
            continue
        return candidate

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("未找到 JSON 对象。")
    return text[start:end + 1].strip()


def _normalize_score(value: Any) -> float:
    """将分数字段规范到 0-100 区间，缺失或非法时默认 0 分。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0:
        return 0.0
    if score > 100:
        return 100.0
    return score


def _calculate_total_score(result: Dict[str, Any]) -> float:
    """严格按权重重算总分，覆盖模型返回的 total_score。"""
    total = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        total += float(result.get(key, 0.0)) * weight
    return round(total, 4)


def _fallback_parse_error() -> Dict[str, Any]:
    """解析失败兜底结果。"""
    return {
        "format_score": 0.0,
        "theme_score": 0.0,
        "fluency_score": 0.0,
        "style_score": 0.0,
        "creativity_score": 0.0,
        "total_score": 0.0,
        "brief_comment": "判分结果解析失败，已使用最低兜底分。",
        "failure_type": "parse_error",
    }


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(description="风格控制生成的 LLM-as-a-Judge 测试桩。")
    parser.add_argument("--judge_url", default="", help="兼容 OpenAI 格式的判分 API URL。")
    parser.add_argument(
        "--judge_key",
        default=os.getenv("JUDGE_API_KEY", ""),
        help="判分 API Key；默认读取环境变量 JUDGE_API_KEY。",
    )
    parser.add_argument("--judge_model", default="deepseek-chat", help="判分模型名称。")
    return parser.parse_args()


def main() -> int:
    """展示风格控制 prompt、判分 prompt、API 调用与解析流程。"""
    args = parse_args()
    sample = {
        "id": "S01",
        "task_type": "style_control",
        "theme": "月夜",
        "keywords": [],
        "prefix": "",
        "style": "思乡",
        "poem_type": "五言绝句",
    }
    generated_poem = "明月照孤城，秋风动客情。故园千里外，清梦到三更。"

    style_messages = build_style_prompt(sample)
    judge_messages = build_judge_prompt(sample, generated_poem)

    if args.judge_url:
        try:
            raw_response = call_judge_api(
                messages=judge_messages,
                api_url=args.judge_url,
                api_key=args.judge_key,
                model_name=args.judge_model,
            )
        except Exception as exc:
            print("判分 API 调用失败：{0}".format(exc), file=sys.stderr)
            return 1
    else:
        raw_response = """
下面是评分结果：
```json
{
  "format_score": 82,
  "theme_score": 90,
  "fluency_score": 84,
  "style_score": 88,
  "creativity_score": 80,
  "total_score": 999,
  "brief_comment": "诗句围绕月夜思乡展开，意象清晰，语言较流畅。",
  "failure_type": "none"
}
```
"""

    parsed_result = parse_judge_result(raw_response)
    output = {
        "style_generation_messages": style_messages,
        "judge_messages": judge_messages,
        "raw_judge_response": raw_response,
        "parsed_judge_result": parsed_result,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
