"""Quick smoke test for the local vLLM OpenAI-compatible server.

Usage:
    python test_client.py                # text-only test on all 3 models
    python test_client.py <path/to/image.jpg>  # multimodal test
"""
from __future__ import annotations
import sys
from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

MODELS = ["qwen3.5-4b-base", "pc-base", "su-base"]


def list_models() -> None:
    print("Available models:")
    for m in client.models.list().data:
        print(f"  - {m.id}")
    print()


def text_test(model: str) -> None:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with one word: hello?"}],
        max_tokens=8,
        temperature=0.0,
    )
    print(f"[{model}] {resp.choices[0].message.content!r}")


def image_test(model: str, image_path: str) -> None:
    img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }],
        max_tokens=64,
        temperature=0.0,
    )
    print(f"[{model}] {resp.choices[0].message.content!r}")


if __name__ == "__main__":
    list_models()
    image = sys.argv[1] if len(sys.argv) > 1 else None
    for m in MODELS:
        try:
            if image:
                image_test(m, image)
            else:
                text_test(m)
        except Exception as e:
            print(f"[{m}] ERROR: {e}")
