"""backends/base.py — the uniform contract every backend implements.

run.py is backend-agnostic: it builds a worklist of WorkItems (one per
record×mode, with the post-ablation images and prompt already constructed) and
hands them to a Backend. The backend loads its model once and returns one
ResultRow per item. This keeps image construction, prompting, ablation and
metrics identical across vLLM / transformers / attention — only the forward pass
differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_model_id(hf_id: str) -> str:
    """Resolve a model id to either a local path or a pass-through hub id.

    A `hf_id` may be:
      - an absolute local path            -> used as-is
      - a repo-relative path that EXISTS  -> made absolute (e.g. models/Qwen3.5-4B)
      - anything else                     -> treated as a HF hub id, UNCHANGED
                                             (e.g. cyankiwi/Qwen3.5-4B-AWQ-4bit)

    The last case is the bug guard: blindly prefixing the repo root onto a hub
    id like "cyankiwi/..." produced a bogus path and an OSError. We only localize
    when the path actually resolves on disk.
    """
    p = Path(hf_id)
    if p.is_absolute():
        return hf_id
    local = _REPO_ROOT / hf_id
    if local.exists():
        return str(local)
    return hf_id  # hub id — leave it alone


@dataclass
class WorkItem:
    index: int                      # record index in the dataset
    record: dict                    # the raw benchmark record
    mode: str                       # ablation mode (full/sat_only/sv_only/blind)
    images: list[Image.Image]       # post-ablation PIL images, in feed order
    roles: list[str]                # role label per image (parallel to `images`)
    user_text: str                  # the textual user turn (prompt.build_user_text)


@dataclass
class ResultRow:
    index: int
    mode: str
    gold: Optional[str]
    prediction: Optional[str]
    raw_response: str
    prob_dict: Optional[dict] = None      # {"A":p,...} normalized over options
    hedged: bool = False
    refused: bool = False
    n_images: int = 0
    roles: list[str] = field(default_factory=list)
    attention: Optional[dict] = None      # per-image attention block (attention backend only)
    extra: dict = field(default_factory=dict)


class Backend:
    """Subclasses load a model in __init__ and implement run()."""

    name = "base"

    def run(self, items: list[WorkItem]) -> list[ResultRow]:
        raise NotImplementedError

    def close(self):
        pass
