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
from typing import Any, Optional

from PIL import Image


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
