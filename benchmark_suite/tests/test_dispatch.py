"""test_dispatch.py — backend resolution + worklist build, NO model loaded."""
from __future__ import annotations

import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent.parent
REPO = SUITE.parent
sys.path.insert(0, str(SUITE))

import run


def test_resolve_backend():
    # attention forces transformers even on raw weights
    assert run.resolve_backend({"key": "m", "attention": True}) == "transformers"
    # lora -> transformers
    assert run.resolve_backend({"key": "m", "lora_path": "x"}) == "transformers"
    # explicit vllm + attention -> error
    try:
        run.resolve_backend({"key": "m", "backend": "vllm", "attention": True})
        assert False, "should have raised"
    except ValueError:
        pass
    # explicit vllm + lora -> error
    try:
        run.resolve_backend({"key": "m", "backend": "vllm", "lora_path": "x"})
        assert False, "should have raised"
    except ValueError:
        pass
    # raw weights, no force: vllm if available else transformers
    b = run.resolve_backend({"key": "m"})
    assert b in ("vllm", "transformers")
    print("resolve_backend OK (raw->%s)" % b)


def test_worklist():
    import yaml
    cfg = yaml.safe_load(open(SUITE / "config.yaml"))
    ds = cfg["dataset"]
    data_path = REPO / ds["path"]
    image_root = REPO / ds["image_root"]
    records = run.load_records(data_path, limit_per_topic=1)
    # one record per topic
    assert len(records) == 14, f"expected 14 topics, got {len(records)}"
    work = run.build_worklist(records, image_root, cfg["defaults"]["modes"], image_max_edge=512)
    assert work, "no work items"
    # every work item's image count matches its role count
    for w in work:
        assert len(w.images) == len(w.roles)
        # blind has no images; sat/sv/full have >=1
        if w.mode == "blind":
            assert len(w.images) == 0
        else:
            assert len(w.images) >= 1
        # the prompt mentions every shown image
        for i in range(len(w.roles)):
            assert f"Image {i+1}:" in w.user_text or not w.roles
    modes_seen = {w.mode for w in work}
    assert "full" in modes_seen and "blind" in modes_seen
    print(f"worklist OK: {len(work)} items, modes={sorted(modes_seen)}")


if __name__ == "__main__":
    test_resolve_backend()
    test_worklist()
    print("\nALL DISPATCH TESTS PASSED")
