#!/usr/bin/env python3
"""Binary-search the highest --max-num-seqs a vLLM server sustains for a model,
on THIS GPU, FAST — so the real run can saturate the card without OOMing.

WHY a probe at all: max-num-seqs is fixed at server launch; guessing too low
wastes the 96 GB Blackwell, too high OOMs during KV-cache reservation / warmup.
We find the ceiling once, cheaply, then launch the real server at it.

ALGORITHM (binary search, not a linear crawl):
  lo = last value KNOWN to start cleanly      (starts at --floor)
  hi = lowest value KNOWN to fail (or None)    (unknown ceiling)
  probe 128 first; double up from a working value until something fails, then
  binary-search the gap between lo and hi until (hi - lo) <= --tol. Each probe
  spins a SHORT-context vLLM ('--probe-max-len', default 4096) just to force the
  KV-cache reservation + a 1-token warmup generate, then tears it down. Short
  context => fast load (~30-60s) and a conservative KV estimate, so the ceiling
  we find is safe for the real (longer) context too.

The KV-cache pool scales with max_model_len, so we ALSO accept --real-max-len:
  the probe scales its memory estimate so the ceiling holds at the real context.
  Concretely: probe at --probe-max-len but cap the search so the chosen value is
  safe at --real-max-len (KV bytes ~ linear in seqs * max_len).

OUTPUT: prints the chosen max-num-seqs as the LAST line on stdout (so a shell can
`SEQS=$(python probe_max_seqs.py ... | tail -1)`), and a JSON trace to stderr.

This uses vLLM in-process (offline LLM), NOT a server — fastest possible probe.
No frontier API. GPU-only.
"""
import argparse
import gc
import json
import sys
import time


def _try_seqs(model_id, n_seqs, probe_max_len, gpu_util, log):
    """Return True if vLLM loads + reserves KV + generates 1 token at n_seqs."""
    t0 = time.monotonic()
    llm = None
    try:
        import torch
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=model_id, dtype="bfloat16",
            max_model_len=probe_max_len, gpu_memory_utilization=gpu_util,
            max_num_seqs=n_seqs, limit_mm_per_prompt={"image": 6},
            trust_remote_code=True, enforce_eager=True,  # skip CUDA-graph capture: faster probe
        )
        # Force a real decode so the sampler/logits scratch is actually allocated.
        llm.generate(["ping"] * min(n_seqs, 8),
                     SamplingParams(temperature=0, max_tokens=1))
        log.append({"seqs": n_seqs, "ok": True, "secs": round(time.monotonic() - t0, 1)})
        return True
    except Exception as e:  # OOM or any startup failure => this value is too high
        msg = repr(e)[:300]
        log.append({"seqs": n_seqs, "ok": False, "err": msg,
                    "secs": round(time.monotonic() - t0, 1)})
        return False
    finally:
        try:
            if llm is not None:
                del llm
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--start", type=int, default=128, help="first probe point")
    ap.add_argument("--floor", type=int, default=8, help="min seqs we'd ever use")
    ap.add_argument("--ceil", type=int, default=512, help="never exceed this")
    ap.add_argument("--tol", type=int, default=8,
                    help="stop when hi-lo <= tol; pick lo")
    ap.add_argument("--probe-max-len", type=int, default=4096)
    ap.add_argument("--real-max-len", type=int, default=24576,
                    help="context the REAL server uses; the chosen seqs is "
                         "scaled down so KV fits at this length")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    args = ap.parse_args()

    log = []
    # KV pool scales ~linearly with seqs*max_len. We probe at probe_max_len but
    # the real server runs at real_max_len, so a value safe in the probe must be
    # divided by (real_max_len / probe_max_len) to stay safe for real. We bake
    # that ratio into the search ceiling.
    scale = max(1.0, args.real_max_len / max(1, args.probe_max_len))

    def probe(n):
        # n is in REAL-context units; convert to probe-context units (more seqs
        # fit at the short probe length) so the probe exercises the same KV bytes.
        probe_n = max(args.floor, int(round(n * scale)))
        return _try_seqs(args.model_id, probe_n, args.probe_max_len, args.gpu_util, log)

    lo = 0          # highest known-good (real units); 0 = none yet
    hi = None       # lowest known-bad (real units)
    cur = min(args.start, args.ceil)

    # Phase 1: bracket. Probe start; if good, double until a failure or ceil.
    if probe(cur):
        lo = cur
        while lo < args.ceil:
            nxt = min(args.ceil, lo * 2)
            if nxt == lo:
                break
            if probe(nxt):
                lo = nxt
            else:
                hi = nxt
                break
    else:
        hi = cur
        # step down by halving until something works or we hit the floor
        while hi > args.floor:
            nxt = max(args.floor, hi // 2)
            if probe(nxt):
                lo = nxt
                break
            else:
                hi = nxt
        if lo == 0:  # even floor failed
            print(json.dumps({"trace": log, "chosen": args.floor,
                              "note": "floor failed too; using floor anyway"}),
                  file=sys.stderr)
            print(args.floor)
            return

    # Phase 2: binary-search the (lo, hi) gap.
    if hi is None:
        chosen = lo if lo else args.floor
    else:
        while hi - lo > args.tol:
            mid = (lo + hi) // 2
            if mid <= lo:
                break
            if probe(mid):
                lo = mid
            else:
                hi = mid
        chosen = lo if lo else args.floor

    chosen = max(args.floor, min(args.ceil, chosen))
    print(json.dumps({"trace": log, "chosen": chosen, "scale": round(scale, 2),
                      "real_max_len": args.real_max_len}), file=sys.stderr)
    print(chosen)  # LAST stdout line = the answer


if __name__ == "__main__":
    main()
