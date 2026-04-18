# EOLLM Pre-Production Audit Report

## Verdict: **NO-GO**

The advertised **22% → 75.3%** result is **not apples-to-apples** and a pure-text heuristic on the validation set already scores **47.7%**. The remote run as currently configured would (a) fall back to the wrong GPU profile and train a completely different model than the one that produced the local number, (b) repeat the same flawed eval methodology, and (c) ship a thesis-level overclaim. Burning the one-shot remote slot on this configuration wastes the slot. Six high-impact fixes are tractable in a day; after them, launch with confidence. Details below.

---

## Top 3 Blockers (must fix before remote launch)

1. **No `rtx_pro_6000_96gb` GPU profile.** `training/config.py:32-57` silently falls back to `safe_16gb` for any unrecognized device name. On the Pro 6000 that means `image_max_edge=448` (local was 768), rank-8 LoRA (local was 16), **frozen vision layers** (local finetuned them), and `initial_batch_guess=1`. The remote run would train a fundamentally different model than the one that produced 75.3%.
2. **The 22% base / 75.3% finetuned comparison is methodologically invalid.** `training/evaluation.py:58-61,124-128` runs the *in-training* eval (producing both numbers) with stochastic sampling (`temperature=0.7, top_p=0.8, top_k=20`) and a strict substring match (`generated.startswith(rec["answer"])`). `training/eval_base.py` runs the same model greedy + lenient-parse and reports **27.7%** for the same base, not 22%. Per-topic "vs Base" in the verdict files is even worse: it implies camera_direction base=0% and building_height base=0%, while the real greedy-eval base is 26.2% and 29.1%.
3. **A text-only heuristic already gets 47.7% on val; one topic hits 94.2% from a length heuristic alone.** Verified under `audit_scratch/out_09_heuristic.txt`. "Pick modal option string per topic" yields 47.7% val accuracy with zero vision. "Pick shortest option" scores 94.2% on `road_surface` (correct = "Asphalt" for 94% of val). The +53 pp headline is at least 20 pp driven by label-prior and length heuristics rather than visual grounding.

Fix these three and you have a defensible run. The other issues below compound but are second-order.

---

## Blocker List (priority order)

| # | Finding | Severity | Owner |
|---|---------|----------|-------|
| 1 | Missing `rtx_pro_6000_96gb` GPU profile; silent fallback to `safe_16gb` | Catastrophic | B1 |
| 2 | In-training eval uses stochastic sampling + strict prefix match — invalidates 22 vs 75.3 | Catastrophic | A1 / B3 |
| 3 | Dataset class-prior + length heuristics score 47.7% on val without any vision | Catastrophic | C-F1 |
| 4 | `remote_setup.sh` (Python 3.9 venv) incompatible with `install.sh` (conda `unsloth`) + torch 2.10.0 known-broken | Critical | B6 / B9 |
| 5 | `camera_direction` / `satellite_arrow` sends 4 sat images but **drops the query street-view** — task is unsolvable as presented | Catastrophic | A2 / C-F7 |
| 6 | `sync.sh` ships 16 GB every invocation, uses literal `~` in `REMOTE_DIR`, no pull-back script | Catastrophic | B2 |
| 7 | No `torch.manual_seed` / `np.random.seed` / CUDA seed anywhere in training/ | Critical | A6 |
| 8 | `wandb.log(...)` unguarded at `train.py:275-277` — W&B outage kills the run | Critical | B4 |
| 9 | `BASE_MODEL` falls back to HF hub silently if local `models/Qwen3.5-4B` missing on remote | Critical | B7 |
| 10 | `MAX_STEPS=N` silently overrides `NUM_EPOCHS` (user-documented invocation is misleading) | Critical | B8 / A10 |
| 11 | Final `base_eval` in `train.py:378-385` runs the finetuned model twice and labels half "Base" | Major | B12 |
| 12 | 6 val locations are byte-identical duplicates of train locations (72 leaked records) | Major | C-F4 |
| 13 | Only 10 question templates per topic + fixed option pool → `land_use` = "Residential" 53% of the time | Major | C-F1 / C-F6 |
| 14 | `dataloader_num_workers=0` based on outdated folklore (contradicted by `eval_base.py`) — likely 2× slowdown | Major | B13 |
| 15 | Probe list `[1,2,4,8,16]` under-probes on 96 GB; no safety margin subtracted from recommended bs | Major | A9 / B14 |
| 16 | `eval_base.py` loose `parse_letter` matches any A/B/C/D char, including "Based" → B | Major | A13 |
| 17 | Answer-letter distribution is 29/29/21/21 (not 25/25/25/25); binary-mismatch topics have 50% random baseline | Critical | A3 / C-F3 |
| 18 | Only 7 label tokens unmasked (6 scaffolding + 1 letter) → loss curves don't measure learning | Critical | A4 |

---

## Findings by severity

### Catastrophic

#### 1. Missing RTX Pro 6000 GPU profile — remote will silently degrade to `safe_16gb`
- **File:** `training/config.py:32-57`
- **What's wrong:** `PROFILES` dict has no entry whose pattern (`"5090"`, `"4090"`, `"3090"`, `"a100"`, `"h100"`, `"h200"`) will match `torch.cuda.get_device_name(0)` on an RTX Pro 6000 Blackwell. Fallback is `safe_16gb` with `image_max_edge=448` (local run used 768), `lora_r=8` (local was 16), `finetune_vision_layers=False` (local trained vision), `initial_batch_guess=1`. Only signal is a single `print("WARNING: Unknown GPU ... using safe_16gb profile")`.
- **Why it matters:** The remote run trains a *different* model than the one that produced 75.3% — different token distribution (smaller images), half the adapter capacity, frozen vision layers despite TRAINING.md labeling vision unfreezing as critical. 90 GB VRAM sits idle. Hours lost, model degraded, comparison invalid.
- **Evidence:**
  ```python
  for key, pattern in [
      ("rtx_5090_32gb", "5090"), ("rtx_4090_24gb", "4090"),
      ("rtx_3090_24gb", "3090"), ("a100_80gb", "a100"),
      ("h100_80gb", "h100"), ("h200_141gb", "h200"),
  ]:
      if pattern in name: return key, PROFILES[key]
  print(f"WARNING: Unknown GPU '{name}', using safe_16gb profile")
  ```
- **Proposed fix:** Add `"rtx_pro_6000_96gb": dict(vram_gb=96, image_max_edge=1024, lora_r=32, lora_alpha=32, lr=1e-4, initial_batch_guess=16, finetune_vision_layers=True)` and a detection pattern (`"pro 6000"` or `"rtx pro"` — verify on remote first via `python -c "import torch; print(torch.cuda.get_device_name(0))"`). Replace the `safe_16gb` fallback with `sys.exit(1)` so unknown hardware is a loud failure, not a silent one.
- **Confidence:** High

#### 2. In-training eval methodology invalidates the 22 → 75.3 headline
- **Files:** `training/evaluation.py:58-61,72,124-132`; compare with `training/eval_base.py:53-61,150-164`
- **What's wrong:** `compute_topic_accuracy` (produces both the "22% base" and the "75.3% finetuned" numbers in the run summary) generates with `temperature=0.7, top_p=0.8, top_k=20` (stochastic) and scores via `generated.startswith(rec["answer"])`. `eval_base.py` generates greedy (`do_sample=False`) and scores via `parse_letter()` (word-boundary regex with fallback). The two measurements on the same base model disagree by 5.7 pp (22% vs 27.7%), and the per-topic "vs Base" column in `epoch_2_verdict.md` is incoherent — it implies base camera_direction=0% and base building_height=0%, but `base_verdict.md` shows 26.2% and 29.1%.
- **Why it matters:** The +53 pp headline is not a real delta. Under a unified greedy + lenient-parse protocol on both sides, the true delta is closer to **+47 pp** (27.7 → ~75), and the finetuned side has ~±1 pp sampling noise baked in from `temperature=0.7`. Per-topic "vs Base" numbers in epoch verdicts are noise, not signal. Shipping 53.3% will overclaim to stakeholders.
- **Evidence:**
  - `evaluation.py:58-61`: stochastic generate + `startswith` scoring.
  - `eval_base.py:154-157`: greedy + `parse_letter`.
  - `base_verdict.md` vs `epoch_2_verdict.md`: camera_direction 26.2% vs "+25.3% over base".
- **Proposed fix:** Replace stochastic generate in `evaluation.py` with `do_sample=False, pad_token_id=pad_id`; import `parse_letter` from `eval_base.py` and use it; set `padding_side="left"` on the tokenizer before generate. Re-run the full-val eval against the already-saved LoRA (`training/runs/20260417_115710_rtx_5090_32gb/lora/`) under the unified protocol to get the real delta before the remote run — gate on user approval since this is a GPU step. Report per-topic accuracy alongside per-topic *random / majority-class* baselines.
- **Confidence:** High

#### 3. Text-only heuristics already score 47.7% on val — the vision model has ≤27 pp of real signal to explain
- **Files:** `dataset_content/EODATA_compressed_final/splits_per_city/train.jsonl`, `validation.jsonl`; verified in `audit_scratch/09_heuristic_baselines.py` → `out_09_heuristic.txt`
- **What's wrong:** Several topics have extremely small option pools and strongly skewed answer distributions:
  - `road_surface`: 95.1% of train answers are "Asphalt" (shortest string). "Always pick shortest" → **94.2% val accuracy**.
  - `green_space`: 67.1% train majority. "Always pick modal string" → **78.3% val**.
  - `land_use`: 53.2% train "Residential". "Always pick modal string" → **59.5% val**.
  - `road_type`: "Always pick modal string" → **63.3% val**.
  - `urban_density`: **55.1% val** from modal string.
  - `junction_type`: **58.0% val** from modal string.
  - `building_height`: **42.1% val** from modal string.
  
  Aggregated: "pick modal option string per topic" = **47.7% val overall** with zero vision. "Pick shortest option" = 33.5% val overall. A model doing *only* label-prior memorization at SFT time walks out at ~50% — before any image understanding.
- **Why it matters:** The real visual-grounding delta between base (27.7%) and finetuned (75.3%) is at most ~27 pp (75.3 − 47.7) of *additional* accuracy beyond what the label prior gives you for free. The headline "+53 pp" is ~50% prior-memorization and ~50% vision. This is exactly the kind of finding the user asked for — "materially mislead about model quality" — and it absolutely qualifies.
- **Evidence:** See `audit_scratch/out_09_heuristic.txt` (reproduced in appendix). Key lines:
  ```
  === VAL results ===
  shortest             overall 0.335
  modal_string         overall 0.477
  best-per-topic       overall 0.474
  ...
  road_surface  val_acc=0.942  (shortest)
  green_space   val_acc=0.783  (modal_string)
  ```
- **Proposed fix (report-side, not code):** In the eval report and thesis, report per-topic accuracy **minus per-topic heuristic baseline**, not minus 25%. Flag `road_surface` as essentially unlearnable by any method (94% is a ceiling set by the data, not the model). Consider the benchmark set as a cleaner headline (class-prior bias is milder there). Dataset-side fix is to balance answer distributions and equalize option lengths — but that's a rebuild, likely out of scope for the remote run.
- **Confidence:** High

#### 4. `camera_direction` task is unanswerable as currently constructed
- **Files:** `training/data.py:70-79`; `dataset_content/EODATA_compressed_final/composite_utils.py:256-260`
- **What's wrong:** For `image_mode == "satellite_arrow"`, `convert_record` intentionally sends only the 4 arrow-overlaid satellite images (comment: "Domain expert correction: no street view query image"). But the question text asks "Based on the street-level perspective, which satellite image arrow indicates the correct camera direction?" — no street-level perspective is shown. `composite_utils.get_images_for_question` populates `result["query_sv"]` correctly; it is then discarded in `data.py`.
- **Why it matters:** With no query SV, the task is blind guessing over 4 options. Epoch-2 `camera_direction` accuracy is 25.3% — **below** the full-val greedy-eval base of 26.2%. The "domain expert correction" cited in the comment is almost certainly a miscommunication: the dataset exposes `query_stv_path` for a reason. 2000 training samples are being wasted and ~561 val samples are dragging overall accuracy down. Fixing this should lift overall accuracy 3–5 pp and finally test whether the model can learn direction grounding.
- **Evidence:**
  - `data.py:70-79`: iterates `result["options"]` only; `result["query_sv"]` is populated but never appended to `user_content`.
  - `composite_utils.py:256-260`: `result["query_sv"] = Image.open(p(query_sv_path))`.
  - `epoch_2_verdict.md`: camera_direction 25.3%; `base_verdict.md`: 26.2%.
- **Proposed fix:** After the option-image loop for `satellite_arrow`, prepend the query SV:
  ```python
  if result.get("query_sv") is not None:
      user_content.append({"type": "image", "image": resize_image(result["query_sv"], max_edge)})
  ```
  This changes `satellite_arrow` from 4 images to 5 (already the worst case the probe accounts for). Re-measure p99 tokens — still well under 8192. Confirm with the user that the "domain expert correction" comment was in error.
- **Confidence:** High

#### 5. `sync.sh` will not reliably deliver a working repo to the remote
- **File:** `sync.sh` (entire), `.gitignore:14-45`, compared with `training/install.sh:12-15`, `remote_setup.sh`
- **What's wrong:** Three interlocking problems:
  1. `REMOTE_DIR="~/training"` uses a literal tilde. rsync's tilde expansion depends on whether the remote shell expands it (usually yes in ssh, but not in every rsync version/flag combo).
  2. `--exclude` list does **not** exclude `dataset_content/` (6.9 GB) or `models/` (8.8 GB). On every invocation, `sync.sh` re-transfers ~16 GB. No resume verification, no post-transfer hash check.
  3. There is no pull-back script. `sync.sh` excludes `training/runs/` from push (correct) but nothing pulls the run artifacts (LoRA adapter, merged bf16, logs, verdict markdowns) from the remote back home. A user who forgets and releases the remote loses everything.
- **Why it matters:** On the one-shot slot, the startup cost is 15–30 min of rsync plus environment setup. A failed mid-transfer leaves a partially-synced tree that silently fails at runtime. More importantly, results may not make it back.
- **Evidence:** `sync.sh:17-36`; absence of a `pull.sh` companion; `.gitignore` excludes dataset/models from git but not from rsync.
- **Proposed fix:**
  - Replace `REMOTE_DIR="~/training"` with `REMOTE_DIR="training"` (rsync handles it relative to home) or `REMOTE_DIR="/home/$REMOTE_USER/training"` with an explicit user.
  - Stage dataset and models as separate explicit rsync calls with `rsync -avP --checksum` so re-runs are near-instant and verified.
  - Add a `pull.sh`: `rsync -avP $REMOTE_HOST:training/training/runs/ ./training/runs/`.
  - Document the initial first-sync size (~16 GB) in `TRAINING.md` so the operator budgets for it.
- **Confidence:** High

### Critical

#### 6. `remote_setup.sh` / `install.sh` / torch version — environment reproduction is broken
- **Files:** `remote_setup.sh:1-37`; `training/install.sh:12-15`; `training_live.log:1`
- **What's wrong:** Three issues:
  1. `remote_setup.sh` creates a Python 3.9 venv and references a `requirements.txt` that does not exist at the repo root (only `dataset/requirements.txt` exists). Python 3.9 + Unsloth 2026.4.2 is very likely incompatible.
  2. `training/install.sh` exits if `CONDA_DEFAULT_ENV != "unsloth"`. The scripted remote path creates a venv, never a conda env. The two are incompatible; there is no scripted conda-env-reproduction path.
  3. Local run's `training_live.log:1` begins with: *"Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.10.0+cu128)."* The entire local run was done on a known-broken torch version. Cpp extensions (likely `cut_cross_entropy`) silently disabled. If remote pip resolves a different torch version, Unsloth's precompiled Triton kernels may fail to load.
- **Why it matters:** The remote will either crash on `python train.py` (missing deps) or — if someone manually hacks the env — run on a drifted torch + unsloth combination that produces different training behavior. This risks wasting the queue slot on setup.
- **Proposed fix:** Freeze the exact local conda env: `conda env export -n unsloth > environment.yml` (strip builds for portability), commit, and change `remote_setup.sh` to `conda env create -f environment.yml`. Drop the Python 3.9 venv path entirely. Decide deliberately whether to upgrade torch to ≥2.11.0 or stay on the known-working 2.10.0+cu128 — and pin it. Verify RTX Pro 6000 Blackwell (sm_120) is CUDA 12.8+ compatible (it is; just confirm the driver).
- **Confidence:** High

#### 7. No deterministic seeding of torch / numpy / CUDA
- **Files:** `training/train.py`, `training/config.py`, `training/evaluation.py`
- **What's wrong:** Grep across `training/` finds zero calls to `torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy.random.seed`, or `torch.use_deterministic_algorithms`. Only `SFTConfig(seed=SEED)` is set, which seeds HF Trainer's dataloader sampler but not model RNG or CUDA kernels. Combined with stochastic in-training eval (finding #2), nothing is reproducible.
- **Why it matters:** The one-shot remote run cannot be reliably compared to the local run even with identical code + data + hyperparams. Per-epoch "improving by 4 pp" might be ~±2 pp noise.
- **Proposed fix:** At `main()` top in `train.py`:
  ```python
  import random, numpy as np, torch
  random.seed(SEED); np.random.seed(SEED)
  torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
  ```
  Combined with the fix to #2 (greedy eval), the full run becomes reproducible modulo CUDA non-determinism — acceptable.
- **Confidence:** High

#### 8. `wandb.log(...)` at train.py:275-277 is unguarded
- **File:** `training/train.py:275-277`
- **What's wrong:** `wandb.init()` runs unconditionally when `REPORT_TO=wandb` (`train.py:141-155`). Local run used `/home/ezel/.netrc` for auth. Remote has no `.netrc` (`sync.sh` pushes from repo root, not `$HOME`). No `WANDB_API_KEY` in env, no `wandb login` step in `remote_setup.sh`. Further, `wandb.log({"baseline/accuracy": ...}, step=0)` is NOT inside try/except, so a W&B outage mid-run (common) crashes training.
- **Why it matters:** Silent blocker. The operator runs with `REPORT_TO=wandb` (matching TRAINING.md and the local run), training hangs on an interactive W&B login prompt or crashes at step 0.
- **Proposed fix:** Wrap the baseline `wandb.log` in try/except (matches the pattern in `callbacks.py:167-190`). Default `REPORT_TO=none` unless `WANDB_API_KEY` is set in env. Add a `WANDB_MODE=offline` option for air-gapped lab boxes. Document: `wandb login` (or export `WANDB_API_KEY`) as a prerequisite in `remote_setup.sh`.
- **Confidence:** High

#### 9. `BASE_MODEL` silently falls back to HF hub if local copy missing
- **Files:** `training/config.py:22-26`; `training/eval_base.py:215-216`
- **What's wrong:** `BASE_MODEL = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else "unsloth/Qwen3.5-4B"`. If the 8.8 GB `models/Qwen3.5-4B/` directory doesn't make it to the remote (fragile via `sync.sh`), `FastVisionModel.from_pretrained("unsloth/Qwen3.5-4B")` triggers an uncontrolled HF Hub fetch: requires internet, possibly HF token, and the hub snapshot may drift from the local one.
- **Why it matters:** Different snapshot = different base model = breaks the local-vs-remote comparison. Air-gapped lab = crash at startup. Mid-download failure = corrupt cache that needs manual cleanup.
- **Proposed fix:** Change the fallback to raise if local is missing and HF is unreachable. Require `BASE_MODEL=/absolute/path` via env var in `remote_setup.sh`. Fail fast on mismatch.
- **Confidence:** High

#### 10. `MAX_STEPS` silently overrides `NUM_EPOCHS` — the documented invocation misleads
- **Files:** `training/train.py:297-323`; `training/TRAINING.md:152`
- **What's wrong:** TRAINING.md example: `MAX_STEPS=400 NUM_EPOCHS=2 LEARNING_RATE=1e-4 REPORT_TO=wandb python train.py`. Actual behavior: `num_train_epochs=1` (silently set when `max_steps > 0`), `max_steps=400` wins, NUM_EPOCHS is ignored. LR scheduler length is scaled to 400 steps. Anyone running the documented command expects 2 epochs and gets ~35% of 1.
- **Why it matters:** Operator confusion. A test-cap MAX_STEPS invocation on the remote silently truncates training.
- **Proposed fix:** Either make MAX_STEPS and NUM_EPOCHS mutually exclusive (error if both set to non-defaults) or document explicitly. Prefer removing MAX_STEPS entirely in favor of NUM_EPOCHS + SMOKE_TEST.
- **Confidence:** High

#### 11. Answer-letter distribution is ~29/29/21/21, not 25/25/25/25
- **Files:** Verified via `audit_scratch/out_06_bias.txt`
- **What's wrong:** Across `splits_per_city/val`: A=29.3%, B=29.2%, C=20.3%, D=21.2%. The skew is driven by binary `mismatch_binary_*` topics (only A/B, 4000 records) pulling A/B up. True "always pick A" baseline on val is **29.3%**, not 25%.
- **Why it matters:** The "25% random" floor cited in all reporting is wrong. The "base at 22%" number is *below* random for this dataset. A finetuned model that learned nothing beyond "output the most frequent letter" scores 29%, not 25%. Combine with finding #3 (47.7% heuristic) and the real minimum bar for "model learned vision" is ~50% on val, not 25%.
- **Proposed fix:** Report per-topic random + majority-class baselines in all eval writeups. Update the `"vs Base"` column in `_write_verdict` (`callbacks.py`) to include `"vs random"` and `"vs majority"` columns.
- **Confidence:** High

#### 12. Only 7 label tokens unmasked — loss curves don't measure learning
- **Files:** `training/train.py:87-106`; `training/TRAINING.md:101-106`; verified by label-masking log line
- **What's wrong:** `train_on_responses_only=True` with boundaries `<|im_start|>user\n` and `<|im_start|>assistant\n` correctly unmasks `<think>\n\n</think>\n\nA<|im_end|>\n` — 7 tokens, 6 of which are fixed scaffolding with ~0 entropy after step ~10. Only the letter token carries signal. Final train loss 0.0939 averaged over 7 tokens ≈ 0.65 on just the letter (~52% correct-letter probability).
- **Why it matters:** You cannot tell from loss whether the model is learning vision or the label prior (finding #3). This is why the model reaches loss 0.13 at step 10 — scaffolding memorization, not learning.
- **Proposed fix:** Add a text-only baseline eval (blank/zero images at inference) during training to quantify the non-visual contribution to accuracy. Optionally add a short CoT-style rationale before the letter in the assistant target so more letter-correlated tokens enter the loss.
- **Confidence:** High

### Major

#### 13. `train.py:378-385` — final `base_eval` runs the finetuned model twice
- **File:** `training/train.py:378-385`
- **What's wrong:** Both `base_eval` and `ft_eval` call `run_eval_samples(model, ...)` on the *same* fully-finetuned `model` object. The comment `# Re-use baseline for base results display` is misleading — nothing baseline-related happens. `eval_samples.md` from the local run shows Base == Finetuned == gold on all 3 samples, which contradicts the full-eval 27.7% base accuracy.
- **Why it matters:** Anyone spot-checking `eval_samples.md` sees fake agreement. Erodes reporting trust.
- **Proposed fix:** Delete the block; `eval_base.py` already does proper base eval. Or cache 3 base-eval results before `get_peft_model()` and reference them after training.
- **Confidence:** High

#### 14. 72 validation records at 6 byte-identical train locations
- **Files:** `splits_per_city/train.jsonl` + `validation.jsonl`; verified via `audit_scratch/04_image_hashes.py` and `05_dup_impact.py`
- **What's wrong:** 6 pairs of sample_ids share byte-identical satellite PNG + all 4 streetview JPGs, with lat/lon matching to ~6 decimals, yet live in different splits (`buenos_aires_0046↔0032`, `kayseri_0052↔0070`, `reykjavik_0029↔0062`, `samsun_0030↔0072`, `stockholm_0029↔0013`, `tallinn_0022↔0010`). 72 val records (1% of val) sit at these 6 locations. 3 have identical question+answer in train.
- **Why it matters:** Direct pixel leakage. Effect on overall val accuracy is ~±1 pp — small but unambiguous cheating. Indicates a dedup bug in `dataset/splitting/`. `splits_seen_unseen` is clean (0 overlaps).
- **Proposed fix:** In splitting code, dedupe by (lat, lon rounded to ~5 m) OR by satellite-image SHA256 before split assignment. Move all 72 val records to train (or drop). For the current remote run, this is acceptable to ignore (±1 pp noise), but note it in the reporting.
- **Confidence:** High

#### 15. Only 10 question templates per topic, no val-only templates
- **Files:** `audit_scratch/out_11_qtext.txt`
- **What's wrong:** Each of 14 topics has exactly 10 unique question strings. Val reuses the same 10. Combined with the fixed option pool (finding #3), the model is effectively memorizing `image + topic → fixed string`.
- **Why it matters:** Explains fast convergence and high apparent accuracy. Not leakage, but a structural weakness.
- **Proposed fix:** Paraphrase templates at training time; reserve ~2 templates per topic for val only. Out of scope for the current one-shot; flag for v2.
- **Confidence:** High

#### 16. `dataloader_num_workers=0` likely costs ~2× speed
- **Files:** `training/train.py:337`; `training/TRAINING.md:192`
- **What's wrong:** TRAINING.md claims `num_workers > 0` breaks because "PIL Images can't be pickled." But `training/eval_base.py` uses `NUM_WORKERS=8` with the same PIL-returning dataset and works fine. Folklore, not verified.
- **Why it matters:** Main-thread image decode + LANCZOS resize bottlenecks training. Local RTX 5090 run: 19 sec/step. On Pro 6000 (compute ~1.5–2× faster), if the data path is CPU-limited, no speedup. A 12-hour run becomes 6 hours with workers.
- **Proposed fix:** Before the remote run, on smoke_test: try `dataloader_num_workers=4, dataloader_persistent_workers=True`. If it works (it will), use it on the remote. If not, document the exact error on `train.py:337`.
- **Confidence:** Medium

#### 17. Probe only tests bs ∈ {1,2,4,8,16}; no safety margin
- **File:** `training/train.py:43-84, 225-229`
- **What's wrong:** `probe_batch_size` caps at 16 and does not subtract a safety margin from the recommended batch size. Activation memory during a real training step with Unsloth's gradient checkpointing + adamw_8bit optimizer state can exceed the probe's single-step estimate by 15–30%, especially on worst-case 5-image `satellite_arrow` batches. Recent commit `f02bdcc` specifically fixed an OOM here, suggesting the logic is fragile.
- **Why it matters:** OOM at step 800 on a 12-hour remote run means losing the slot.
- **Proposed fix:** Extend probe list to `[1, 2, 4, 8, 16, 24, 32]`. After probe, subtract one size-tier or multiply by 0.8: `batch_size = max(1, int(recommended * 0.8))`. Probe worst-case 5-image batches twice (covers optimizer-state peak).
- **Confidence:** Medium

#### 18. `eval_base.parse_letter` loose fallback matches any A/B/C/D char
- **File:** `training/eval_base.py:53-61`
- **What's wrong:** Fallback regex `[ABCD]` (no word boundary) matches "B" inside "Based" or "By", "A" inside "Answer" (which is before the actual letter). This can silently miscount base-model rambling responses.
- **Why it matters:** Explains part of why base is reported at 27.7% — likely over-counts false B's. Real base might be 30–32%.
- **Proposed fix:** Tighten fallback to `\b[ABCD][\.\):\s]` (letter followed by punctuation/space/eol), or drop the loose fallback entirely and mark unparseable answers wrong.
- **Confidence:** Medium

#### 19. Per-topic answer imbalance in val (e.g., amenity_richness 31% B)
- **Files:** `audit_scratch/out_06_bias.txt`
- **What's wrong:** Val per-topic majority class is 26–32% (not 25%): `amenity_richness` B=31.4%, `land_use` D=28.2%, `junction_type` B=27.9%, etc.
- **Why it matters:** Per-topic "75%" numbers look impressive against a 25% random baseline but less so against the 28–31% majority-class baseline. Reporting needs to show both.
- **Proposed fix:** Add per-topic majority baselines to `_write_verdict`.
- **Confidence:** High

#### 20. Doc drift: TRAINING.md says bs=4 × grad_accum=6; the run actually used bs=8 × grad_accum=3
- **Files:** `training/TRAINING.md:77-79`; `training/runs/20260417_115710_rtx_5090_32gb/summary.txt:8`
- **What's wrong:** Documentation says peak VRAM 17 GB at bs=4. Actual run was bs=8, peak 20 GB. If an operator reads TRAINING.md to pick a remote profile, they'll start wrong (though the probe will correct).
- **Proposed fix:** Update TRAINING.md from `summary.txt`.
- **Confidence:** High

### Minor

- **21.** `training_live.log:37` "Flash Attention 2 installation seems broken. Using Xformers instead." — ~20% slower on long sequences. Install a FA2 wheel matching CUDA 12.8 / torch 2.10 on the remote. (Minor)
- **22.** `train.py:28-32` imports `config` (pulling torch) before `import unsloth` — produces "Unsloth should be imported first" warning. Minor slowdown risk. Move `import unsloth` to the very top.
- **23.** `warmup_steps = min(40, total_steps // 10)` floor-at-40 rule: for runs > 400 steps, warmup is <10% (contradicts TRAINING.md). Switch to `max(40, total_steps // 10)` if the remote run is ≥3 epochs.
- **24.** `torch.manual_seed` not set inside `run_eval_samples` / `compute_topic_accuracy` — even after fixing #2 to greedy, any sampling path needs the seed. Add `torch.manual_seed(SEED)` inside each eval function.
- **25.** `evaluation.py:72` uses `startswith(rec["answer"])`, which yields false positives ("Apple" startswith "A") and false negatives ("The answer is A" doesn't startswith "A"). Subsumed by fix #2 (switch to `parse_letter`).

---

## Validated-clean

Things the audit checked and found good:

- **Label masking is correct.** `train_on_responses_only=True` with ChatML boundaries unmasks exactly the 7 tokens documented in TRAINING.md.
- **No train/val sample_id or question_id overlap** in either split strategy.
- **`splits_seen_unseen`** is cleanly geographically separated (32 train cities / 8 val cities, 0 coordinate proximity).
- **Benchmark is disjoint** from train+val by sample_id and question_id (modulo one hash collision that is a false positive from coord rounding).
- **No hardcoded `/home/ezel/` paths** in any `.py`, `.sh`, or `.md` under `training/` (all in generated run artifacts).
- **All image paths in train/val/benchmark JSONL resolve** — zero missing files, zero files <5 KB (across 264,560 image references in `splits_per_city`).
- **`image_mode → image count` is internally consistent** for all records.
- **Mismatch yes/no balance** is ~50/50 across train, val, benchmark.
- **Option shuffle dedup** (commit `1b8a529`) verified: (sample_id, topic) is unique in both splits.
- **LoRA config** `rank=16, alpha=16, target_modules="all-linear", finetune_vision_layers=True`, `load_in_4bit=False, load_in_16bit=True` — correctly passed.
- **`max_seq_length=8192`** has ~4× headroom over measured p99.
- **Filenames are NOT passed to the model** (typed content parts only, no path-OCR vector).
- **`composite_utils.get_images_for_question`** correctly populates `query_sv` for `satellite_arrow` — the bug is in `data.py` dropping it, not in the data.
- **Schema consistency:** all records have the full expected field set.

---

## Open questions for the user

1. **The `camera_direction` "domain expert correction" in `data.py:71-72`** — is the comment historically accurate or was it a mistake? The code effect contradicts the dataset's stated intent (`query_stv_path` is populated). Decide before remote launch.
2. **Should the remote run use `splits_per_city` (as local) or `splits_seen_unseen`?** The latter is a stronger test of generalization (geographic disjoint) and has zero leakage. The 75.3% was on `splits_per_city`; running on `splits_seen_unseen` will produce a lower but more defensible number.
3. **Acceptable to re-run full-val eval on the saved local LoRA adapter** (`training/runs/20260417_115710_rtx_5090_32gb/lora/`) under the unified greedy+`parse_letter` protocol before the remote run? This is a GPU step (~15 min on RTX 5090) and would give the corrected local delta — a useful calibration point.
4. **Keep W&B for the remote run** (requires `WANDB_API_KEY` on remote) or run locally-offline?
5. **Rank-16 LoRA on local → rank-32 on remote** (Pro 6000 has the VRAM) — worth the change, or keep rank-16 for a clean comparison?
6. **Decide on torch version:** keep the known-broken 2.10.0+cu128 for exact parity with the local run, or upgrade to ≥2.11.0 as the warning recommends?

---

## Appendix — key programmatic-check outputs

### Heuristic baseline on val (`audit_scratch/out_09_heuristic.txt`)
```
=== VAL results ===
shortest             overall 0.335
longest              overall 0.255
modal_letter         overall 0.292
modal_string         overall 0.477
best-per-topic       overall 0.474

Per-topic best-heuristic val accuracy:
  amenity_richness      0.342   (longest)
  building_height       0.421   (modal_string)
  camera_direction      0.264   (shortest)
  green_space           0.783   (modal_string)
  junction_type         0.580   (modal_string)
  land_use              0.595   (modal_string)
  mismatch_binary_easy  0.537   (shortest)
  mismatch_binary_hard  0.485   (shortest)
  mismatch_mcq_easy     0.260   (modal_string)
  mismatch_mcq_hard     0.239   (modal_letter)
  road_surface          0.942   (shortest)
  road_type             0.633   (modal_string)
  transit_density       0.299   (longest)
  urban_density         0.551   (modal_string)
```

### Answer-letter distribution (`audit_scratch/out_06_bias.txt`)
```
=== splits_per_city/val (6984) ===
Overall answer dist: A=0.293, B=0.292, C=0.203, D=0.212
Majority-class baseline: 0.293
```

### Finetuned vs base reality check
```
epoch_2_verdict.md  : base=22.0%  finetuned=75.3%  delta=+53.3
base_verdict.md     : base=27.7%                   (full val, greedy, lenient)
Per-topic contradiction — camera_direction:
  epoch_1_verdict.md: "26.0%  vs Base +26.0%" (implies base=0%)
  base_verdict.md:   base=26.2%
Per-topic contradiction — building_height:
  epoch_1_verdict.md: "64.0%  vs Base +64.0%" (implies base=0%)
  base_verdict.md:   base=29.1%
```

### 72-record image-byte leakage (`audit_scratch/out_05_dup_impact.txt`)
```
val records at duplicated locations: 72 / 6984
6 pairs:
  val=buenos_aires_0046 ↔ train=buenos_aires_0032
  val=kayseri_0052      ↔ train=kayseri_0070
  val=reykjavik_0029    ↔ train=reykjavik_0062
  val=samsun_0030       ↔ train=samsun_0072
  val=stockholm_0029    ↔ train=stockholm_0013
  val=tallinn_0022      ↔ train=tallinn_0010
eq_question+eq_answer=3  eq_question+diff_answer=2  diff_question=41
```

### Missing GPU profile (`training/config.py:32-57`)
```
PROFILES keys: rtx_5090_32gb, rtx_4090_24gb, rtx_3090_24gb,
               a100_80gb, h100_80gb, h200_141gb, safe_16gb
Detection patterns: "5090","4090","3090","a100","h100","h200"
(no pattern matches "RTX PRO 6000")
Fallback: safe_16gb → image_max_edge=448, lora_r=8, finetune_vision=False
```

All scratch analysis scripts and outputs are preserved under `/home/ezel/Development/EOLLM/audit_scratch/`.
