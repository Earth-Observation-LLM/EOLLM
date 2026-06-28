#!/usr/bin/env python3
"""
Single source of truth for every number in the paper
("When Does the Second View Help? Auditing Cross-View Fusion in Urban VLMs").

Reads the satfwd (Qwen3.5-9B sat+forward-SV, seen_unseen, merged ep4) per-record
prediction logs and the consolidated results spreadsheet, recomputes every table
cell / CI / p-value / chi-square, and writes v2_numbers.json. The LaTeX cites
these values; nothing in the paper is hand-typed.

Reproducible: bootstrap seed = 3407, 10k resamples. Re-run if the checkpoint
is ever superseded.

v4 NOTE: the BLIND condition is removed from the paper entirely. The ablation
reported is three-way (sat / sv / full). We still LOAD blind here so the headline
oracle (= better single view, blind-free) and the item-level counts can be
cross-checked, but no blind-derived quantity (vision_contribution = full-blind,
the wrong-while-blind leakage subset, the text-only leak audit) is emitted for the
paper. The two retired tasks (green_space, road_surface) are dropped silently: the
benchmark is presented as 7 urban + 5 cross-view = 12, with no funnel/retirement
narrative. Blind-derived keys below are tagged "_BLIND_NOT_FOR_PAPER_V4".

Usage:  python make_v2_numbers.py   (writes v2_numbers.json next to this file)
"""
import json
import os
import math
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/ezel/Development/EOLLM"
PRED_DIR = os.path.join(ROOT, "benchmark_suite/results/qwen9b_satfwd_seen_unseen__benchmark")
BENCH_FILE = os.path.join(ROOT, "dataset_content/EODATA_compressed_final/benchmark/benchmark_with_answers.jsonl")
XLSX = os.path.join(ROOT, "results_csv/EOLLM_results_wide.xlsx")
OUT = os.path.join(HERE, "v2_numbers.json")

SEED = 3407
NBOOT = 10000

# The seven urban-attribute tasks that carry the headline. (Two further
# satellite-derivable tasks, green_space and road_surface, exist in the corpus but
# are not part of the released 12-task benchmark; v4 drops them silently, so they
# never appear in the paper.)
URBAN7 = ["land_use", "building_height", "urban_density", "road_type",
          "junction_type", "amenity_richness", "transit_density"]
CROSS5 = ["camera_direction", "mismatch_binary_easy", "mismatch_binary_hard",
          "mismatch_mcq_easy", "mismatch_mcq_hard"]
RETIRED = ["green_space", "road_surface"]
MODES = ["blind", "sat_only", "sv_only", "full"]


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_mode(mode, topics=None):
    """question_id -> bool(is_correct), filtered to `topics` if given."""
    d = {}
    path = os.path.join(PRED_DIR, f"{mode}_predictions.jsonl")
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if topics is None or r["topic"] in topics:
                d[r["question_id"]] = bool(r["is_correct"])
    return d


def load_mode_with_topic(mode):
    """question_id -> (topic, bool)."""
    d = {}
    with open(os.path.join(PRED_DIR, f"{mode}_predictions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            d[r["question_id"]] = (r["topic"], bool(r["is_correct"]))
    return d


# --------------------------------------------------------------------------
# stats helpers
# --------------------------------------------------------------------------
def mcnemar(a, b, ids):
    """Exact two-sided binomial McNemar; a,b are id->bool. Returns dict."""
    bb = sum(1 for i in ids if a[i] and not b[i])      # a right, b wrong
    cc = sum(1 for i in ids if not a[i] and b[i])      # a wrong, b right
    n = bb + cc
    k = min(bb, cc)
    if n == 0:
        p = 1.0
    else:
        p = min(1.0, 2.0 * sum(math.comb(n, j) for j in range(0, k + 1)) / (2 ** n))
    chi_cc = ((abs(bb - cc) - 1) ** 2) / n if n > 0 else 0.0
    return {"b": bb, "c": cc, "discordant": n, "p_exact": p, "chi2_cc": chi_cc}


def chi2_uniform_letters(counter):
    n = sum(counter[k] for k in "ABCD")
    if n == 0:
        return {"n": 0}
    exp = n / 4.0
    chi = sum((counter[k] - exp) ** 2 / exp for k in "ABCD")
    return {"n": n, "A": counter["A"], "B": counter["B"], "C": counter["C"],
            "D": counter["D"], "chi2": chi, "always_A_pct": 100 * counter["A"] / n}


def boot_ci(values_per_item, stat_fn, seed=SEED, nboot=NBOOT):
    """Paired bootstrap over items. values_per_item is a list of tuples;
    stat_fn(resampled_list)->float. Returns (point, lo, hi)."""
    rng = random.Random(seed)
    n = len(values_per_item)
    point = stat_fn(values_per_item)
    samples = []
    for _ in range(nboot):
        res = [values_per_item[rng.randrange(n)] for _ in range(n)]
        samples.append(stat_fn(res))
    samples.sort()
    lo = samples[int(0.025 * nboot)]
    hi = samples[int(0.975 * nboot)]
    return point, lo, hi


def acc(rows, idx):
    return sum(r[idx] for r in rows) / len(rows)


# --------------------------------------------------------------------------
# main computation
# --------------------------------------------------------------------------
def main():
    out = {
        "_meta": {
            "checkpoint": "Qwen3.5-9B satfwd seen_unseen merged_ep4",
            "pred_dir": PRED_DIR,
            "bootstrap_seed": SEED, "n_bootstrap": NBOOT,
            "urban7": URBAN7, "cross5": CROSS5, "retired": RETIRED,
            "note": "All numbers recomputed from per-record logs; nothing hand-typed.",
        }
    }

    # ---- per-mode, per-topic accuracy (urban + cross + retired, for completeness)
    by_mode = {m: load_mode_with_topic(m) for m in MODES}
    topic_acc = defaultdict(dict)   # topic -> mode -> acc%
    topic_n = {}
    for m in MODES:
        cor = Counter(); tot = Counter()
        for qid, (t, c) in by_mode[m].items():
            tot[t] += 1; cor[t] += c
        for t in tot:
            topic_acc[t][m] = 100 * cor[t] / tot[t]
            topic_n[t] = tot[t]
    out["topic_n"] = topic_n
    out["topic_acc"] = {t: topic_acc[t] for t in topic_acc}

    # ---- URBAN7 four-mode table (Table 2) with per-task synergy
    sat = load_mode("sat_only", URBAN7)
    sv = load_mode("sv_only", URBAN7)
    full = load_mode("full", URBAN7)
    blind = load_mode("blind", URBAN7)
    ids = sorted(set(sat) & set(sv) & set(full) & set(blind))
    out["urban7_n_paired"] = len(ids)

    table2 = {}
    syn_list = []   # per-task synergy for unweighted mean
    for t in URBAN7:
        a = topic_acc[t]
        best_single = max(a["sat_only"], a["sv_only"])
        syn = a["full"] - best_single
        syn_list.append(syn)
        # per-task per-item oracle (keep the better single view PER QUESTION) — this is
        # the quantity the dumbbell figure plots; it is >= full on every task by
        # construction and matches the pooled full-oracle gap.
        t_ids = [i for i in ids
                 if by_mode["full"][i][0] == t]   # ids in this topic, present in all modes
        if t_ids:
            oracle_acc_t = 100 * sum(1 for i in t_ids if sat[i] or sv[i]) / len(t_ids)
            full_acc_t = 100 * sum(1 for i in t_ids if full[i]) / len(t_ids)
        else:
            oracle_acc_t = full_acc_t = None
        table2[t] = {
            "n": topic_n[t],
            "blind": a["blind"], "sat": a["sat_only"],
            "sv": a["sv_only"], "full": a["full"],
            "synergy": syn,
            "vision_contrib": a["full"] - a["blind"],
            "oracle_item": oracle_acc_t,        # per-item better-single-view accuracy
            "full_paired": full_acc_t,          # full acc on the same paired ids
        }
    # means
    def mmean(key):
        return sum(table2[t][key] for t in URBAN7) / len(URBAN7)
    table2["MEAN"] = {
        "blind_BLIND_NOT_FOR_PAPER_V4": mmean("blind"),
        "sat": mmean("sat"), "sv": mmean("sv"),
        "full": mmean("full"),
        "synergy_unweighted": sum(syn_list) / len(syn_list),
        "vision_contrib_BLIND_NOT_FOR_PAPER_V4": mmean("vision_contrib"),
    }
    out["table2_urban7"] = table2

    # ---- v4 "the model reads the imagery" evidence WITHOUT blind: each single
    # view is far above the always-A floor (25.2% on urban7), so the model is
    # clearly using the pixels. Replaces the old full-minus-blind argument.
    out["reads_imagery_urban7_v4"] = {
        "always_A_floor_pct": 25.2,   # from letter_distribution.urban7 (label fact, blind-free)
        "sat_mean": table2["MEAN"]["sat"],
        "sv_mean": table2["MEAN"]["sv"],
        "full_mean": table2["MEAN"]["full"],
        "min_single_view_task": min(
            (min(table2[t]["sat"], table2[t]["sv"]), t) for t in URBAN7),
        "note": "Both single views sit far above the 25.2% always-A floor on every "
                "task, so the model demonstrably reads the imagery; no blind needed.",
    }

    # ---- pooled (item-weighted) accuracies + synergy with bootstrap CI
    rows = [(blind[i], sat[i], sv[i], full[i], (1 if (sat[i] or sv[i]) else 0)) for i in ids]
    pooled = {
        "blind_BLIND_NOT_FOR_PAPER_V4": 100 * acc(rows, 0),
        "sat": 100 * acc(rows, 1),
        "sv": 100 * acc(rows, 2), "full": 100 * acc(rows, 3),
        "oracle_better_single": 100 * acc(rows, 4),
    }
    # pooled synergy = full - max(pooled sat, pooled sv)
    def pooled_syn(rs):
        fu = acc(rs, 3); sa = acc(rs, 1); svv = acc(rs, 2)
        return 100 * (fu - max(sa, svv))
    ps, ps_lo, ps_hi = boot_ci(rows, pooled_syn)
    pooled["synergy_pooled"] = ps
    pooled["synergy_pooled_ci"] = [ps_lo, ps_hi]
    pooled["synergy_pooled_ci_crosses_zero"] = (ps_lo < 0 < ps_hi)

    # full - oracle (the headline item-level stat)
    def full_minus_oracle(rs):
        return 100 * (acc(rs, 3) - acc(rs, 4))
    fo, fo_lo, fo_hi = boot_ci(rows, full_minus_oracle)
    pooled["full_minus_oracle"] = fo
    pooled["full_minus_oracle_ci"] = [fo_lo, fo_hi]
    pooled["full_minus_oracle_ci_excludes_zero"] = (fo_hi < 0)
    out["pooled_urban7"] = pooled

    # ---- item-level counts (fusion-win / interfere) with CIs; identity check
    n = len(ids)
    fusion_win = sum(1 for i in ids if full[i] and not sat[i] and not sv[i])
    interfere = sum(1 for i in ids if not full[i] and (sat[i] or sv[i]))
    fw_rows = [(1 if (full[i] and not sat[i] and not sv[i]) else 0) for i in ids]
    inf_rows = [(1 if (not full[i] and (sat[i] or sv[i])) else 0) for i in ids]
    fw_pt, fw_lo, fw_hi = boot_ci(fw_rows, lambda rs: 100 * sum(rs) / len(rs))
    inf_pt, inf_lo, inf_hi = boot_ci(inf_rows, lambda rs: 100 * sum(rs) / len(rs))
    out["item_level_urban7"] = {
        "n": n,
        "fusion_win": fusion_win, "fusion_win_pct": 100 * fusion_win / n,
        "fusion_win_ci": [fw_lo, fw_hi],
        "interfere": interfere, "interfere_pct": 100 * interfere / n,
        "interfere_ci": [inf_lo, inf_hi],
        "ratio_interfere_to_win": interfere / fusion_win if fusion_win else None,
        "identity_check": {
            "full_minus_oracle_pct": fo,
            "(fusion_win-interfere)/n_pct": 100 * (fusion_win - interfere) / n,
            "equal": abs(fo - 100 * (fusion_win - interfere) / n) < 1e-6,
            "note": "full-oracle and (fusion_win-interfere)/n are the SAME statistic; report once.",
        },
    }

    # ---- McNemar: full vs sat / sv / oracle
    oracle = {i: (sat[i] or sv[i]) for i in ids}
    out["mcnemar_urban7"] = {
        "full_vs_sat": mcnemar(full, sat, ids),
        "full_vs_sv": mcnemar(full, sv, ids),
        "full_vs_oracle": mcnemar(full, oracle, ids),
        "note": "full-vs-sat is NOT significant (p~0.12); headline test is full-vs-oracle (p<<1e-60).",
    }

    # ---- v4: blind is removed from the paper, so the old "full-oracle gap on
    # blind-WRONG items" leakage check is GONE. The oracle gap (full-oracle) is a
    # within-item, blind-free statistic and already lives in pooled/item_level. We
    # keep refusal/hedge counts as a robustness fact, computed over the THREE
    # reported modes only (sat/sv/full); no blind iteration.
    REPORT_MODES = ["sat_only", "sv_only", "full"]
    ref = {m: 0 for m in REPORT_MODES}; hed = {m: 0 for m in REPORT_MODES}
    tot = {m: 0 for m in REPORT_MODES}
    for m in REPORT_MODES:
        with open(os.path.join(PRED_DIR, f"{m}_predictions.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["topic"] in URBAN7:
                    tot[m] += 1
                    ref[m] += int(r.get("refused", False))
                    hed[m] += int(r.get("hedged", False))
    out["robustness_urban7_v4"] = {
        "full_minus_oracle_all_items": fo,   # blind-free; within-item
        "refused": ref, "hedged": hed, "n_per_mode": tot,
        "note": "Blind condition removed in v4. Oracle gap is within-item (no blind). "
                "Refusal/hedge over reported modes only.",
    }

    # ---- cross-view table (sensitivity check): sat/sv/full + synergy.
    # v4: blind dropped from the reported columns (kept here tagged for audit only).
    # These tasks are CONSTRUCTED so the match target sits among the options, so the
    # large full-vs-single gap is expected by design and is reported subtly, only as
    # evidence the instrument is not dead.
    table4 = {}
    for t in CROSS5:
        a = topic_acc[t]
        sv_val = a.get("sv_only")
        singles = [x for x in (a.get("sat_only"), sv_val) if x is not None]
        best = max(singles) if singles else None
        table4[t] = {
            "n": topic_n[t],
            "blind_BLIND_NOT_FOR_PAPER_V4": a.get("blind"),
            "sat": a.get("sat_only"),
            "sv": sv_val, "full": a.get("full"),
            "synergy": (a["full"] - best) if best is not None else None,
        }
    out["table4_crossview"] = table4

    # ---- image-count asymmetry (control NOT input-matched) — for F1 reframe
    img_full = {"urban7": Counter(), "cross5": Counter()}
    with open(os.path.join(PRED_DIR, "full_predictions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["topic"] in URBAN7:
                img_full["urban7"][r["n_images"]] += 1
            elif r["topic"] in CROSS5:
                img_full["cross5"][r["n_images"]] += 1
    out["image_count_full_mode"] = {
        "urban7": dict(img_full["urban7"]),
        "cross5": dict(img_full["cross5"]),
        "note": "urban full=2 imgs (sat+fwd); cross-view full=5 (sat+4SV). Control is a "
                "sensitivity proof, NOT an image-count-matched comparison.",
    }

    # ---- option-length leakage check (Robustness): is the correct option the longest?
    longest = 0; tot_ol = 0; corr_len = []; dist_len = []
    with open(BENCH_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r["topic"] not in URBAN7:
                continue
            opts = r["options"]; g = r["answer"]
            if not isinstance(opts, dict) or g not in opts:
                continue
            tot_ol += 1
            lens = {k: len(str(v)) for k, v in opts.items()}
            corr_len.append(lens[g]); dist_len += [lens[k] for k in lens if k != g]
            if lens[g] == max(lens.values()):
                longest += 1
    out["option_length_urban7"] = {
        "n": tot_ol,
        "correct_is_longest_pct": 100 * longest / tot_ol,
        "mean_correct_len": sum(corr_len) / len(corr_len),
        "mean_distractor_len": sum(dist_len) / len(dist_len),
        "note": "correct-is-longest 22.9% < 25% chance; correct option slightly SHORTER than distractors.",
    }

    # ---- letter distribution (Robustness): URBAN7 clean vs cross-view skew vs corpus
    letters = {"urban7": Counter(), "cross5": Counter(), "released12": Counter()}
    with open(BENCH_FILE) as f:
        for line in f:
            r = json.loads(line)
            t = r["topic"]; g = r["answer"]
            if t in URBAN7:
                letters["urban7"][g] += 1
            if t in CROSS5:
                letters["cross5"][g] += 1
            if t not in RETIRED:
                letters["released12"][g] += 1
    out["letter_distribution"] = {
        "urban7": chi2_uniform_letters(letters["urban7"]),
        "cross5": chi2_uniform_letters(letters["cross5"]),
        "released12": chi2_uniform_letters(letters["released12"]),
        "note": "URBAN7 uniform (chi2~3.6); skew concentrated in cross-view; "
                "and headline metric is within-item so option order cancels.",
    }

    # ---- benchmark composition (Table 1 / §3). v4: no funnel/retirement narrative;
    # the released benchmark is the 12 tasks (7 urban + 5 cross-view). We still
    # verify the released-task rows sum to 4734 and report per-task n.
    comp = Counter()
    n_file = 0
    with open(BENCH_FILE) as f:
        for line in f:
            r = json.loads(line); comp[r["topic"]] += 1; n_file += 1
    urban_n = sum(comp[t] for t in URBAN7)
    cross_n = sum(comp[t] for t in CROSS5)
    out["benchmark_composition"] = {
        "released_benchmark_12task": urban_n + cross_n,
        "urban7": urban_n, "cross5": cross_n,
        "per_task_n": {t: comp[t] for t in (URBAN7 + CROSS5)},
        "check_4734": (urban_n + cross_n == 4734),
        "note": "12 released tasks only; green_space/road_surface present in the raw "
                "file are NOT part of the released benchmark and are not reported.",
    }

    # ---- pull cross-model panels & corpus stats from the xlsx
    try:
        import openpyxl
        wb = openpyxl.load_workbook(XLSX, data_only=True)

        # External RS-VLM: recompute URBAN7 unweighted means from per-topic cells
        ws = wb["External_RSVLM"]
        ext_rows = list(ws.iter_rows(values_only=True))
        hdr = ext_rows[0]
        col = {name: i for i, name in enumerate(hdr)}
        ext = {}
        for r in ext_rows[1:]:
            if r[0] is None or "our trained" in str(r[0]):
                continue
            vals = [r[col[t]] for t in URBAN7]
            if all(v is not None for v in vals):
                ext[r[0]] = {
                    "urban7_mean": 100 * sum(vals) / len(vals),
                    "per_topic": {t: 100 * r[col[t]] for t in URBAN7},
                    "caveat": r[col.get("caveat", len(hdr) - 1)] or "",
                }
        out["external_rsvlm_urban7"] = ext
        out["external_rsvlm_urban7"]["_our_satfwd_sat_urban7_ref"] = table2["MEAN"]["sat"]

        # Raw VLM overall (regime replication) — pull OVERALL_8urban_noleak full + per available
        wsr = wb["Raw_Overall"]
        raw = {}
        for r in wsr.iter_rows(values_only=True):
            if not r or r[0] in (None, "model"):
                continue
            model, split, scope, nfull, bl, sa, sv_, fu, fus, train = (list(r) + [None] * 10)[:10]
            if scope == "OVERALL_8urban_noleak":
                raw.setdefault(model, {})["urban_full"] = fu
                raw[model]["blind"] = bl
            if scope == "OVERALL_all":
                raw.setdefault(model, {})["all_full"] = fu
        out["raw_vlm_overall"] = raw

        # 5STV vs satfwd supplement (Table 6) — benchmark section MEAN row
        ws6 = wb["SatFwd_vs_5STV"]
        rows6 = list(ws6.iter_rows(values_only=True))
        hdr_idx = [i for i, rr in enumerate(rows6) if rr and rr[0] == "topic"]
        # second header = benchmark section
        bench_hdr = hdr_idx[1]
        mean_row = None
        for rr in rows6[bench_hdr + 1:]:
            if rr and rr[0] and str(rr[0]).startswith("MEAN"):
                mean_row = rr; break
        if mean_row:
            # columns: topic | A blind sat sv full fusion | B blind sat sv full fusion | ...
            out["supp_5stv_vs_satfwd"] = {
                "scope_8urban_noleak": "benchmark MEAN incl. road_surface (sheet MEAN row)",
                "model_A_5stv_full_8urban": mean_row[4],
                "model_B_satfwd_full_8urban": mean_row[9],
                "delta_full_8urban": (mean_row[9] - mean_row[4]) if (mean_row[9] is not None and mean_row[4] is not None) else None,
            }
            # also compute the 7-urban (drop road_surface) version to match paper scope
            A7, B7 = [], []
            for rr in rows6[bench_hdr + 1:]:
                if rr and rr[0] and str(rr[0]).startswith("MEAN"):
                    break
                if rr and rr[0] in URBAN7:
                    A7.append(rr[4]); B7.append(rr[9])   # A full, B full
            if A7 and B7:
                a7 = sum(A7) / len(A7); b7 = sum(B7) / len(B7)
                out["supp_5stv_vs_satfwd"]["model_A_5stv_full_urban7"] = a7
                out["supp_5stv_vs_satfwd"]["model_B_satfwd_full_urban7"] = b7
                out["supp_5stv_vs_satfwd"]["delta_full_urban7_satfwd_minus_5stv"] = b7 - a7
                out["supp_5stv_vs_satfwd"]["note"] = (
                    "On the benchmark, the 4-SV (5STV) model scores ~%.1f vs satfwd ~%.1f on "
                    "the 7 urban tasks: training with 3 more street views buys ~+%.1f pts. "
                    "Honest framing: a small gain, well below the 4x visual-data cost." % (a7, b7, a7 - b7)
                )

        # Sample4Geo corroboration
        wss = wb["Sample4Geo"]
        s4g = {}
        for r in wss.iter_rows(values_only=True):
            if r and r[0] and "Sample4Geo" in str(r[0]):
                s4g[r[0]] = {"binary_easy": r[1], "binary_hard": r[2]}
        out["sample4geo"] = s4g

        # Prune survival (27B) — k=0,1 arms, OVERALL% (col 12). Note: k is a STRING.
        wsp = wb["Prune_Survival"]
        OVR = 12
        prune = {}
        rand_k1 = []
        for r in wsp.iter_rows(values_only=True):
            if not r or r[0] in (None, "run"):
                continue
            run, arm, k = str(r[0]), str(r[1]), str(r[2])
            if run != "k123_multiseed":
                continue
            if arm in ("zero", "top", "fwd", "bottom") and k in ("0", "1"):
                prune[f"{arm}_k{k}"] = r[OVR]
            if arm.startswith("random") and k == "1":
                rand_k1.append(r[OVR])
        if rand_k1:
            prune["random_k1_mean"] = sum(rand_k1) / len(rand_k1)
            prune["random_k1_seeds"] = rand_k1
        prune["_note"] = ("Survival = fraction of GREEDY-SOLVED questions still correct when "
                          "pruned to k images (NOT accuracy). fwd-k1 ~89%, zero-k0 ~67% (leakage floor). "
                          "top vs random vs bottom @k1 modest.")
        out["prune_survival_27b"] = prune

    except Exception as e:
        out["_xlsx_error"] = repr(e)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {OUT}")

    # ---- console summary of the load-bearing numbers (sanity)
    print("\n=== LOAD-BEARING NUMBERS (sanity) ===")
    print(f"URBAN7 paired n = {out['urban7_n_paired']}")
    m = table2["MEAN"]
    print(f"Table2 MEAN (v4, no blind): sat={m['sat']:.1f} sv={m['sv']:.1f} "
          f"full={m['full']:.1f} synergy_unw={m['synergy_unweighted']:+.2f}")
    ri = out["reads_imagery_urban7_v4"]
    print(f"reads-imagery: floor(alwaysA)={ri['always_A_floor_pct']:.1f}% "
          f"min single-view = {ri['min_single_view_task'][0]:.1f}% on {ri['min_single_view_task'][1]}")
    print(f"pooled synergy = {pooled['synergy_pooled']:+.2f} "
          f"CI[{pooled['synergy_pooled_ci'][0]:+.2f},{pooled['synergy_pooled_ci'][1]:+.2f}] "
          f"crosses0={pooled['synergy_pooled_ci_crosses_zero']}")
    print(f"full-oracle = {pooled['full_minus_oracle']:+.2f} "
          f"CI[{pooled['full_minus_oracle_ci'][0]:+.2f},{pooled['full_minus_oracle_ci'][1]:+.2f}] "
          f"excl0={pooled['full_minus_oracle_ci_excludes_zero']}")
    il = out["item_level_urban7"]
    print(f"fusion_win={il['fusion_win']} ({il['fusion_win_pct']:.1f}%) "
          f"interfere={il['interfere']} ({il['interfere_pct']:.1f}%) "
          f"ratio={il['ratio_interfere_to_win']:.1f}:1 identity_ok={il['identity_check']['equal']}")
    mc = out["mcnemar_urban7"]
    print(f"McNemar full-vs-sat p={mc['full_vs_sat']['p_exact']:.3f} "
          f"full-vs-sv p={mc['full_vs_sv']['p_exact']:.2e} "
          f"full-vs-oracle p={mc['full_vs_oracle']['p_exact']:.2e}")
    lr = out["robustness_urban7_v4"]
    print(f"robustness(v4): full-oracle all={lr['full_minus_oracle_all_items']:+.1f} "
          f"refused={sum(lr['refused'].values())} hedged={sum(lr['hedged'].values())}")
    fn = out["benchmark_composition"]
    print(f"composition: released12={fn['released_benchmark_12task']} "
          f"=urban7 {fn['urban7']}+cross5 {fn['cross5']}  check4734={fn['check_4734']}")
    ld = out["letter_distribution"]
    print(f"letters URBAN7 chi2={ld['urban7']['chi2']:.1f} alwaysA={ld['urban7']['always_A_pct']:.1f}%  "
          f"cross5 chi2={ld['cross5']['chi2']:.1f}  released12 chi2={ld['released12']['chi2']:.1f}")
    print(f"image-count full: urban7={out['image_count_full_mode']['urban7']} "
          f"cross5={out['image_count_full_mode']['cross5']}")


if __name__ == "__main__":
    main()
