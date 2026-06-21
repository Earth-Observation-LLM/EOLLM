"""stats.py — insight statistics over a model's attention results.

The goal is WISDOM, not averages: stats like "X% of correct answers fixated on a
single street-view image" or "Y% of wrong mismatch answers actually attended most
to the GOLD option image but still picked wrong". These tie where-the-model-looked
to whether-it-was-right.

CORRECTNESS CONSTRAINTS (verified against the data — getting these wrong corrupts
every number):

1. `image_role` semantics differ by image_mode. The sat-vs-street-view bucket is
   ONLY meaningful where the images ARE the two perspectives:
     satellite_marked : [satellite_marked, 4x streetview_<angle>]   -> sat vs SV OK
     streetview_mega  : [satellite_marked, streetview_mega]         -> sat vs SV OK
   In the OPTION modes the non-first images are ANSWER CHOICES, not "the view":
     satellite_arrow  : [query_streetview, option_A..D_satellite_arrow]
     streetview_binary/composite : [satellite_marked, streetview_option_A..D]
   For those we compute OPTION-image attention instead (did it look at the gold /
   chosen option?), never a sat-vs-SV split.

2. raw vs norm. `*_norm_pct` sums to 100 across images (within-image split);
   `*_raw_pct` sums to image_attention_total (~7-10%, images-vs-text). Never mix.

3. attention is not explanation — these are descriptive/correlational. The causal
   companion is the ablation synergy in fusion.py. Stats here say "looked at",
   not "because of".

4. prob_dict is null in attention runs (the attention backend doesn't extract
   logprobs), so confidence-weighted stats are omitted.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

# modes where image[0]=satellite and the rest are the street-view perspective
PERSPECTIVE_MODES = {"satellite_marked", "streetview_mega"}
# modes where images 2..N are answer-option images (letter-mapped)
OPTION_MODES = {"satellite_arrow", "streetview_binary", "streetview_composite"}

_OPT_RE = re.compile(r"option_([A-D])", re.I)


def _is_sat(role): r = role.lower(); return ("satellite" in r) or ("sat_" in r)
def _is_sv(role):  r = role.lower(); return ("streetview" in r) or ("stv" in r)


def option_letter(role):
    """For option-image modes, map a role to the answer letter it represents."""
    m = _OPT_RE.search(role)
    return m.group(1).upper() if m else None


def _entropy_norm(shares):
    ps = [s / 100.0 for s in shares if s > 0]
    if len(ps) <= 1:
        return 0.0
    tot = sum(ps) or 1e-9
    ps = [p / tot for p in ps]
    H = -sum(p * math.log(p) for p in ps)
    return H / math.log(len(ps))


def _att(r):
    a = r.get("attention")
    if not a or not a.get("images") or a.get("note"):
        return None
    return a


def sat_sv_split(r, key="choice_norm_pct"):
    """(sat_pct, sv_pct) for PERSPECTIVE modes only; None otherwise."""
    if r.get("image_mode") not in PERSPECTIVE_MODES:
        return None
    a = _att(r)
    if not a:
        return None
    sat = sum(im[key] for im in a["images"] if _is_sat(im["role"]))
    sv = sum(im[key] for im in a["images"] if _is_sv(im["role"]))
    return (sat, sv)


def option_attention(r, key="choice_norm_pct"):
    """For OPTION modes: dict letter->attention% on that option image, plus which
    letter got the most attention. None for non-option modes."""
    if r.get("image_mode") not in OPTION_MODES:
        return None
    a = _att(r)
    if not a:
        return None
    by_letter = {}
    for im in a["images"]:
        L = option_letter(im["role"])
        if L:
            by_letter[L] = im[key]
    if not by_letter:
        return None
    argmax = max(by_letter, key=by_letter.get)
    return {"by_letter": by_letter, "argmax_letter": argmax}


def _pct(num, den):
    return round(100 * num / den, 1) if den else None


def aggregate(rows):
    usable = [r for r in rows if _att(r)]
    n = len(usable)
    if not n:
        return {"n": 0}

    persp = [r for r in usable if r.get("image_mode") in PERSPECTIVE_MODES]
    opt = [r for r in usable if r.get("image_mode") in OPTION_MODES]

    out = {"n": n, "n_perspective": len(persp), "n_option": len(opt),
           "insights": [], "by_topic": {}, "by_layer": []}

    # ---------- corpus sat-vs-sv (perspective modes only) -------------------
    if persp:
        sat = sum(sat_sv_split(r)[0] for r in persp) / len(persp)
        sv = sum(sat_sv_split(r)[1] for r in persp) / len(persp)
        out["sat_vs_sv"] = {"satellite_pct": round(sat, 1), "streetview_pct": round(sv, 1),
                            "n": len(persp),
                            "note": "choice-token share among images; perspective modes only (marked+mega)"}

    # ---------- image-vs-text engagement -----------------------------------
    it_all = [r["attention"]["image_attention_total_avg_pct"] for r in usable]
    out["image_vs_text"] = {"mean_image_total_pct": round(sum(it_all) / n, 2),
                            "note": "share of ALL attention on images (rest on text)"}

    # ---------- WISDOM INSIGHTS (the headline numbers) ----------------------
    ins = out["insights"]

    # I1 — of CORRECT perspective answers, % that fixated a single street-view
    #      (one SV image holds the dominant share among images, > all others incl sat)
    def dominant_is_single_sv(r):
        a = _att(r)
        top = max(a["images"], key=lambda im: im["choice_norm_pct"])
        return _is_sv(top["role"]) and not _is_sat(top["role"])
    corr_persp = [r for r in persp if r["is_correct"]]
    if corr_persp:
        k = sum(1 for r in corr_persp if dominant_is_single_sv(r))
        ins.append({
            "id": "correct_fixated_single_sv",
            "label": "of CORRECT answers (perspective modes), focused most on a single street-view image",
            "pct": _pct(k, len(corr_persp)), "k": k, "n": len(corr_persp)})

    # I2 — same for satellite dominance among correct
    if corr_persp:
        k = sum(1 for r in corr_persp
                if _is_sat(max(r["attention"]["images"], key=lambda im: im["choice_norm_pct"])["role"]))
        ins.append({
            "id": "correct_focused_satellite",
            "label": "of CORRECT answers (perspective modes), focused most on the satellite",
            "pct": _pct(k, len(corr_persp)), "k": k, "n": len(corr_persp)})

    # I3 — of WRONG option answers (mismatch/arrow), % that attended MOST to the
    #      GOLD option image yet still answered wrong (looked right, decided wrong)
    wrong_opt = [r for r in opt if not r["is_correct"]]
    if wrong_opt:
        k = 0
        for r in wrong_opt:
            oa = option_attention(r)
            if oa and oa["argmax_letter"] == r["gold"]:
                k += 1
        ins.append({
            "id": "wrong_but_looked_at_gold_option",
            "label": "of WRONG answers (option modes), attended MOST to the gold option image but still chose wrong",
            "pct": _pct(k, len(wrong_opt)), "k": k, "n": len(wrong_opt)})

    # I4 — of CORRECT option answers, % whose top-attended option == the answer
    #      (attention-choice coupling when right)
    corr_opt = [r for r in opt if r["is_correct"]]
    if corr_opt:
        k = 0
        for r in corr_opt:
            oa = option_attention(r)
            if oa and oa["argmax_letter"] == r["prediction"]:
                k += 1
        ins.append({
            "id": "correct_looked_at_chosen_option",
            "label": "of CORRECT answers (option modes), attended MOST to the option image it picked",
            "pct": _pct(k, len(corr_opt)), "k": k, "n": len(corr_opt)})

    # I5 — overall: did argmax-attended option match the PREDICTION (coupling)
    if opt:
        k = 0
        for r in opt:
            oa = option_attention(r)
            if oa and oa["argmax_letter"] == r["prediction"]:
                k += 1
        ins.append({
            "id": "attention_choice_coupling",
            "label": "of ALL option-mode answers, top-attended option image == the model's pick (attention–choice coupling)",
            "pct": _pct(k, len(opt)), "k": k, "n": len(opt)})

    # I6 — fixation gap: mean entropy correct vs wrong (perspective)
    if persp:
        ent_c = [_entropy_norm([im["choice_norm_pct"] for im in r["attention"]["images"]])
                 for r in persp if r["is_correct"]]
        ent_w = [_entropy_norm([im["choice_norm_pct"] for im in r["attention"]["images"]])
                 for r in persp if not r["is_correct"]]
        ins.append({
            "id": "fixation_correct_vs_wrong",
            "label": "mean attention spread (entropy) — correct vs wrong (perspective); lower=more fixated",
            "correct": round(sum(ent_c)/len(ent_c), 3) if ent_c else None,
            "wrong": round(sum(ent_w)/len(ent_w), 3) if ent_w else None,
            "n_correct": len(ent_c), "n_wrong": len(ent_w)})

    # ---------- per-topic sat/sv (perspective) or gold-option-hit (option) ---
    by_topic = defaultdict(list)
    for r in usable:
        by_topic[r["topic"]].append(r)
    for t, rs in sorted(by_topic.items()):
        mode_kind = "perspective" if rs[0].get("image_mode") in PERSPECTIVE_MODES else \
                    ("option" if rs[0].get("image_mode") in OPTION_MODES else "other")
        acc = _pct(sum(1 for r in rs if r["is_correct"]), len(rs))
        entry = {"n": len(rs), "acc": acc, "kind": mode_kind,
                 "img_total_pct": round(sum(r["attention"]["image_attention_total_avg_pct"] for r in rs)/len(rs), 2)}
        if mode_kind == "perspective":
            sat = sum(sat_sv_split(r)[0] for r in rs) / len(rs)
            sv = sum(sat_sv_split(r)[1] for r in rs) / len(rs)
            entry["sat_share"] = round(sat, 1); entry["sv_share"] = round(sv, 1)
        elif mode_kind == "option":
            # % where top-attended option is the GOLD option
            hit = sum(1 for r in rs if (oa := option_attention(r)) and oa["argmax_letter"] == r["gold"])
            entry["gold_option_attn_hit_pct"] = _pct(hit, len(rs))
        out["by_topic"][t] = entry

    # ---------- per-layer image attention (raw, choice token) ---------------
    layer_tot = defaultdict(list)
    for r in usable:
        for pl in r["attention"].get("per_layer", []):
            layer_tot[pl["layer"]].append(sum(pl["choice_raw_pct"]))
    out["by_layer"] = [{"layer": k, "img_total_choice_pct": round(sum(v)/len(v), 3)}
                       for k, v in sorted(layer_tot.items())]

    # ---------- extraction hygiene -----------------------------------------
    clean = sum(1 for r in usable if r["attention"].get("spans_match_images")
                and abs((r["attention"].get("row_softmax_check") or 0) - 1) < 0.02)
    out["hygiene"] = {"clean_pct": _pct(clean, n), "n": n}
    return out


def per_record_extras(att, record=None):
    """Per-record helpers for the selected sample."""
    if not att or not att.get("images"):
        return {}
    shares = [im["choice_norm_pct"] for im in att["images"]]
    top = max(att["images"], key=lambda im: im["choice_norm_pct"])
    out = {
        "entropy": round(_entropy_norm(shares), 3),
        "top_image": {"image": top["image"], "role": top["role"],
                      "choice_norm_pct": round(top["choice_norm_pct"], 1)},
    }
    if record is not None:
        mode = record.get("image_mode")
        if mode in PERSPECTIVE_MODES:
            ss = sat_sv_split(record)
            if ss:
                out["sat_share_pct"] = round(ss[0], 1)
                out["sv_share_pct"] = round(ss[1], 1)
                out["kind"] = "perspective"
        elif mode in OPTION_MODES:
            oa = option_attention(record)
            if oa:
                out["kind"] = "option"
                out["option_attn"] = {k: round(v, 1) for k, v in oa["by_letter"].items()}
                out["argmax_option"] = oa["argmax_letter"]
                out["gold"] = record.get("gold")
                out["prediction"] = record.get("prediction")
                out["looked_at_gold"] = (oa["argmax_letter"] == record.get("gold"))
    return out
