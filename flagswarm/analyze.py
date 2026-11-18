"""Stage analysis: paired block-level losses, comparators, ranking scores, windows/doses, A/A noise.

All endpoint numbers come from provider calls (source="live") or the scripted harness (source="scripted"); the
predictor columns are frozen model outputs (predictions.json, written before the run).
Loss convention: D = Y(baseline) - Y(arm) at the readout round; positive = the muted speaker helped the truth.
Uncertainty: bootstrap over independent blocks, resampled within flag-case strata.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .manifest import stream
from .runner import safe


def Y(readouts, truth, r, exclude=None):
    rows = [x for x in readouts if x["round"] == r and x["agent"] != exclude]
    n = len(rows)
    k = sum(x["country"] == truth for x in rows)
    inv = sum(not x["valid"] for x in rows)
    return k / n, (k + inv) / n, inv   # point (invalid = not truth), upper bound (invalid = truth), invalid count


def plurality(readouts, truth, r):
    c = Counter(x["country"] for x in readouts if x["round"] == r and x["valid"])
    if not c:
        return np.nan, None
    top = max(c.values())
    w = sorted(k for k, v in c.items() if v == top)
    return (1 / len(w) if truth in w else 0.0), tuple(w)


def load_branches(stage_dir: Path, man: dict) -> dict:
    out = {}
    for b in man["blocks"]:
        for a in b["arms"]:
            f = stage_dir / "branches" / f"{safe(b['block_id'])}__{safe(a['name'])}.json"
            if f.exists():
                out[(b["block_id"], a["name"])] = json.loads(f.read_text())
    return out


def effects_table(stage_dir: Path) -> pd.DataFrame:
    man = json.loads((stage_dir / "manifest.json").read_text())
    preds = {p["block_id"]: p for p in json.loads((stage_dir / "predictions.json").read_text())["blocks"]}
    br = load_branches(stage_dir, man)
    rows = []
    for b in man["blocks"]:
        bid, ex = b["block_id"], b["extra"]
        base = br.get((bid, "baseline"))
        if base is None:
            continue
        for a in b["arms"]:
            arm = br.get((bid, a["name"]))
            if arm is None:
                rows.append({"block_id": bid, "arm": a["name"], "missing": True})
                continue
            target = a["target"]
            for r in b["readout_rounds"]:
                yb, yb_hi, ib = Y(base["readouts"], b["truth"], r)
                ya, ya_hi, ia = Y(arm["readouts"], b["truth"], r)
                yb_o = Y(base["readouts"], b["truth"], r, target)[0] if target is not None else np.nan
                ya_o = Y(arm["readouts"], b["truth"], r, target)[0] if target is not None else np.nan
                pb, wb = plurality(base["readouts"], b["truth"], r)
                pa, wa = plurality(arm["readouts"], b["truth"], r)
                rows.append({
                    "source": arm["source"], "block_id": bid, "case": ex["case"], "cohort": ex["cohort"], "N": ex["N"],
                    "order": ex["order"], "arm": a["name"], "kind": a["kind"], "target": target,
                    "window": a["window"], "h": a["h"], "slots": a["slots"], "round": r,
                    "role": (b["roles"][target] if target is not None else None),
                    "named_T": ex["named"]["T"], "named_R": ex["named"]["R"],
                    "Y_base": yb, "Y_arm": ya, "D": yb - ya, "D_lo": yb - ya_hi, "D_hi": yb_hi - ya,
                    "D_others": yb_o - ya_o, "plural_base": pb, "plural_arm": pa, "D_plural": pb - pa,
                    "winner_switch": (wb != wa), "invalid_votes": ib + ia,
                    "invalid_events": sum(not e["valid"] for e in arm["events"]),
                    "n_dropped": sum(e["dropped"] for e in arm["events"]),
                    "M_target": (sum(1 for s, _ in b["schedule"] if s == target) if target is not None else None),
                    "pred_full": (preds[bid]["pred_full_mute"][target] if target is not None else None),
                    "pred_chi": (preds[bid]["pred_chi"][target] if target is not None else None),
                    "missing": False,
                })
    return pd.DataFrame(rows)


def boot_mean(df: pd.DataFrame, col: str, strata: str = "case", B: int = 2000, seed: int = 0) -> dict:
    """Mean of per-block values with a stratified bootstrap over blocks."""
    d = df.dropna(subset=[col])
    if d.empty:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in d.groupby(strata)]
    if min(len(g) for g in groups) < 2:   # single-block strata would give zero-width intervals: pool them
        groups = [d[col].to_numpy()]
    bs = [np.mean(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(B)]
    return {"mean": float(d[col].mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": int(len(d))}


def comparator_targets(man_block: dict, pred: dict, seed: int) -> dict:
    """Pre-intervention comparators: predicted top, uniform random, opportunity-matched and role-matched to the
    predicted top (closest eligible-contact count, then earliest first contact, then seeded random)."""
    N = len(man_block["crops"])
    sched = man_block["schedule"]
    M = pred["eligible_contacts"]
    first = [next((t for t, (s, _) in enumerate(sched) if s == i), 10**9) for i in range(N)]
    top = pred["top_helpful"]
    rng = stream(seed, "comparator", man_block["block_id"])
    rand = rng.randrange(N)

    def match(pool):
        pool = [i for i in pool if i != top]
        if not pool:
            return None
        best = min(abs(M[i] - M[top]) for i in pool)
        pool = [i for i in pool if abs(M[i] - M[top]) == best]
        bf = min(abs(first[i] - first[top]) for i in pool)
        pool = [i for i in pool if abs(first[i] - first[top]) == bf]
        return rng.choice(pool)

    roles = man_block["roles"]
    return {"top": top, "random": rand, "opp_matched": match(range(N)),
            "role_matched": match([i for i in range(N) if roles[i] == roles[top]]),
            "named_T": man_block["extra"]["named"]["T"], "named_R": man_block["extra"]["named"]["R"]}


def analyze(stage_dir: Path, readout_round: int = 10) -> dict:
    stage_dir = Path(stage_dir)
    man = json.loads((stage_dir / "manifest.json").read_text())
    preds = {p["block_id"]: p for p in json.loads((stage_dir / "predictions.json").read_text())["blocks"]}
    eff = effects_table(stage_dir)
    out_dir = stage_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    eff.to_csv(out_dir / "effects.csv", index=False)
    e = eff[(eff["missing"] == False) & (eff["round"] == readout_round)]
    summary = {"source": sorted(e["source"].dropna().unique().tolist()), "readout_round": readout_round,
               "causal_convention": man["causal_convention"], "cells": {}}
    blocks = {b["block_id"]: b for b in man["blocks"]}
    rank_rows = []
    for (N, cohort), cell in e.groupby(["N", "cohort"]):
        c = {"n_blocks": int(cell["block_id"].nunique())}
        mutes = cell[cell["kind"] == "mute_window"]
        full = mutes[mutes["window"].isna()]
        aa = cell[cell["arm"] == "aa"]
        c["baseline_Y"] = boot_mean(cell[cell["arm"] == "baseline"], "Y_base")
        c["aa_D"] = boot_mean(aa, "D")
        c["aa_sd"] = float(aa["D"].std()) if len(aa) > 1 else np.nan
        c["aa_switch_rate"] = float(aa["winner_switch"].mean()) if len(aa) else np.nan
        c["mute_switch_rate"] = float(full["winner_switch"].mean()) if len(full) else np.nan
        c["invalid_votes"] = int(cell["invalid_votes"].sum())
        # comparators (one row per block per comparator)
        comp_rows = []
        for bid, g in full.groupby("block_id"):
            ct = comparator_targets(blocks[bid], preds[bid], man["seed"])
            byt = {int(t): r for t, r in zip(g["target"], g.to_dict("records"))}
            for name, t in ct.items():
                if t is not None and t in byt:
                    comp_rows.append({"block_id": bid, "case": byt[t]["case"], "comparator": name, "target": t,
                                      "D": byt[t]["D"], "D_others": byt[t]["D_others"], "D_plural": byt[t]["D_plural"],
                                      "pred_full": byt[t]["pred_full"]})
            # ranking scores across the N speakers of this block (held-out mutes)
            obs = np.array([byt[i]["D"] for i in range(len(blocks[bid]["crops"])) if i in byt])
            if len(obs) == len(blocks[bid]["crops"]) and np.std(obs) > 0:
                p = preds[bid]
                p_inf = max(p["p_crop"])
                S = [(p_inf - pi) * Ei for pi, Ei in zip(p["p_crop"], p["temporal_closeness"])]
                scores = {"pred_full_mute": p["pred_full_mute"], "pred_chi": p["pred_chi"], "temporal_closeness": p["temporal_closeness"],
                          "private_accuracy": p["p_crop"], "paper_S": S, "eligible_contacts": p["eligible_contacts"]}
                rr = {"N": N, "cohort": cohort, "block_id": bid}
                for k, s in scores.items():
                    rr[f"rho_{k}"] = spearmanr(s, obs).statistic if np.std(s) > 0 else np.nan
                rr["top_hit"] = int(np.argmax(p["pred_full_mute"]) == int(np.argmax(obs)))
                rr["top_minus_random_D"] = obs[p["top_helpful"]] - obs.mean()
                rank_rows.append(rr)
        comp = pd.DataFrame(comp_rows)
        c["comparators"] = {k: {"D": boot_mean(g, "D"), "D_others": boot_mean(g, "D_others"), "D_plural": boot_mean(g, "D_plural"),
                                "pred": float(g["pred_full"].mean())} for k, g in comp.groupby("comparator")} if len(comp) else {}
        if len(comp):
            piv = comp.pivot_table(index=["block_id", "case"], columns="comparator", values="D").reset_index()
            for a_, b_ in (("top", "random"), ("top", "opp_matched"), ("top", "role_matched"), ("named_T", "named_R")):
                if a_ in piv and b_ in piv:
                    piv[f"{a_}-{b_}"] = piv[a_] - piv[b_]
                    c[f"diff_{a_}_minus_{b_}"] = boot_mean(piv, f"{a_}-{b_}")
        # signed full-mute loss by role over all speakers
        c["full_mute_by_role"] = {r: boot_mean(g, "D") | {"pred": float(g["pred_full"].mean())} for r, g in full.groupby("role")}
        # windows and doses on subset blocks
        win = cell[cell["arm"].str.startswith(("early:", "late:"))]
        for kind in ("early", "late"):
            for who in ("T", "R"):
                g = win[win["arm"].str.startswith(kind) & (win["target"] == win[f"named_{who}"])]
                pk = [preds[b_][f"pred_{kind}"][int(t_)] for b_, t_ in zip(g["block_id"], g["target"])]
                c[f"{kind}_named_{who}"] = boot_mean(g, "D") | {"pred": float(np.mean(pk)) if pk else np.nan}
        sub_ids = set(cell[cell["arm"].str.startswith("dose")]["block_id"])
        for who in ("T", "R"):
            g = full[(full["block_id"].isin(sub_ids)) & (full["target"] == full[f"named_{who}"])]
            c[f"full_named_{who}_subset"] = boot_mean(g, "D") | {"pred": float(g["pred_full"].mean()) if len(g) else np.nan}
        doses = {}
        for h in (0.1, 0.25, 0.5):
            doses[str(h)] = boot_mean(cell[cell["arm"].str.startswith(f"dose{h}:")], "D")
        doses["1.0"] = c["full_named_T_subset"]
        c["doses_named_T"] = doses
        om = cell[cell["arm"].str.startswith("omit")]
        if len(om):
            chi_hat = om.groupby("block_id").apply(lambda g: g["D"].mean() * g["M_target"].iloc[0], include_groups=False)
            c["chi_hat_named_T"] = {"mean": float(chi_hat.mean()), "sd": float(chi_hat.std()), "n_blocks": int(len(chi_hat)),
                                    "pred_chi_mean": float(om.groupby("block_id")["pred_chi"].first().mean())}
        summary["cells"][f"N{N}|{cohort}"] = c
    rank = pd.DataFrame(rank_rows)
    rank.to_csv(out_dir / "ranking.csv", index=False)
    if len(rank):
        summary["ranking"] = {f"N{N}|{cohort}": {col: boot_mean(g.assign(case=g["block_id"].str.split("|").str[0]), col)
                                                   for col in g.columns if col.startswith("rho_") or col in ("top_hit", "top_minus_random_D")}
                              for (N, cohort), g in rank.groupby(["N", "cohort"])}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def paired_by_round(eff: pd.DataFrame, arm_prefix: str, target_col: str = "named_T") -> pd.DataFrame:
    """Per readout round: mean loss and stratified-bootstrap CI for the arms starting with arm_prefix aimed at the
    named target."""
    rows = []
    sel = eff[(eff["missing"] == False) & eff["arm"].str.startswith(arm_prefix) & (eff["target"] == eff[target_col])]
    for (N, cohort, r), g in sel.groupby(["N", "cohort", "round"]):
        rows.append({"N": N, "cohort": cohort, "round": r, "arm": arm_prefix, **boot_mean(g, "D"),
                     "D_plural": g["D_plural"].mean(), "Y_base": g["Y_base"].mean()})
    return pd.DataFrame(rows)


def analyze_stage3(stage_dir: Path) -> pd.DataFrame:
    stage_dir = Path(stage_dir)
    eff = effects_table(stage_dir)
    (stage_dir / "analysis").mkdir(exist_ok=True)
    eff.to_csv(stage_dir / "analysis" / "effects.csv", index=False)
    aa = eff[(eff["missing"] == False) & (eff["arm"] == "aa")]
    aa_rows = [{"round": r, "arm": "aa", **boot_mean(g, "D"), "sd": g["D"].std()} for r, g in aa.groupby("round")]
    out = pd.concat([paired_by_round(eff, "mute:"), paired_by_round(eff, "restore"), pd.DataFrame(aa_rows)], ignore_index=True)
    out.to_csv(stage_dir / "analysis" / "by_round.csv", index=False)
    return out


def analyze_stage4(stage_dir: Path, readout_round: int = 10) -> pd.DataFrame:
    """Mechanism contrasts on matched blocks: m=1 vs m=3 named-target mutes, placeholder vs silent drop,
    rate-/time-matched omissions from other speakers vs the named truth target's mute."""
    stage_dir = Path(stage_dir)
    eff = effects_table(stage_dir)
    (stage_dir / "analysis").mkdir(exist_ok=True)
    eff.to_csv(stage_dir / "analysis" / "effects.csv", index=False)
    e = eff[(eff["missing"] == False) & (eff["round"] == readout_round)].copy()
    e["m"] = e["block_id"].str[1].astype(int)
    rows = []
    for cohort, c in e.groupby("cohort"):
        for label, sel in (("mute named T", (c.arm.str.startswith("mute:")) & (c.target == c.named_T)),
                           ("mute named R", (c.arm.str.startswith("mute:")) & (c.target == c.named_R)),
                           ("placeholder named T", c.arm.str.startswith("placeholder:")),
                           ("rate-matched others", c.arm.str.startswith("ratematched:")),
                           ("A/A", c.arm == "aa")):
            for m, g in c[sel].groupby("m"):
                rows.append({"cohort": cohort, "m": m, "contrast": label, **boot_mean(g, "D"), "Y_base": g["Y_base"].mean(),
                             "switch": g["winner_switch"].mean()})
        # paired differences within block (m=3 arms)
        piv = c[c.m == 3].assign(a=c.arm.str.split(":").str[0]).pivot_table(index=["block_id", "case"], columns="a", values="D").reset_index()
        tmute = c[(c.m == 3) & c.arm.str.startswith("mute:") & (c.target == c.named_T)].set_index("block_id")["D"]
        piv["muteT"] = piv["block_id"].map(tmute)
        for a_, b_ in (("muteT", "placeholder"), ("muteT", "ratematched")):
            piv[f"{a_}-{b_}"] = piv[a_] - piv[b_]
            rows.append({"cohort": cohort, "m": 3, "contrast": f"{a_} minus {b_}", **boot_mean(piv, f"{a_}-{b_}")})
        # bandwidth: named-T mute loss at m=3 minus m=1 on the same source block
        src = c[c.arm.str.startswith("mute:") & (c.target == c.named_T)].assign(src=lambda d: d.block_id.str[3:])
        bw = src.pivot_table(index=["src", "case"], columns="m", values="D").reset_index()
        if 1 in bw and 3 in bw:
            bw["d"] = bw[3] - bw[1]
            rows.append({"cohort": cohort, "m": 0, "contrast": "named-T mute: m3 minus m1", **boot_mean(bw, "d")})
    out = pd.DataFrame(rows)
    out.to_csv(stage_dir / "analysis" / "mechanism.csv", index=False)
    return out


def lock_in(stage_dir: Path, arm: str = "baseline", early: int = 100) -> pd.DataFrame:
    """Per readout round of one arm, over games: share of games whose flexible (non-clamped) agents all give the
    same valid label (consensus), share with at most one flexible dissenter (near_consensus), mean truth share Y, and the share of flexible agents whose label at this round
    differs from their label at round `early` (flip_from_early; NaN for rounds <= early)."""
    stage_dir = Path(stage_dir)
    man = json.loads((stage_dir / "manifest.json").read_text())
    br = load_branches(stage_dir, man)
    rows = []
    for b in man["blocks"]:
        x = br.get((b["block_id"], arm))
        if x is None:
            continue
        flex = set(range(len(b["crops"]))) - {int(k) for k in (b.get("clamped") or {})}
        lab = {(r["round"], r["agent"]): (r["country"] if r["valid"] else None) for r in x["readouts"]}
        for r in b["readout_rounds"]:
            labs = [lab[(r, a)] for a in flex]
            flips = [lab[(r, a)] != lab[(early, a)] for a in flex] if r > early else [np.nan]
            top = Counter(labs).most_common(1)[0]
            rows.append({"block_id": b["block_id"], "round": r, "consensus": float(None not in labs and len(set(labs)) == 1),
                         "near_consensus": float(top[0] is not None and top[1] >= len(flex) - 1),
                         "Y": Y(x["readouts"], b["truth"], r)[0], "flip_from_early": float(np.mean(flips))})
    df = pd.DataFrame(rows)
    return df.groupby("round").agg(games=("block_id", "nunique"), consensus=("consensus", "mean"),
                                   near_consensus=("near_consensus", "mean"),
                                   Y_base=("Y", "mean"), flip_from_early=("flip_from_early", "mean")).reset_index()


def compare_memory(model: str, stages=("stage3", "stage3_h32")) -> pd.DataFrame:
    """Stage 3 vs its memory variant on identical blocks: mute/restore losses by round (bootstrap CIs) next to the
    baseline lock-in measures, plus the per-block paired difference of the sustained-mute loss (H32 minus H8)."""
    root = Path(__file__).resolve().parents[1] / "data" / "live" / model
    out = []
    effs = {}
    for st in stages:
        H = json.loads((root / st / "manifest.json").read_text()).get("H", 8)
        br = pd.read_csv(root / st / "analysis" / "by_round.csv")
        li = lock_in(root / st)
        for _, row in br[br.arm.isin(["mute:", "restore"])].iterrows():
            out.append({"stage": st, "H": H, "round": row["round"], "arm": row["arm"], "mean": row["mean"],
                        "lo": row["lo"], "hi": row["hi"], "Y_base": row["Y_base"]})
        for _, row in li.iterrows():
            out.append({"stage": st, "H": H, "round": row["round"], "arm": "baseline_lockin", "consensus": row["consensus"],
                        "near_consensus": row["near_consensus"],
                        "flip_from_early": row["flip_from_early"], "Y_base": row["Y_base"], "games": row["games"]})
        e = pd.read_csv(root / st / "analysis" / "effects.csv")
        effs[H] = e[(e.missing == False) & e.arm.str.startswith("mute:")].set_index(["block_id", "round"])[["D", "case"]]
    if len(effs) == 2:
        (h0, e0), (h1, e1) = sorted(effs.items())
        j = e0.join(e1, lsuffix="_a", rsuffix="_b", how="inner").reset_index()
        j["d"], j["case"] = j["D_b"] - j["D_a"], j["case_a"]
        for r, g in j.groupby("round"):
            out.append({"stage": f"H{h1}-H{h0}", "H": None, "round": r, "arm": "mute: paired diff", **boot_mean(g, "d")})
    df = pd.DataFrame(out)
    (root / stages[-1] / "analysis").mkdir(exist_ok=True)
    df.to_csv(root / stages[-1] / "analysis" / "memory_compare.csv", index=False)
    return df
