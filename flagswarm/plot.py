"""Figures from saved results only (no provider calls). Each figure gets a CSV of the numbers it shows.

python -m flagswarm.plot offline|calibration|stage1 [--model gpt-6-luna] [--stage stage1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import env
from . import theory as th
from .figstyle import PALETTE, apply_publication_style, create_subplots, finalize_figure, make_trend

FIG = env.ROOT / "figures"


def offline(name: str = "v1") -> None:
    apply_publication_style()
    d = env.ROOT / "data" / "offline" / name
    pvt = pd.read_csv(d / "planted_vs_theory.csv")
    fig, axes = create_subplots(1, 2, figsize=(13, 5))
    colors = {8: PALETTE["blue_main"], 16: PALETTE["teal"], 32: PALETTE["red_strong"]}
    for ax, prof, title in zip(axes, ("window_1to5", "sustained"), ("Mute rounds 1-5, then restore", "Sustained mute")):
        for N, g in pvt[(pvt.profile == prof) & (~pvt.self_contacts)].groupby("N"):
            ax.plot(g["round"], g["theory_loss"], color=colors[N], label=f"N={N} exact")
            ax.errorbar(g["round"], g["mc_loss"], yerr=2 * g["mc_se"], fmt="o", ms=4, color=colors[N], alpha=0.7, label=f"N={N} MC")
        ax.axvline(10, color=PALETTE["gray_dark"], ls=":", lw=1.5)
        ax.set_title(title, fontsize=15); ax.set_xlabel("rounds"); ax.set_ylabel("truth-share loss D")
    axes[0].legend(fontsize=10, ncol=2)
    pvt.to_csv(FIG / "offline_planted_vs_theory.csv", index=False)
    finalize_figure(fig, FIG / "offline_planted_vs_theory")

    sc = pd.read_csv(d / "scaling_curves.csv")
    fig, axes = create_subplots(1, 2, figsize=(13, 5))
    cmap = [PALETTE[k] for k in ("blue_main", "blue_secondary", "teal", "green_3", "violet", "red_strong")]
    for ax, regime, title in zip(axes, ("fixed_count", "fixed_fraction"), ("Fixed witness count (z=1)", "Fixed witness fraction (z=N/8)")):
        for k, (N, g) in enumerate(sc[sc.regime == regime].groupby("N")):
            g = g[g["round"] > 0]
            ax.plot(g["round"], g["loss"], color=cmap[k], label=f"N={N}")
        ax.axvline(10, color=PALETTE["gray_dark"], ls=":", lw=1.5)
        ax.set_xscale("log"); ax.set_xlabel("rounds (log)"); ax.set_ylabel("loss from muting one truth witness")
        ax.set_title(title, fontsize=15)
    axes[0].legend(fontsize=10)
    sc.to_csv(FIG / "offline_scaling_curves.csv", index=False)
    finalize_figure(fig, FIG / "offline_scaling_curves")


def calibration(model: str) -> None:
    apply_publication_style(); FIG.mkdir(exist_ok=True)
    cal = json.loads((env.ROOT / "data" / "live" / model / "calibration" / "calibration.json").read_text())
    rows = []
    for case, rr in cal["confirm_rates"].items():
        for h, v in rr.items():
            roles = [k for k, hs in cal["carried"][case].items() if h in hs]
            q = [k for k, hs in cal["qualified"][case].items() if h in hs]
            rows.append({"case": case, "hash": h[:8], "carried_as": "/".join(roles), "qualified_as": "/".join(q),
                         **{f"p_{k}": v[k]["p_truth"] for k in ("empty", "truth_heavy", "rival_heavy")},
                         **{f"n_{k}": v[k]["n"] for k in ("empty", "truth_heavy", "rival_heavy")}})
    df = pd.DataFrame(rows)
    df.to_csv(FIG / "calibration_confirm_rates.csv", index=False)
    fig, axes = create_subplots(2, 4, figsize=(20, 9), sharey=True)
    for ax, (case, g) in zip(axes, df.groupby("case", sort=False)):
        x = np.arange(len(g))
        for k, (col, lab, c) in enumerate((("p_empty", "empty", PALETTE["gray_dark"]), ("p_truth_heavy", "6 truth / 2 rival", PALETTE["blue_main"]),
                                            ("p_rival_heavy", "2 truth / 6 rival", PALETTE["red_strong"]))):
            ax.scatter(x + (k - 1) * 0.18, g[col], color=c, s=55, label=lab, zorder=3)
        for xi, (_, r) in zip(x, g.iterrows()):
            ax.vlines(xi, r["p_rival_heavy"], r["p_truth_heavy"], color=PALETTE["neutral"], lw=4, zorder=1)
        ax.axhline(0.5, color=PALETTE["neutral"], ls=":", lw=1.2)
        ax.set_xticks(x); ax.set_xticklabels([f"{c}{'*' if q else ''}" for c, q in zip(g["carried_as"], g["qualified_as"])], fontsize=11)
        ax.set_title(case.replace(">", " > "), fontsize=14); ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("P(name truth)"); axes[4].set_ylabel("P(name truth)")
    axes[0].legend(fontsize=10, loc="lower left")
    fig.suptitle(f"{model}: confirmatory crop probes (24 calls/regime; x label = role carried, * = qualified)", fontsize=15)
    finalize_figure(fig, FIG / "calibration_crop_probes")

    ev = pd.DataFrame(json.loads((env.ROOT / "data" / "live" / model / "qcal_games" / "qcal_events.json").read_text()))
    pred = json.loads((env.ROOT / "data" / "live" / model / "predictor.json").read_text())
    ev["x"] = ev["label_truth"].astype(float) - ev["pb"]
    ev["dp"] = ev["pa"] - ev["pb"]
    ev.to_csv(FIG / "calibration_listener_events.csv", index=False)
    fig, axes = create_subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, (sel, title, key) in zip(axes, ((ev.context == "zero", "zero history", "zero_history"),
                                            ((ev.context == "full") & (ev.fifo_len_before < 8), "in-game memory, FIFO not full", None),
                                            ((ev.context == "full") & (ev.fifo_len_before == 8), "in-game memory, FIFO full", "full_fifo"))):
        g = ev[sel]
        ax.scatter(g["x"] + np.random.default_rng(0).normal(0, 0.01, len(g)), g["dp"], s=14, alpha=0.35, color=PALETTE["blue_secondary"])
        xs = np.linspace(-1, 1, 3)
        q_all = pred["q"]
        ax.plot(xs, q_all * xs, color=PALETTE["red_strong"], label=f"frozen q={q_all:.2f}")
        if key and pred["fits"].get(key):
            qk = pred["fits"][key]["q"]; ax.plot(xs, qk * xs, color=PALETTE["teal"], ls="--", label=f"this subset q={qk:.2f}")
        ax.set_title(f"{title} (n={len(g)})", fontsize=14); ax.set_xlabel("1[message = truth] - p_before")
        ax.legend(fontsize=10)
    axes[0].set_ylabel("p_after - p_before")
    finalize_figure(fig, FIG / "calibration_listener_gain")


def stage1(model: str, stage: str = "stage1") -> None:
    apply_publication_style(); FIG.mkdir(exist_ok=True)
    sdir = env.ROOT / "data" / "live" / model / stage
    s = json.loads((sdir / "analysis" / "summary.json").read_text())
    cells = list(s["cells"])
    # 1) comparators
    comps = ["named_T", "top", "opp_matched", "role_matched", "random", "named_R"]
    labels = {"named_T": "named truth", "top": "predicted top", "opp_matched": "opportunity-matched",
              "role_matched": "role-matched", "random": "random", "named_R": "named rival"}
    colors = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["green_3"], PALETTE["green_2"], PALETTE["neutral"], PALETTE["red_strong"]]
    rows = []
    fig, axes = create_subplots(1, len(cells), figsize=(4.2 * len(cells), 5.5), sharey=True)
    for ax, cell in zip(axes, cells):
        cc = s["cells"][cell]["comparators"]
        aa = s["cells"][cell]["aa_D"]
        ax.axhspan(aa["lo"], aa["hi"], color=PALETTE["highlight"], alpha=0.25, lw=0, zorder=0)
        for k, c in enumerate(comps):
            if c not in cc:
                continue
            m = cc[c]["D"]
            if m["n"] < 5:   # e.g. role-matched with a single truth witness: no eligible match in most blocks
                ax.text(k, 0.005, f"n={m['n']}", ha="center", va="bottom", fontsize=9, rotation=90)
                rows.append({"cell": cell, "comparator": c, **m, "pred": cc[c]["pred"], "plotted": False})
                continue
            ax.bar(k, m["mean"], color=colors[k], edgecolor="black", lw=1.2, zorder=2)
            ax.text(k, max(m["hi"], 0) + 0.01, f"{m['n']}", ha="center", va="bottom", fontsize=8, color=PALETTE["gray_dark"])
            ax.errorbar(k, m["mean"], yerr=[[m["mean"] - m["lo"]], [m["hi"] - m["mean"]]], color="black", capsize=4, lw=1.5)
            ax.scatter(k, cc[c]["pred"], marker="D", color="white", edgecolor="black", zorder=4, s=40)
            rows.append({"cell": cell, "comparator": c, **m, "pred": cc[c]["pred"], "plotted": True})
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(comps))); ax.set_xticklabels([labels[c] for c in comps], rotation=55, ha="right", fontsize=11)
        ax.set_title(cell.replace("|", "  "), fontsize=13)
    axes[0].set_ylabel("full-mute loss D at round 10")
    fig.suptitle("Bars: observed (95% block bootstrap, n blocks above); diamonds: frozen neutral prediction; yellow: A/A contrast 95% band; comparators with n<5 omitted", fontsize=12)
    pd.DataFrame(rows).to_csv(FIG / f"{stage}_comparators.csv", index=False)
    finalize_figure(fig, FIG / f"{stage}_comparators")

    # 2) early / late / full for named targets
    rows = []
    fig, axes = create_subplots(1, len(cells), figsize=(4.2 * len(cells), 5), sharey=True)
    for ax, cell in zip(axes, cells):
        c = s["cells"][cell]
        for j, who in enumerate(("T", "R")):
            for k, kind in enumerate(("early", "late", "full")):
                m = c[f"{kind}_named_{who}"] if kind != "full" else c[f"full_named_{who}_subset"]
                x = j * 4 + k
                ax.bar(x, m["mean"], color=[PALETTE["blue_secondary"], PALETTE["teal"], PALETTE["blue_main"]][k] if who == "T"
                       else [PALETTE["red_1"], PALETTE["red_2"], PALETTE["red_strong"]][k], edgecolor="black", lw=1.2)
                ax.errorbar(x, m["mean"], yerr=[[m["mean"] - m["lo"]], [m["hi"] - m["mean"]]], color="black", capsize=3, lw=1.3)
                rows.append({"cell": cell, "who": who, "kind": kind, **m})
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks([0, 1, 2, 4, 5, 6]); ax.set_xticklabels(["T early", "T late", "T full", "R early", "R late", "R full"], rotation=55, ha="right", fontsize=11)
        ax.set_title(cell.replace("|", "  "), fontsize=13)
    axes[0].set_ylabel("loss D at round 10 (subset blocks)")
    pd.DataFrame(rows).to_csv(FIG / f"{stage}_windows.csv", index=False)
    finalize_figure(fig, FIG / f"{stage}_windows")

    # 3) ranking scores
    if "ranking" in s:
        cols = ["rho_pred_full_mute", "rho_pred_chi", "rho_temporal_closeness", "rho_private_accuracy", "rho_paper_S", "rho_eligible_contacts"]
        names = ["neutral full-mute", "neutral chi", "temporal closeness", "private accuracy", "paper S", "eligible contacts"]
        rows = []
        fig, axes = create_subplots(1, len(s["ranking"]), figsize=(4.2 * len(s["ranking"]), 5), sharey=True)
        for ax, (cell, r) in zip(axes, s["ranking"].items()):
            for k, col in enumerate(cols):
                m = r.get(col)
                if not m or m["n"] == 0:
                    continue
                ax.bar(k, m["mean"], color=DEFAULT[k % len(DEFAULT)], edgecolor="black", lw=1.2)
                ax.errorbar(k, m["mean"], yerr=[[m["mean"] - m["lo"]], [m["hi"] - m["mean"]]], color="black", capsize=3, lw=1.3)
                rows.append({"cell": cell, "score": col, **m})
            ax.axhline(0, color="black", lw=1)
            ax.set_xticks(range(len(cols))); ax.set_xticklabels(names, rotation=55, ha="right", fontsize=11)
            ax.set_title(cell.replace("|", "  "), fontsize=13)
        axes[0].set_ylabel("Spearman rho with observed full-mute loss\n(within block, single replicate)")
        pd.DataFrame(rows).to_csv(FIG / f"{stage}_ranking.csv", index=False)
        finalize_figure(fig, FIG / f"{stage}_ranking")

    # 4) response vs rounds for the named truth witness in the clamped cohorts: fixed count vs fixed fraction
    eff = pd.read_csv(sdir / "analysis" / "effects.csv")
    pred_q = json.loads((sdir / "predictions.json").read_text())["q"]
    sel = eff[(eff.kind == "mute_window") & eff.window.isna() & (eff.target == eff.named_T) & eff.cohort.str.startswith("clamped")]
    rows = []
    fig, axes = create_subplots(1, 1, figsize=(7.5, 5.5))
    ax = axes[0]
    cmap = {("clamped_rare", 8): PALETTE["blue_main"], ("clamped_rare", 16): PALETTE["teal"], ("clamped_redundant", 16): PALETTE["red_strong"]}
    for (cohort, N), g in sel.groupby(["cohort", "N"]):
        m = g.groupby("round")["D"].agg(["mean", "sem", "count"]).reset_index()
        ax.errorbar(m["round"], m["mean"], yerr=1.96 * m["sem"], fmt="o-", color=cmap[(cohort, N)], capsize=4, label=f"{cohort.split('_')[1]} N={N} (live)")
        z = 1 if cohort == "clamped_rare" else N // 8
        rr = np.arange(0, 11)
        theo = [th.closed_form_loss(N, z, z, pred_q, 1.0, r * N, 0, False) for r in rr]
        ax.plot(rr, theo, color=cmap[(cohort, N)], ls="--", lw=1.8, alpha=0.8)
        for _, r in m.iterrows():
            rows.append({"cohort": cohort, "N": N, "round": r["round"], "D_mean": r["mean"], "D_sem": r["sem"], "n_blocks": r["count"]})
    ax.axhline(0, color="black", lw=1); ax.axvline(10, color=PALETTE["gray_dark"], ls=":", lw=1.5)
    ax.set_xlabel("readout round"); ax.set_ylabel("loss from muting the clamped truth witness")
    ax.legend(fontsize=10); ax.set_title(f"dashed: stationary neutral theory, frozen q={pred_q:.2f}", fontsize=12)
    pd.DataFrame(rows).to_csv(FIG / f"{stage}_witness_vs_rounds.csv", index=False)
    finalize_figure(fig, FIG / f"{stage}_witness_vs_rounds")


def stage3(model: str, stage: str = "stage3") -> None:
    """Clock and recovery: sustained vs restored mute of the clamped truth witness at N=16, readouts to T_late."""
    apply_publication_style(); FIG.mkdir(exist_ok=True)
    sdir = env.ROOT / "data" / "live" / model / stage
    br = pd.read_csv(sdir / "analysis" / "by_round.csv")
    man = json.loads((sdir / "manifest.json").read_text())
    q = json.loads((sdir / "predictions.json").read_text())["q"]
    N, T = 16, man["T_late"]
    fig, axes = create_subplots(1, 1, figsize=(8.5, 5.5)); ax = axes[0]
    aa = br[br.arm == "aa"].sort_values("round")
    ax.fill_between(aa["round"].clip(lower=0.5), aa["lo"], aa["hi"], color=PALETTE["highlight"], alpha=0.25, lw=0, label="A/A 95% band")
    for arm, c, lab in (("mute:", PALETTE["blue_main"], "sustained mute (live)"), ("restore", PALETTE["red_strong"], "muted rounds 1-5, then restored (live)")):
        g = br[br.arm == arm].sort_values("round")
        x = g["round"].clip(lower=0.5)
        ax.errorbar(x, g["mean"], yerr=[g["mean"] - g["lo"], g["hi"] - g["mean"]], fmt="o-", color=c, capsize=4, label=lab)
    rr = np.unique(np.r_[np.linspace(0.5, T, 400), np.arange(1, T + 1)])
    sus = [th.closed_form_loss(N, 1, 1, q, 1.0, int(round(r * N)), 0, False) for r in rr]
    res = [th.closed_form_loss(N, 1, 1, q, 1.0, int(min(r, 5) * N), int(max(r - 5, 0) * N), False) for r in rr]
    ax.plot(rr, sus, color=PALETTE["blue_main"], ls="--", lw=1.6, alpha=0.8, label=f"neutral theory, q={q:.2f}")
    ax.plot(rr, res, color=PALETTE["red_strong"], ls="--", lw=1.6, alpha=0.8)
    ax.axhline(0, color="black", lw=1); ax.axvline(10, color=PALETTE["gray_dark"], ls=":", lw=1.5)
    ax.set_xscale("log"); ax.set_xlabel("readout round (log)"); ax.set_ylabel("loss from muting the clamped truth witness")
    ax.set_title(f"N=16, 1 truth + 1 rival clamped witness, T_late={T} rounds (model-based horizon)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    br.to_csv(FIG / f"{stage}_clock_recovery.csv", index=False)
    finalize_figure(fig, FIG / f"{stage}_clock_recovery")


def memory(model: str, stages=("stage3", "stage3_h32")) -> None:
    """Stage 3 (8-report FIFO) vs the 32-report variant on identical blocks. Left: loss from muting the clamped truth
    witness by readout round, sustained (filled) and restored after round 5 (open), with the H-independent neutral
    prediction at the frozen q (the neutral copier has no memory length). Right: baseline lock-in, the share of games
    whose 14 flexible agents all give one label, and with at most one dissenter."""
    from .analyze import compare_memory
    apply_publication_style(); FIG.mkdir(exist_ok=True)
    df = compare_memory(model, stages)
    sdir = env.ROOT / "data" / "live" / model / stages[0]
    q = json.loads((sdir / "predictions.json").read_text())["q"]
    T = json.loads((sdir / "manifest.json").read_text())["T_late"]
    N = 16
    fig, axes = create_subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    col = {8: PALETTE["gray_dark"], 32: PALETTE["blue_main"]}
    off = {8: 0.94, 32: 1.06}   # horizontal offset on the log axis so the error bars don't overlap
    for H in (8, 32):
        for arm, fmt, lab in (("mute:", "o-", "sustained mute"), ("restore", "s--", "restored after round 5")):
            g = df[(df.H == H) & (df.arm == arm)].sort_values("round")
            if g.empty:
                continue
            x = g["round"].clip(lower=0.5) * off[H]
            ax.errorbar(x, g["mean"], yerr=[g["mean"] - g["lo"], g["hi"] - g["mean"]], fmt=fmt, color=col[H], capsize=3,
                        mfc=(col[H] if arm == "mute:" else "white"), alpha=(1 if arm == "mute:" else 0.8), label=f"H={H}: {lab}")
    rr = np.unique(np.r_[np.linspace(0.5, T, 400), np.arange(1, T + 1)])
    ax.plot(rr, [th.closed_form_loss(N, 1, 1, q, 1.0, int(round(r * N)), 0, False) for r in rr], color=PALETTE["red_strong"],
            ls=":", lw=2, label=f"neutral theory, sustained (q={q:.2f})")
    ax.axhline(0, color="black", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("readout round (log)"); ax.set_ylabel("loss from muting the clamped truth witness")
    ax.set_title("N=16, 32 identical games per memory length (live gpt-6-luna)", fontsize=12); ax.legend(fontsize=9, loc="upper left")
    ax = axes[1]
    for H in (8, 32):
        g = df[(df.H == H) & (df.arm == "baseline_lockin")].sort_values("round")
        x = g["round"].clip(lower=0.5)
        ax.plot(x, g["consensus"], "o-", color=col[H], label=f"H={H}: all 14 flexible agree")
        ax.plot(x, g["near_consensus"], "^:", color=col[H], alpha=0.7, label=f"H={H}: at most one dissenter")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_xlabel("readout round (log)"); ax.set_ylabel("share of baseline games")
    ax.set_title("lock-in of the unmuted swarm", fontsize=12); ax.legend(fontsize=9, loc="lower right")
    df.to_csv(FIG / "stage3_memory_h8_vs_h32.csv", index=False)
    finalize_figure(fig, FIG / "stage3_memory_h8_vs_h32")


def population(model: str) -> None:
    """Named clamped truth-witness full mute at round 10 vs N: fixed witness count (rare) vs fixed fraction
    (redundant), live (Stages 1-2) with neutral theory at the frozen q."""
    apply_publication_style(); FIG.mkdir(exist_ok=True)
    from .analyze import boot_mean
    q = json.loads((env.ROOT / "data" / "live" / model / "predictor.json").read_text())["q"]
    rows = []
    for stage in ("stage1", "stage2"):
        e = pd.read_csv(env.ROOT / "data" / "live" / model / stage / "analysis" / "effects.csv")
        e = e[(e.missing == False) & (e["round"] == 10) & e.arm.str.startswith("mute:") & (e.target == e.named_T) & e.cohort.str.startswith("clamped")]
        for (N, cohort), g in e.groupby(["N", "cohort"]):
            rows.append({"stage": stage, "N": N, "cohort": cohort, **boot_mean(g, "D")})
    df = pd.DataFrame(rows)
    # at N=8 the rare and redundant constructions coincide: show the N=8 point on both curves
    n8 = df[(df.N == 8) & (df.cohort == "clamped_rare")]
    if len(n8):
        df = pd.concat([df, n8.assign(cohort="clamped_redundant", shared_N8=True)], ignore_index=True)
    fig, axes = create_subplots(1, 1, figsize=(7.5, 5.5)); ax = axes[0]
    for cohort, c, lab in (("clamped_rare", PALETTE["blue_main"], "fixed count (1+1 witnesses)"), ("clamped_redundant", PALETTE["red_strong"], "fixed fraction (N/8 each)")):
        g = df[df.cohort == cohort].sort_values("N")
        ax.errorbar(g["N"] * (1.03 if cohort.endswith("redundant") else 0.97), g["mean"], yerr=[g["mean"] - g["lo"], g["hi"] - g["mean"]],
                    fmt="o", color=c, capsize=4, ms=8, label=f"{lab} (live)")
        Ns = np.array([8, 16, 32, 64])
        z = [1 if cohort == "clamped_rare" else max(1, n // 8) for n in Ns]
        ax.plot(Ns, [th.closed_form_loss(n, zz, zz, q, 1.0, 10 * n, 0, False) for n, zz in zip(Ns, z)], color=c, ls="--", lw=1.6)
    ax.axhline(0, color="black", lw=1); ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32, 64]); ax.set_xticklabels(["8", "16", "32", "64"])
    ax.set_xlabel("population size N"); ax.set_ylabel("loss at round 10\n(mute one truth witness)")
    ax.set_title(f"dashed: stationary neutral theory, frozen q={q:.2f}", fontsize=12); ax.legend(fontsize=10)
    df.to_csv(FIG / "population_witness_loss.csv", index=False)
    finalize_figure(fig, FIG / "population_witness_loss")


DEFAULT = [PALETTE[k] for k in ("blue_main", "teal", "green_3", "violet", "red_strong", "neutral")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["offline", "calibration", "stage1", "stage3", "population", "memory"])
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--stage", default="stage1")
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    {"offline": lambda: offline(), "calibration": lambda: calibration(a.model), "stage1": lambda: stage1(a.model, a.stage), "stage3": lambda: stage3(a.model), "population": lambda: population(a.model), "memory": lambda: memory(a.model)}[a.what]()
