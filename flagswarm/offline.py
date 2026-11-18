"""Offline stage (source="offline"): planted-control Monte Carlo vs the exact recursion, both schedulers, and the
fixed-witness-count vs fixed-witness-fraction curves that the live population stage is compared against.

Outputs data/offline/<run>/{resolved_config.json, planted_vs_theory.csv, scaling_curves.csv}. Plots live in plot.py.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import env
from . import theory as th

OUT = env.ROOT / "data" / "offline"


def planted_vs_theory(seed: int = 11, reps: int = 20000) -> pd.DataFrame:
    """Mute the truth witness for 5 rounds then restore, and a sustained mute; readouts every round to 20.
    Both schedulers; N in {8, 16, 32}; z_T = z_R = 1; q = 1/2; initial mean at the stationary value 1/2."""
    rows = []
    for self_contacts in (False, True):
        for N in (8, 16, 32):
            g = th.PlantedGame(N, 1, 1, 0.5, self_contacts)
            R = 20
            for label, h in (("window_1to5", np.r_[np.ones(5 * N), np.zeros(15 * N)]), ("sustained", np.ones(R * N))):
                ro = [r * N for r in range(R + 1)]
                sim = th.simulate_planted(g, h, reps, seed + N + 100 * self_contacts, readout_slots=ro)
                xb = th.mean_recursion(N, 1, 1, 0.5, np.zeros(R * N), 0.5, self_contacts)
                xm = th.mean_recursion(N, 1, 1, 0.5, h, 0.5, self_contacts)
                for c, slot in enumerate(ro):
                    d = sim["Y_base"][:, c] - sim["Y_muted"][:, c]
                    rows.append({"source": "offline", "self_contacts": self_contacts, "N": N, "profile": label,
                                 "round": slot // N, "mc_loss": d.mean(), "mc_se": d.std(ddof=1) / np.sqrt(reps), "reps": reps,
                                 "theory_loss": th.expected_truth_share(N, 1, 1, xb[slot]) - th.expected_truth_share(N, 1, 1, xm[slot])})
    return pd.DataFrame(rows)


def scaling_curves(q: float = 0.5) -> pd.DataFrame:
    """Exact neutral-model loss of a sustained mute of one truth witness vs rounds, for fixed witness count
    (z_T = z_R = 1) and fixed witness fraction (z_T = z_R = N/8), N = 8..256. Self-contacts excluded."""
    rows = []
    for N in (8, 16, 32, 64, 128, 256):
        for regime, z in (("fixed_count", 1), ("fixed_fraction", max(1, N // 8))):
            Z = 2 * z
            T = 3000 if z == 1 else 40
            for r in range(T + 1):
                L = r * N
                loss = th.closed_form_loss(N, z, z, q, 1.0, L, 0, False) if Z > 1 else np.nan
                rows.append({"source": "offline", "N": N, "regime": regime, "z_T": z, "z_R": z, "q": q, "round": r, "loss": loss})
    return pd.DataFrame(rows)


def run(name: str = "v1") -> dict:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pvt = planted_vs_theory()
    pvt.to_csv(out / "planted_vs_theory.csv", index=False)
    sc = scaling_curves()
    sc.to_csv(out / "scaling_curves.csv", index=False)
    z = ((pvt.mc_loss - pvt.theory_loss) / pvt.mc_se.replace(0, np.nan)).abs()
    summary = {"source": "offline", "max_abs_z_mc_vs_theory": float(z.max()), "frac_abs_z_gt_3": float((z > 3).mean()),
               "n_points": int(len(pvt)), "elapsed_s": round(time.time() - t0, 1)}
    (out / "resolved_config.json").write_text(json.dumps({"stage": "offline", "name": name, "summary": summary,
                                                          "source_snapshot": env.source_snapshot()}, indent=1))
    return summary
