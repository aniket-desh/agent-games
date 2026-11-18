"""Listener calibration and the frozen neutral predictor (source="live" for probes; predictions are model outputs).

Separate calibration games (tag "qcal:", own seed) supply real incoming messages. For each sampled delivery
i -> j we probe the listener's truth probability with its copied memory before and after appending the message
(FIFO-truncated to 8), and also from zero history (empty vs the single message). Then fit, by least squares,
    p_after - p_before ~= q (1[incoming = truth] - p_before)
with q in [0, 1], plus separate truth/rival gains as a diagnostic. The fit is frozen to predictor.json before any
intervention is run.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np

from . import env
from . import theory as th
from .calibrate import live_dir
from .llm import CallLog, OpenAIResponses, Request, Spend, ledger_record, remaining_budget
from .manifest import block_to_json, load_panel, make_block, stage1_arms, stream, write_manifest
from .runner import run_manifest


def build_qcal_games(model: str, blocks: int = 8, N: int = 8, rounds: int = 10, seed: int = 2) -> Path:
    panel = load_panel(model)
    out = []
    for cohort, cases in (("natural", list(panel["cases"])), ("clamped_rare", panel["clamped_cases"])):
        for b in range(blocks):
            blk, extra = make_block(panel, cases[b % len(cases)], cohort, N, b, rounds, [0, rounds // 2, rounds], seed, tag="qcal:")
            from .game import Arm
            out.append(block_to_json(blk, [Arm("baseline")], extra))
    return write_manifest(model, "qcal_games", out, seed, rounds, {"purpose": "listener calibration only; never used for endpoints"})


def sample_deliveries(branch: dict, k: int, rng: random.Random) -> list[dict]:
    """Delivered reports into non-clamped listeners, half truth-labelled and half rival-labelled where possible."""
    clamped = {int(i) for i, r in enumerate(branch["roles"]) if r in ("T", "R")} if any(r in ("T", "R") for r in branch["roles"]) else set()
    mem: dict[int, list[str]] = {a: [] for a in range(branch["N"])}
    cand = []
    for e in branch["events"]:
        line = f"{e['country']} | {e['reason']}" if e["delivered"] else None
        if line is not None and e["listener"] not in clamped:
            cand.append({"t": e["t"], "listener": e["listener"], "speaker": e["speaker"], "line": line,
                         "label_truth": e["country"] == branch["truth"], "memory_before": list(mem[e["listener"]])})
        if line is not None:
            mem[e["listener"]] = (mem[e["listener"]] + [line])[-8:]
    pos = [c for c in cand if c["label_truth"]]
    neg = [c for c in cand if not c["label_truth"]]
    rng.shuffle(pos); rng.shuffle(neg)
    take = pos[: k // 2] + neg[: k // 2]
    take += (pos[k // 2:] + neg[k // 2:])[: k - len(take)]   # top up if one label is scarce (imbalance recorded)
    return take


async def probe_pairs(model: str, per_game: int = 30, draws: int = 4, seed: int = 3, concurrency: int = 128) -> dict:
    gdir = live_dir(model) / "qcal_games"
    man = json.loads((gdir / "manifest.json").read_text())
    spend = Spend(model, remaining_budget(2.0))
    log = CallLog(gdir / "probe_calls.jsonl")
    be = OpenAIResponses(model, log, spend, concurrency=concurrency)
    jobs = []
    for bj in man["blocks"]:
        br = json.loads((gdir / "branches" / f"{bj['block_id'].replace('|', '__').replace('>', '-').replace(':', '_')}__baseline.json").read_text())
        rng = stream(seed, "qcal", bj["block_id"])
        for d in sample_deliveries(br, per_game, rng):
            j = d["listener"]
            h = br["crops"][j]
            png = env.png_bytes_by_hash(h, [br["truth"]])
            ctx = {"full": (d["memory_before"], (d["memory_before"] + [d["line"]])[-8:]), "zero": ([], [d["line"]])}
            for context, (before, after) in ctx.items():
                for cond, mem in (("before", before), ("after", after)):
                    for k in range(draws):
                        order = k % 2
                        cands = [br["truth"], br["rival"]] if order == 0 else [br["rival"], br["truth"]]
                        jobs.append({"block": bj["block_id"], "cohort": bj["extra"]["cohort"], "t": d["t"], "listener": j,
                                     "role": br["roles"][j], "label_truth": d["label_truth"], "context": context, "cond": cond,
                                     "order": order, "draw": k, "truth": br["truth"], "fifo_len_before": len(before),
                                     "req": Request("readout", 1, cands, mem, h, png, f"qcal:{bj['block_id']}:{d['t']}:{context}:{cond}:{k}", "probe")})
    answers = await asyncio.gather(*[be(j["req"]) for j in jobs])
    log.close()
    rows = [{k: v for k, v in j.items() if k != "req"} | {"country": a.country, "valid": a.valid} for j, a in zip(jobs, answers)]
    with (gdir / "probe_rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ledger_record(f"qcal_probes:{model}", spend)
    return {"n_rows": len(rows), "spend": spend.summary(), "valid": sum(r["valid"] for r in rows)}


def fit_q(rows: list[dict]) -> dict:
    """Least squares on per-event mean changes. Returns overall and diagnostic fits with residuals."""
    from collections import defaultdict
    agg = defaultdict(lambda: {"before": [], "after": []})
    meta = {}
    for r in rows:
        if not r["valid"]:
            continue
        key = (r["block"], r["t"], r["context"])
        agg[key][r["cond"]].append(r["country"] == r["truth"])
        meta[key] = r
    ev = []
    for key, v in agg.items():
        if v["before"] and v["after"]:
            m = meta[key]
            ev.append({"context": key[2], "label_truth": m["label_truth"], "role": m["role"], "cohort": m["cohort"],
                       "fifo_len_before": m["fifo_len_before"], "pb": float(np.mean(v["before"])), "pa": float(np.mean(v["after"]))})

    def ls(sub):
        if not sub:
            return None
        x = np.array([(1.0 if e["label_truth"] else 0.0) - e["pb"] for e in sub])
        y = np.array([e["pa"] - e["pb"] for e in sub])
        q = float(np.clip((x @ y) / (x @ x), 0, 1)) if x @ x > 0 else float("nan")
        resid = y - q * x
        # bootstrap over events for an interval
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(500):
            idx = rng.integers(0, len(x), len(x))
            xx, yy = x[idx], y[idx]
            bs.append(np.clip((xx @ yy) / (xx @ xx), 0, 1) if xx @ xx > 0 else np.nan)
        return {"q": q, "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "n_events": len(sub),
                "rmse": float(np.sqrt(np.mean(resid**2))), "r2": float(1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))) if len(y) > 1 else None,
                "mean_dp_truth_msg": float(np.mean([e["pa"] - e["pb"] for e in sub if e["label_truth"]] or [np.nan])),
                "mean_dp_rival_msg": float(np.mean([e["pa"] - e["pb"] for e in sub if not e["label_truth"]] or [np.nan]))}

    return {
        "all": ls(ev),
        "full_context": ls([e for e in ev if e["context"] == "full"]),
        "zero_history": ls([e for e in ev if e["context"] == "zero"]),
        "full_fifo": ls([e for e in ev if e["context"] == "full" and e["fifo_len_before"] == 8]),
        "truth_msgs": ls([e for e in ev if e["label_truth"]]),
        "rival_msgs": ls([e for e in ev if not e["label_truth"]]),
        "natural_listeners": ls([e for e in ev if e["cohort"] == "natural"]),
        "persuadable_listeners": ls([e for e in ev if e["role"] == "P"]),
        "events": ev,
    }


def freeze_predictor(model: str) -> dict:
    gdir = live_dir(model) / "qcal_games"
    path = live_dir(model) / "predictor.json"
    if path.exists():
        raise SystemExit("predictor.json already frozen")
    rows = [json.loads(l) for l in (gdir / "probe_rows.jsonl").open()]
    fit = fit_q(rows)
    q = fit["full_context"]["q"]   # in-game listeners carry a context; zero-history is a diagnostic
    pred = {"source": "live-calibrated model", "model": model, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "q": q, "q_rule": "least squares on full-context events (in-game memory before and after one real message)",
            "planning_q_if_unstable": 0.5, "receiver_gain_clamped": 0.0,
            "fits": {k: v for k, v in fit.items() if k != "events"}}
    path.write_text(json.dumps(pred, indent=1))
    (gdir / "qcal_events.json").write_text(json.dumps(fit["events"]))
    return pred


def block_predictions(bj: dict, q: float) -> dict:
    """Frozen neutral-model predictions for one block: full-mute loss and weak-response susceptibility per speaker,
    early/late windows, plus the paper benchmark scores. Uses only pre-intervention information."""
    N = len(bj["crops"])
    sched = np.array(bj["schedule"])
    p0 = np.array(bj["extra"]["p_crop"], float)
    clamped = {int(k) for k in (bj.get("clamped") or {})}
    gains = np.array([0.0 if i in clamped else q for i in range(N)])
    S = len(sched)
    full = [th.full_mute_loss(p0, sched, gains, i) for i in range(N)]
    chi = th.susceptibility_by_sender(p0, sched, gains).tolist()
    early = [th.full_mute_loss(p0, sched, gains, i, (0, S // 2)) for i in range(N)]
    late = [th.full_mute_loss(p0, sched, gains, i, (S // 2, S)) for i in range(N)]
    E = th.temporal_closeness(N, sched).tolist()
    M = [int((sched[:, 0] == i).sum()) for i in range(N)]
    base = float(th.forward(p0, sched, gains)[-1].mean())
    return {"block_id": bj["block_id"], "q": q, "pred_baseline_Y": base, "pred_full_mute": full, "pred_chi": chi,
            "pred_early": early, "pred_late": late, "temporal_closeness": E, "eligible_contacts": M,
            "top_helpful": int(np.argmax(full)), "p_crop": p0.tolist()}


def freeze_predictions(model: str, stage: str) -> Path:
    sdir = live_dir(model) / stage
    path = sdir / "predictions.json"
    if path.exists():
        raise SystemExit("predictions already frozen")
    q = json.loads((live_dir(model) / "predictor.json").read_text())["q"]
    man = json.loads((sdir / "manifest.json").read_text())
    preds = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "q": q, "blocks": [block_predictions(b, q) for b in man["blocks"]]}
    path.write_text(json.dumps(preds))
    return path
