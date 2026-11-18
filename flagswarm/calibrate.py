"""Live preflight and individual-behavior calibration (crop panel discovery, confirmation, qualification, freeze).

Every number written here comes from provider calls (source="live") unless marked otherwise.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import env
from .llm import CallLog, OpenAIResponses, Request, Spend, ledger_record, remaining_budget

TEMPLATE_REASON = env.TEMPLATE_REASON


def template_line(country: str) -> str:
    return f"{country} | {TEMPLATE_REASON}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - w, c + w)


def live_dir(model: str) -> Path:
    return env.ROOT / "data" / "live" / model


# ---------------------------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------------------------

async def preflight(model: str, n_parallel: int = 32) -> dict:
    out = live_dir(model) / "preflight"
    spend = Spend(model, remaining_budget(2.0))
    be = OpenAIResponses(model, CallLog(out / "calls.jsonl"), spend, concurrency=n_parallel)
    t0 = time.time()
    cls = env.crop_classes("France")
    blue = next(c for c in cls if c.positions[0] == (0, 0))
    mem8 = [template_line("France" if i % 2 else "Peru") for i in range(8)]
    reqs = [
        Request("report", 3, ["France", "Peru"], [], blue.hash, blue.png, "preflight", "report_empty"),
        Request("report", 3, ["France", "Peru"], mem8, blue.hash, blue.png, "preflight", "report_mem8"),
        Request("readout", 1, ["France", "Peru"], mem8, blue.hash, blue.png, "preflight", "readout_mem8"),
        Request("fullflag", 1, ["Peru", "France"], [], env.sha(env.full_flag_png("France")), env.full_flag_png("France"), "preflight", "fullflag"),
    ]
    single = []
    for r in reqs:
        a = await be(r)
        single.append({"event": r.event, "valid": a.valid, "error": a.error, "country": a.country, "reason": a.reason,
                       "usage": a.usage, "returned_model": a.meta.get("returned_model"), "latency_s": a.meta.get("latency_s")})
    # throughput probe: n_parallel independent draws of the full-memory report
    t1 = time.time()
    par = await asyncio.gather(*[be(Request("report", 3, ["France", "Peru"], mem8, blue.hash, blue.png, "preflight", f"par:{i}")) for i in range(n_parallel)])
    wall = time.time() - t1
    be.log.close()
    usage_by_event = {s["event"]: s["usage"] for s in single}
    res = {
        "source": "live", "model_requested": model, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "single": single,
        "parallel": {"n": n_parallel, "wall_s": round(wall, 2), "valid": sum(a.valid for a in par),
                     "errors": [a.error for a in par if not a.valid],
                     "mean_latency_s": float(np.mean([a.meta.get("latency_s", np.nan) for a in par if a.valid] or [np.nan])),
                     "mean_input_tokens": float(np.mean([a.usage.get("input_tokens", 0) for a in par if a.valid] or [0])),
                     "mean_output_tokens": float(np.mean([a.usage.get("output_tokens", 0) for a in par if a.valid] or [0])),
                     "france_share": float(np.mean([a.country == "France" for a in par if a.valid] or [np.nan]))},
        "ratelimit_headers": be.ratelimit_headers, "failed_attempts": be.failed_attempts,
        "spend": spend.summary(), "elapsed_s": round(time.time() - t0, 1), "upstream": env.upstream_provenance(),
    }
    (out / "preflight.json").write_text(json.dumps(res, indent=1))
    ledger_record(f"preflight:{model}", spend)
    return res


# ---------------------------------------------------------------------------------------------
# Individual calibration: full-flag control, crop discovery, confirmation, qualification
# ---------------------------------------------------------------------------------------------

REGIMES = {"empty": (0, 0), "truth_heavy": (6, 2), "rival_heavy": (2, 6)}   # (truth reports, rival reports)
THRESH = {"witness_empty": 0.85, "witness_opposed": 0.75, "persuadable_rise": 0.25}
DISCOVERY_MAX_HASHES, DISCOVERY_DRAWS, CONFIRM_PER_ROLE, CONFIRM_DRAWS = 12, 8, 3, 24


def regime_memory(truth: str, rival: str, regime: str, rng: random.Random) -> list[str]:
    nT, nR = REGIMES[regime]
    lines = [template_line(truth)] * nT + [template_line(rival)] * nR
    rng.shuffle(lines)   # shuffle memory position
    return lines


def discovery_hashes(truth: str, rival: str, seed: int) -> list[env.CropClass]:
    """Up to 12 distinct crop images of the TRUE flag, stratified by compatibility (shared with the rival or
    truth-only) and native frequency (multiplicity tercile), proportional allocation, recorded seed."""
    cls = list(env.crop_classes(truth))
    if len(cls) <= DISCOVERY_MAX_HASHES:
        return cls
    rng = random.Random(seed)
    strata = defaultdict(list)
    mults = sorted(c.multiplicity for c in cls)
    for c in cls:
        compat = env.compatibility(c.hash, [truth, rival])
        tercile = min(2, int(3 * mults.index(c.multiplicity) / len(mults)))
        strata[(compat, tercile)].append(c)
    keep = []
    quota = {k: DISCOVERY_MAX_HASHES * len(v) / len(cls) for k, v in strata.items()}
    base = {k: int(math.floor(q)) for k, q in quota.items()}
    extra = DISCOVERY_MAX_HASHES - sum(base.values())
    for k in sorted(quota, key=lambda k: (quota[k] - base[k], rng.random()), reverse=True)[:extra]:
        base[k] += 1
    for k, v in strata.items():
        keep += rng.sample(v, min(base[k], len(v)))
    return sorted(keep, key=lambda c: c.positions[0])


async def run_probes(be, jobs: list[dict]) -> list[dict]:
    async def one(j):
        a = await be(Request("report", 3, j["candidates"], j["memory"], j["hash"], j["png"], j["ns"], "probe"))
        return {k: v for k, v in j.items() if k not in ("png", "memory")} | {
            "memory": j["memory"], "country": a.country, "valid": a.valid, "error": a.error,
            "reason": a.reason, "usage": a.usage, "returned_model": a.meta.get("returned_model")}
    return await asyncio.gather(*[one(j) for j in jobs])


def probe_jobs(stage: str, truth: str, rival: str, hashes: list[env.CropClass], draws: int, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}|{stage}|{truth}|{rival}")
    jobs = []
    for c in hashes:
        for regime in REGIMES:
            for order in (0, 1):
                cands = [truth, rival] if order == 0 else [rival, truth]
                for d in range(draws // 2):
                    jobs.append({"stage": stage, "truth": truth, "rival": rival, "hash": c.hash, "png": c.png,
                                 "multiplicity": c.multiplicity, "compat": list(env.compatibility(c.hash, [truth, rival])),
                                 "regime": regime, "order": order, "candidates": cands, "draw": d,
                                 "memory": regime_memory(truth, rival, regime, rng),
                                 "ns": f"calib:{stage}:{truth}>{rival}:{c.hash[:16]}:{regime}:{order}:{d}"})
    return jobs


def rates(rows: list[dict]) -> dict:
    """Truth probability by (hash, regime) pooled over orders, plus order-specific rates and Wilson intervals."""
    out = defaultdict(dict)
    by = defaultdict(list)
    for r in rows:
        by[(r["hash"], r["regime"])].append(r)
        by[(r["hash"], r["regime"], r["order"])].append(r)
    for key, rs in by.items():
        valid = [r for r in rs if r["valid"]]
        k = sum(r["country"] == r["truth"] for r in valid)
        n = len(valid)
        entry = {"k_truth": k, "n": n, "n_invalid": len(rs) - n, "p_truth": k / n if n else float("nan"), "wilson": wilson(k, n)}
        if len(key) == 2:
            out[key[0]][key[1]] = entry
        else:
            out[key[0]][f"{key[1]}|order{key[2]}"] = entry
    return dict(out)


def classify(r: dict) -> dict:
    """Role scores from truth probabilities. Positive score = meets the threshold."""
    pe, pt, pr = r["empty"]["p_truth"], r["truth_heavy"]["p_truth"], r["rival_heavy"]["p_truth"]
    T = min(pe - THRESH["witness_empty"], pr - THRESH["witness_opposed"])            # truth label despite 6 rival / 2 truth
    R = min((1 - pe) - THRESH["witness_empty"], (1 - pt) - THRESH["witness_opposed"])  # rival label despite 6 truth / 2 rival
    P = min(pt - pr - THRESH["persuadable_rise"], 0.5 - pr, pt - 0.5)                  # rises >= .25 and crosses .5
    return {"T": T, "R": R, "P": P}


async def calibrate(model: str, seed: int = 20261118, max_usd: float = 5.0, concurrency: int = 128) -> dict:
    out = live_dir(model) / "calibration"
    if (live_dir(model) / "panel.json").exists():
        raise SystemExit("panel.json already frozen; refusing to recalibrate over it")
    spend = Spend(model, remaining_budget(max_usd))
    log = CallLog(out / "calls.jsonl")
    be = OpenAIResponses(model, log, spend, concurrency=concurrency)
    t0 = time.time()

    # Full-flag control: 8 directed cases x 2 orders x 4 draws = 64 balanced calls, empty memory, country only.
    ff_jobs = []
    for truth, rival in env.DIRECTED_CASES:
        png = env.full_flag_png(truth)
        for order in (0, 1):
            for d in range(4):
                cands = [truth, rival] if order == 0 else [rival, truth]
                ff_jobs.append((truth, rival, order, d, Request("fullflag", 1, cands, [], env.sha(png), png, f"calib:fullflag:{truth}>{rival}:{order}:{d}", "probe")))
    ff_ans = await asyncio.gather(*[be(j[-1]) for j in ff_jobs])
    fullflag = [{"truth": t, "rival": r, "order": o, "draw": d, "country": a.country, "valid": a.valid, "error": a.error}
                for (t, r, o, d, _), a in zip(ff_jobs, ff_ans)]
    ff_k = sum(x["country"] == x["truth"] for x in fullflag)

    # Discovery
    disc_hashes = {case: discovery_hashes(*case, seed) for case in env.DIRECTED_CASES}
    disc_rows = []
    for case, hs in disc_hashes.items():
        disc_rows += await run_probes(be, probe_jobs("discovery", *case, hs, DISCOVERY_DRAWS, seed))
    disc_rates = {case: rates([r for r in disc_rows if (r["truth"], r["rival"]) == case]) for case in env.DIRECTED_CASES}

    # Carry forward up to 3 candidate hashes per role per case (best discovery scores; drop hopeless ones).
    carried = {}
    for case, rr in disc_rates.items():
        scored = {h: classify(v) for h, v in rr.items()}
        carried[case] = {role: [h for h, _ in sorted(scored.items(), key=lambda kv: -kv[1][role]) if scored[h][role] > -0.3][:CONFIRM_PER_ROLE]
                         for role in ("T", "R", "P")}
    conf_rows = []
    for case, roles in carried.items():
        hs = sorted({h for v in roles.values() for h in v})
        cls = [c for c in env.crop_classes(case[0]) if c.hash in hs]
        conf_rows += await run_probes(be, probe_jobs("confirm", *case, cls, CONFIRM_DRAWS, seed))
    conf_rates = {case: rates([r for r in conf_rows if (r["truth"], r["rival"]) == case]) for case in env.DIRECTED_CASES}
    log.close()

    # Qualification on the confirmatory probes only.
    qualified = {}
    for case, rr in conf_rates.items():
        q = {"T": [], "R": [], "P": []}
        for role, hs in carried[case].items():
            for h in hs:
                if h in rr and classify(rr[h])[role] >= 0:
                    q[role].append(h)
        qualified[f"{case[0]}>{case[1]}"] = q

    all_rows = disc_rows + conf_rows
    n_valid = sum(r["valid"] for r in all_rows) + sum(x["valid"] for x in fullflag)
    n_all = len(all_rows) + len(fullflag)
    res = {
        "source": "live", "model": model, "seed": seed, "thresholds": THRESH, "template_reason": TEMPLATE_REASON,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_s": round(time.time() - t0, 1),
        "fullflag": {"k": ff_k, "n": len(fullflag), "accuracy": ff_k / len(fullflag), "wilson": wilson(ff_k, len(fullflag)),
                     "gate_0.90": ff_k / len(fullflag) >= 0.90, "rows": fullflag},
        "validity": {"valid": n_valid, "n": n_all, "rate": n_valid / n_all, "gate_0.99": n_valid / n_all >= 0.99,
                     "errors": sorted({r["error"] for r in all_rows if not r["valid"]} | {x["error"] for x in fullflag if not x["valid"]})},
        "discovery_hashes": {f"{t}>{r}": [c.hash for c in hs] for (t, r), hs in disc_hashes.items()},
        "discovery_rates": {f"{t}>{r}": v for (t, r), v in disc_rates.items()},
        "carried": {f"{t}>{r}": v for (t, r), v in carried.items()},
        "confirm_rates": {f"{t}>{r}": v for (t, r), v in conf_rates.items()},
        "qualified": qualified, "spend": spend.summary(), "returned_models": sorted({r["returned_model"] for r in all_rows if r["returned_model"]}),
        "upstream": env.upstream_provenance(), "source_snapshot": env.source_snapshot(),
    }
    (out / "calibration.json").write_text(json.dumps(res, indent=1, default=str))
    with (out / "probes.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    ledger_record(f"calibration:{model}", spend)
    return res


async def crop_only(model: str, draws: int = 16, seed: int = 20261118, concurrency: int = 128) -> dict:
    """Isolated re-query of every distinct crop of every truth flag against its rival: empty memory, m=3 report
    prompt, draws/2 per candidate order. Initializes the neutral predictor (p_i) and the paper benchmark."""
    out = live_dir(model) / "calibration"
    spend = Spend(model, remaining_budget(1.0))
    log = CallLog(out / "crop_only_calls.jsonl")
    be = OpenAIResponses(model, log, spend, concurrency=concurrency)
    rows = []
    for truth, rival in env.DIRECTED_CASES:
        jobs = []
        for c in env.crop_classes(truth):
            for order in (0, 1):
                cands = [truth, rival] if order == 0 else [rival, truth]
                for d in range(draws // 2):
                    jobs.append({"stage": "crop_only", "truth": truth, "rival": rival, "hash": c.hash, "png": c.png,
                                 "multiplicity": c.multiplicity, "compat": list(env.compatibility(c.hash, [truth, rival])),
                                 "regime": "empty", "order": order, "candidates": cands, "draw": d, "memory": [],
                                 "ns": f"calib:crop_only:{truth}>{rival}:{c.hash[:16]}:{order}:{d}"})
        rows += await run_probes(be, jobs)
    log.close()
    res = {"source": "live", "model": model, "draws": draws, "spend": spend.summary(), "rates": {}}
    for truth, rival in env.DIRECTED_CASES:
        rr = rates([r for r in rows if (r["truth"], r["rival"]) == (truth, rival)])
        res["rates"][f"{truth}>{rival}"] = {h: {"p_truth": v["empty"]["p_truth"], "n": v["empty"]["n"], "wilson": v["empty"]["wilson"],
                                                "p_order0": v["empty|order0"]["p_truth"], "p_order1": v["empty|order1"]["p_truth"],
                                                "multiplicity": env.native_multiplicity(h, truth)} for h, v in rr.items()}
    (out / "crop_only.json").write_text(json.dumps(res, indent=1))
    ledger_record(f"crop_only:{model}", spend)
    return res


def freeze_panel(model: str) -> dict:
    """Write panel.json from the confirmatory probes and crop-only rates. Refuses to overwrite."""
    path = live_dir(model) / "panel.json"
    if path.exists():
        raise SystemExit("panel.json already frozen")
    cal = json.loads((live_dir(model) / "calibration" / "calibration.json").read_text())
    co = json.loads((live_dir(model) / "calibration" / "crop_only.json").read_text())
    cases = {}
    for case, q in cal["qualified"].items():
        truth = case.split(">")[0]
        entry = {}
        for role in ("T", "R", "P"):
            entry[role] = [{"hash": h, "multiplicity": env.native_multiplicity(h, truth),
                            "confirm_rates": {k: cal["confirm_rates"][case][h][k]["p_truth"] for k in REGIMES}} for h in q[role]]
        entry["status"] = "viable" if all(entry[r] for r in ("T", "R", "P")) else \
            "missing:" + ",".join(r for r in ("T", "R", "P") if not entry[r])
        cases[case] = entry
    viable = [c for c, e in cases.items() if e["status"] == "viable"]
    panel = {
        "source": "live", "model": model, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "thresholds": THRESH,
        "template_reason": TEMPLATE_REASON, "calibration_seed": cal["seed"],
        "gates": {"fullflag_accuracy": cal["fullflag"]["accuracy"], "fullflag_pass": cal["fullflag"]["gate_0.90"],
                  "validity_rate": cal["validity"]["rate"], "validity_pass": cal["validity"]["gate_0.99"]},
        "cases": cases, "viable_cases": viable,
        "controlled_study": "available" if len(viable) >= 2 else ("single-case" if len(viable) == 1 else "unavailable"),
        "clamped_cases": [c for c, e in cases.items() if e["P"]],
        "crop_only": co["rates"],
        "note": ("No case has a behavior-verified rival-favoring witness; true-favoring witnesses exist only for France>Peru. "
                 "Luna yields to a 6-2 opposing majority on almost every crop (a social ceiling, retained as a measurement). "
                 "The two-witness question is therefore studied in the separately labeled clamped_witness hybrid "
                 "(witness reports fixed, listeners are live persuadable-crop agents) plus the natural cohort."),
        "calibration_sha256": env.sha((live_dir(model) / "calibration" / "calibration.json").read_bytes()),
    }
    path.write_text(json.dumps(panel, indent=1))
    return panel
