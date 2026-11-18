"""Trial manifests: blocks (truth case, cohort, crops, slot assignment, schedule, candidate order) and arms
(interventions), generated from the frozen panel with a separate seed stream per randomization. Also the cost
estimate printed before any run.

Seed streams (each derived from the manifest seed and a stream name, so changing one never shifts another):
  crops      crop draws within behavior classes / natural positions
  slots      crop-to-slot assignment
  schedule   directed contacts
  order      candidate order (counterbalanced by block index; independent of truth)
  masks      drop uniforms (game.Arm.dropped hashes block|target|slot|mask_seed)
  subset     which blocks get the omission / window / dose arms
  comparator random-speaker comparator draws (analysis)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from . import env
from .game import Arm, Block

CAUSAL_CONVENTION = ("Interventions act on delivery only: a dropped report appends nothing, evicts nothing, gives no sign "
                     "of silence and consumes its slot; the muted agent still listens and answers readouts. Loss "
                     "D = E[Y(baseline) - Y(intervention)] in truth share at the readout; positive = the speaker helped.")


def stream(seed: int, name: str, *parts) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{seed}|{name}|{'|'.join(map(str, parts))}".encode()).hexdigest()[:16], 16))


def load_panel(model: str) -> dict:
    return json.loads((env.ROOT / "data" / "live" / model / "panel.json").read_text())


def class_draw(rng: random.Random, entries: list[dict], k: int) -> list[str]:
    """Sample k hashes with replacement within one validated class, weighted by crop-position multiplicity
    renormalized within the class."""
    if k == 0:
        return []
    w = [e["multiplicity"] for e in entries]
    return rng.choices([e["hash"] for e in entries], weights=w, k=k)


def make_block(panel: dict, case: str, cohort: str, N: int, b: int, rounds: int, readout_rounds: list[int], seed: int,
               m: int = 3, tag: str = "") -> tuple[Block, dict]:
    """cohort: natural (uniform original positions, no behavior selection), clamped_rare (1 clamped truth + 1 clamped
    rival witness + N-2 live persuadable-crop agents), clamped_redundant (N/8 of each witness, 3N/4 persuadable)."""
    truth, rival = case.split(">")
    pc = panel["cases"][case]
    co = panel["crop_only"][case]
    rng_c = stream(seed, "crops", tag, case, cohort, N, b)
    clamped = None
    if cohort == "natural":
        boxes = [(rng_c.randrange(env.CANVAS_H - env.TILE_H + 1), rng_c.randrange(env.CANVAS_W - env.TILE_W + 1)) for _ in range(N)]
        crops = [env.crop_png_for_position(truth, t, l)[0] for t, l in boxes]
        roles = ["N"] * N
    elif cohort in ("clamped_rare", "clamped_redundant"):
        nW = 1 if cohort == "clamped_rare" else max(1, N // 8)
        roles = ["T"] * nW + ["R"] * nW + ["P"] * (N - 2 * nW)
        # witnesses never call the model; they hold a persuadable crop only so every slot has an image
        crops = class_draw(rng_c, pc["P"], N)
        perm = list(range(N))
        stream(seed, "slots", tag, case, cohort, N, b).shuffle(perm)
        roles = [roles[i] for i in perm]
        clamped = {i: (truth if r == "T" else rival) for i, r in enumerate(roles) if r in ("T", "R")}
    else:
        raise ValueError(cohort)
    p_crop = [1.0 if r == "T" else 0.0 if r == "R" else co[h]["p_truth"] for h, r in zip(crops, roles)]
    order = b % 2   # counterbalanced by block index, independent of truth
    candidates = [truth, rival] if order == 0 else [rival, truth]
    sched_seed = stream(seed, "schedule", tag, case, cohort, N, b).randrange(2**31)
    schedule = env.generate_pair_schedule(N, rounds * N, sched_seed)
    pngs = {h: env.png_bytes_by_hash(h, [truth]) for h in set(crops)}
    blk = Block(f"{tag}{case}|{cohort}|N{N}|b{b:03d}", truth, rival, candidates, crops, pngs, schedule, readout_rounds,
                m=m, roles=roles, clamped=clamped)
    # Schedule-blind named targets: clamped witnesses, or in natural games the most / least truth-favoring crop.
    rng_n = stream(seed, "named", blk.block_id)
    if clamped:
        named = {"T": next(i for i, r in enumerate(roles) if r == "T"), "R": next(i for i, r in enumerate(roles) if r == "R")}
    else:
        hi, lo = max(p_crop), min(p_crop)
        named = {"T": rng_n.choice([i for i, p in enumerate(p_crop) if p == hi]),
                 "R": rng_n.choice([i for i, p in enumerate(p_crop) if p == lo])}
    extra = {"cohort": cohort, "case": case, "N": N, "b": b, "p_crop": p_crop, "named": named, "order": order}
    return blk, extra


def stage1_arms(block: Block, extra: dict, subset: bool, seed: int) -> list[Arm]:
    """Baseline, independent A/A, full mute of every speaker; on subset blocks also early/late windows for both
    named targets, and doses plus single-message omissions for the named truth-favoring target."""
    N, S = block.N, len(block.schedule)
    arms = [Arm("baseline"), Arm("aa", "aa", namespace="aa")]
    arms += [Arm(f"mute:{i}", "mute_window", target=i) for i in range(N)]
    if subset:
        half = S // 2
        for i in sorted(set(extra["named"].values())):
            arms += [Arm(f"early:{i}", "mute_window", target=i, window=(0, half)),
                     Arm(f"late:{i}", "mute_window", target=i, window=(half, S))]
        i = extra["named"]["T"]
        arms += [Arm(f"dose{h}:{i}", "drop_message", target=i, h=h, mask_seed=1) for h in (0.1, 0.25, 0.5)]
        elig = [t for t in range(S) if block.schedule[t][0] == i]
        pick = sorted(stream(seed, "subset", block.block_id, "omit").sample(elig, min(4, len(elig))))
        arms += [Arm(f"omit{t}:{i}", "drop_message", target=i, slots=(t,)) for t in pick]
    return arms


def block_to_json(block: Block, arms: list[Arm], extra: dict) -> dict:
    d = asdict(block)
    d.pop("pngs")
    d["arms"] = [asdict(a) for a in arms]
    d["extra"] = extra
    return d


def block_from_json(d: dict) -> tuple[Block, list[Arm]]:
    d = dict(d)
    d.pop("extra", None)
    arms = [Arm(**{**a, "window": tuple(a["window"]) if a["window"] else None, "slots": tuple(a["slots"]) if a["slots"] else None}) for a in d.pop("arms")]
    d["schedule"] = [tuple(x) for x in d["schedule"]]
    if d.get("clamped"):
        d["clamped"] = {int(k): v for k, v in d["clamped"].items()}
    d["pngs"] = {h: env.png_bytes_by_hash(h, [d["truth"]]) for h in set(d["crops"])}
    return Block(**d), arms


def write_manifest(model: str, name: str, blocks: list[dict], seed: int, rounds: int, extra: dict | None = None) -> Path:
    out = env.ROOT / "data" / "live" / model / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists():
        raise SystemExit(f"{path} exists; manifests are immutable once written")
    man = {"stage": name, "model": model, "seed": seed, "rounds": rounds, "causal_convention": CAUSAL_CONVENTION,
           "panel_sha256": env.sha((env.ROOT / "data" / "live" / model / "panel.json").read_bytes()),
           "upstream": env.upstream_provenance(), "source_snapshot": env.source_snapshot(), **(extra or {}), "blocks": blocks}
    path.write_text(json.dumps(man))
    return path


def build_stage1(model: str, Ns=(8, 16), blocks_per_cell: int = 32, subset_blocks: int = 8, rounds: int = 10,
                 seed: int = 1, name: str = "stage1") -> Path:
    panel = load_panel(model)
    blocks = []
    for N in Ns:
        cohorts = ["natural", "clamped_rare"] + (["clamped_redundant"] if N >= 16 else [])   # rare == redundant at N=8
        for cohort in cohorts:
            cases = panel["clamped_cases"] if cohort.startswith("clamped") else list(panel["cases"])
            sub = set(stream(seed, "subset", N, cohort).sample(range(blocks_per_cell), subset_blocks))
            for b in range(blocks_per_cell):
                case = cases[(b // 2) % len(cases)]   # balanced across ordered cases, both candidate orders per case
                blk, extra = make_block(panel, case, cohort, N, b, rounds, [0, rounds // 2, rounds], seed)
                blocks.append(block_to_json(blk, stage1_arms(blk, extra, b in sub, seed), extra))
    return write_manifest(model, name, blocks, seed, rounds, {"readout_rounds": [0, rounds // 2, rounds]})


# ---------------------------------------------------------------------------------------------
# Cost estimate (pre-sharing upper bound, no retries)
# ---------------------------------------------------------------------------------------------

def estimate(man: dict, in_tok: float, out_tok: float, price=(0.10, 0.50), rpm: float = 30000) -> dict:
    """N(r + Q) calls per branch; tokens per call from measured usage. Shared pre-fork calls make the real
    figure lower; the realized cost is logged per run."""
    calls = 0
    branches = 0
    for b in man["blocks"]:
        N = len(b["crops"])
        per_branch = N * (len(b["schedule"]) // N + len(b["readout_rounds"]))
        calls += per_branch * len(b["arms"])
        branches += len(b["arms"])
    usd = calls * (in_tok * price[0] + out_tok * price[1]) / 1e6
    return {"projection": True, "blocks": len(man["blocks"]), "branches": branches, "calls_upper": calls,
            "tokens_per_call": {"input": in_tok, "output": out_tok}, "usd_upper": round(usd, 2),
            "minutes_at_rate_limit": round(calls / rpm, 1)}


def estimate_table(in_tok: float, out_tok: float, price=(0.10, 0.50)) -> list[dict]:
    """The spec's illustrative rows, re-priced with measured tokens (projection)."""
    per = (in_tok * price[0] + out_tok * price[1]) / 1e6
    rows = [
        ("individual discovery+confirmation", 8 * 12 * 3 * 8 + 8 * 9 * 3 * 24),
        ("exhaustive N=8, 1 cohort, 32 blocks, 10 arms, 10 rounds, Q=3", 8 * 13 * 10 * 32),
        ("exhaustive N=16, 1 cohort, 32 blocks, 18 arms, 10 rounds, Q=3", 16 * 13 * 18 * 32),
        ("short sweep 6 sizes x 2 cohorts x 64 blocks x 5 arms", sum(N * 13 * 5 * 64 * 2 for N in (8, 16, 32, 64, 128, 256))),
        ("N=128 Z=2 rare, 64 blocks, baseline+mute, 762 rounds, Q=4", 128 * 766 * 2 * 64),
        ("N=256 Z=2 rare, 64 blocks, baseline+mute, 1530 rounds, Q=4", 256 * 1534 * 2 * 64),
    ]
    return [{"row": r, "calls": c, "usd": round(c * per, 2), "projection": True} for r, c in rows]


def rate_matched_slots(schedule: list, target: int) -> tuple[int, ...]:
    """For each outgoing slot of the target, the nearest unused slot (ties: earlier) whose speaker is someone else:
    the same number of omissions at matched times, from other speakers."""
    used, out = set(), []
    for t in [t for t, (s, _) in enumerate(schedule) if s == target]:
        for d in range(1, len(schedule)):
            cands = [u for u in (t - d, t + d) if 0 <= u < len(schedule) and schedule[u][0] != target and u not in used]
            if cands:
                used.add(cands[0]); out.append(cands[0]); break
    return tuple(sorted(out))


def build_stage4(model: str, source_stage: str = "stage1", N: int = 8, name: str = "stage4") -> Path:
    """Mechanism checks on blocks copied from the source stage (identical crops, schedule, candidate order):
    m=3: baseline, mute named T / R, visible placeholder for named T, rate-/time-matched omissions from others;
    m=1: baseline, A/A, mute named T / R (country-only bandwidth, no reasons)."""
    src = json.loads((env.ROOT / "data" / "live" / model / source_stage / "manifest.json").read_text())
    blocks = []
    for bj in src["blocks"]:
        if bj["extra"]["N"] != N:
            continue
        T, R = bj["extra"]["named"]["T"], bj["extra"]["named"]["R"]
        for m in (3, 1):
            d = json.loads(json.dumps(bj))
            d["block_id"] = f"m{m}:{bj['block_id']}"
            d["m"] = m
            d["extra"] = {**bj["extra"], "m": m, "source_block": bj["block_id"]}
            arms = [Arm("baseline"), Arm(f"mute:{T}", "mute_window", target=T), Arm(f"mute:{R}", "mute_window", target=R)]
            if m == 3:
                arms += [Arm(f"placeholder:{T}", "placeholder", target=T),
                         Arm(f"ratematched:{T}", "drop_message", target=T, slots=rate_matched_slots(d["schedule"], T))]
            else:
                arms += [Arm("aa", "aa", namespace="aa")]
            d["arms"] = [asdict(a) for a in arms]
            blocks.append(d)
    return write_manifest(model, name, blocks, src["seed"], src["rounds"], {"readout_rounds": src["readout_rounds"],
                          "source_stage": source_stage, "purpose": "mechanism checks (Stage 4)"})


def build_stage2(model: str, Ns=(32, 64), blocks_per_cell=(32, 16), rounds: int = 10, seed: int = 4, name: str = "stage2") -> Path:
    """Population stage at 10 rounds: named truth and rival witnesses, predicted top, uniform random and
    opportunity-matched speakers (all chosen from pre-intervention information with the frozen predictor),
    plus baseline and A/A. Natural cohort only at the smallest N."""
    from .analyze import comparator_targets
    from .qcal import block_predictions
    panel = load_panel(model)
    q = json.loads((env.ROOT / "data" / "live" / model / "predictor.json").read_text())["q"]
    blocks = []
    for N, nb in zip(Ns, blocks_per_cell):
        cohorts = ["clamped_rare", "clamped_redundant"] + (["natural"] if N == Ns[0] else [])
        for cohort in cohorts:
            cases = panel["clamped_cases"] if cohort.startswith("clamped") else list(panel["cases"])
            for b in range(nb):
                case = cases[(b // 2) % len(cases)]
                blk, extra = make_block(panel, case, cohort, N, b, rounds, [0, rounds // 2, rounds], seed, tag="s2:")
                bj = block_to_json(blk, [], extra)
                ct = comparator_targets(bj, block_predictions(bj, q), seed)
                targets = sorted({t for t in ct.values() if t is not None})
                arms = [Arm("baseline"), Arm("aa", "aa", namespace="aa")] + [Arm(f"mute:{t}", "mute_window", target=t) for t in targets]
                bj["arms"] = [asdict(a) for a in arms]
                blocks.append(bj)
    return write_manifest(model, name, blocks, seed, rounds, {"readout_rounds": [0, rounds // 2, rounds]})


def build_stage3(model: str, N: int = 16, blocks: int = 32, restore_after: int = 5, seed: int = 5, name: str = "stage3",
                 H: int = 8, aa: bool = True) -> Path:
    """Clock and recovery: from the same initial state, keep the clamped truth witness muted for the whole game or
    restore it after `restore_after` rounds. Horizon T_late = ceil(3 (N-1) / (q (Z-1))) rounds with the frozen q
    (Z = 2), readouts at 0, 5, 10, 20, 50, 100 and T_late (post-restoration times 5+5, 5+15, ...).
    H sets the FIFO length (memory variant). seed and tag stay "s3:", so crops, schedules and clamped witnesses match
    Stage 3 block for block and only the memory differs. aa=False drops the A/A arm."""
    from .theory import late_horizon_rounds
    panel = load_panel(model)
    q = json.loads((env.ROOT / "data" / "live" / model / "predictor.json").read_text())["q"]
    T = late_horizon_rounds(N, q, 2)
    ro = sorted({0, 5, 10, 20, 50, 100, T})
    out = []
    for b in range(blocks):
        case = panel["clamped_cases"][(b // 2) % len(panel["clamped_cases"])]
        blk, extra = make_block(panel, case, "clamped_rare", N, b, T, ro, seed, tag="s3:")
        blk.H = H
        i = extra["named"]["T"]
        arms = [Arm("baseline")] + ([Arm("aa", "aa", namespace="aa")] if aa else []) + [Arm(f"mute:{i}", "mute_window", target=i),
                Arm(f"restore{restore_after}:{i}", "mute_window", target=i, window=(0, restore_after * N))]
        out.append(block_to_json(blk, arms, extra))
    return write_manifest(model, name, out, seed, T, {"readout_rounds": ro, "T_late": T, "q_planning": q, "H": H})
