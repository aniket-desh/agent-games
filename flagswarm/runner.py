"""Run a manifest: blocks in parallel, all arms of a block in parallel (sharing one replicate namespace per arm
namespace), resumable. A finished branch is written once to branches/<block>__<arm>.json; a block's calls go to
calls/<block>.jsonl and are reloaded on resume, so nothing already answered is billed twice.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from . import env
from .game import run_branch
from .llm import BudgetExceeded, CallLog, OpenAIResponses, ScriptedCopying, Spend, ledger_record, remaining_budget
from .manifest import block_from_json


def safe(s: str) -> str:
    return s.replace("|", "__").replace(">", "-").replace(":", "_")


def write_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


async def run_manifest(manifest_path: Path, backend: str = "live", max_usd: float | None = None, concurrency: int = 256,
                       block_concurrency: int = 16, scripted: dict | None = None, out_dir: Path | None = None,
                       only_blocks: set[str] | None = None) -> dict:
    man = json.loads(Path(manifest_path).read_text())
    model = man["model"]
    out = out_dir or Path(manifest_path).parent
    (out / "branches").mkdir(parents=True, exist_ok=True)
    (out / "calls").mkdir(parents=True, exist_ok=True)
    source = "live" if backend == "live" else "scripted"
    spend = Spend(model, remaining_budget(max_usd)) if backend == "live" else None
    root = OpenAIResponses(model, None, spend, concurrency=concurrency) if backend == "live" else None
    resolved = {"manifest": str(manifest_path), "source": source, "backend": backend, "model": model if backend == "live" else "scripted",
                "max_usd": spend.max_usd if spend else None, "concurrency": concurrency, "block_concurrency": block_concurrency,
                "scripted": scripted, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "source_snapshot": env.source_snapshot(),
                "causal_convention": man["causal_convention"]}
    (out / f"resolved_config_{int(time.time())}.json").write_text(json.dumps(resolved, indent=1))
    gate = asyncio.Semaphore(block_concurrency)
    stop = asyncio.Event()
    stats = {"blocks_done": 0, "branches_written": 0, "branches_skipped": 0, "fresh_calls": 0, "shared": 0, "resumed": 0}
    t0 = time.time()

    async def do_block(bj: dict) -> None:
        if stop.is_set() or (only_blocks and bj["block_id"] not in only_blocks):
            return
        todo = [a for a in bj["arms"] if not (out / "branches" / f"{safe(bj['block_id'])}__{safe(a['name'])}.json").exists()]
        stats["branches_skipped"] += len(bj["arms"]) - len(todo)
        if not todo:
            return
        async with gate:
            if stop.is_set():
                return
            block, arms = block_from_json(bj)
            arms = [a for a in arms if a.name in {t["name"] for t in todo}]
            log = CallLog(out / "calls" / f"{safe(block.block_id)}.jsonl")
            if backend == "live":
                be = OpenAIResponses(model, log, spend, shared_from=root)
            else:
                be = ScriptedCopying(block.truth, block.rival, scripted["p_crop"][f"{block.truth}>{block.rival}"], scripted["q"], log=log)

            async def one(arm):
                res = await run_branch(block, arm, be)
                rec = {"source": source, "block_id": block.block_id, "arm": arm.name, "arm_spec": arm.__dict__,
                       "truth": block.truth, "rival": block.rival, "candidates": block.candidates, "crops": block.crops,
                       "roles": block.roles, "N": block.N, "readout_rounds": block.readout_rounds,
                       "events": res.events, "readouts": res.readouts}
                write_atomic(out / "branches" / f"{safe(block.block_id)}__{safe(arm.name)}.json", rec)
                stats["branches_written"] += 1

            try:
                await asyncio.gather(*[one(a) for a in arms])
            except BudgetExceeded:
                stop.set()
            finally:
                log.close()
                stats["fresh_calls"] += be.n_fresh; stats["shared"] += be.n_shared; stats["resumed"] += be.n_resumed
                stats["blocks_done"] += 1
                el = time.time() - t0
                print(f"[{el:7.0f}s] block {stats['blocks_done']} {block.block_id} fresh={be.n_fresh} shared={be.n_shared} "
                      f"resumed={be.n_resumed} spend=${spend.usd if spend else 0:.3f}", flush=True)

    try:
        await asyncio.gather(*[do_block(b) for b in man["blocks"]])
    finally:
        if spend is not None:
            ledger_record(f"{man['stage']}:{model}", spend)
    stats.update({"elapsed_s": round(time.time() - t0, 1), "budget_stopped": stop.is_set(),
                  "spend": spend.summary() if spend else None,
                  "failed_attempts": root.failed_attempts if root else 0})
    (out / f"run_stats_{int(time.time())}.json").write_text(json.dumps(stats, indent=1))
    return stats
