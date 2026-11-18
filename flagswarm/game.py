"""One game trajectory (a branch) under a fixed block: crops, candidate order, directed contacts, readout rounds.

Causal convention (recorded in every output): an intervention acts only on DELIVERY of a scheduled report.
  drop_message / mute_window: the speaker still produces its report (direct mode) or is skipped (efficient
  mode, logged); the listener's FIFO receives nothing, evicts nothing, gets no sign of silence; the slot is
  consumed; the muted agent still listens and answers readouts.
  placeholder: the listener receives a visible neutral line in place of the report (occupies one FIFO slot).
Memory: each agent holds its last H=8 DELIVERED received reports, oldest first. Speaking and readouts never
change any memory. Calls are stateless: the local context is rebuilt from the delivery log at every call.

Concurrency is dependency-aware: the call at slot t needs only the speaker's memory before t, i.e. the outcomes
of earlier slots whose listener is that speaker. Each slot waits on exactly those, so histories are identical to
sequential execution while independent speakers run in parallel.
"""
from __future__ import annotations

import asyncio
import bisect
import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import env
from .llm import Answer, Backend, Request

PLACEHOLDER = "(no report)"
INTERVENTIONS = ("baseline", "aa", "drop_message", "mute_window", "placeholder", "clamped_witness")


@dataclass
class Block:
    block_id: str
    truth: str
    rival: str | None                 # None for the 28-label transfer
    candidates: list[str]             # presented order
    crops: list[str]                  # image hash per agent slot
    pngs: dict[str, bytes]            # hash -> PNG bytes
    schedule: list[tuple[int, int]]   # (speaker, listener), N slots per round
    readout_rounds: list[int]
    m: int = 3
    H: int = 8
    roles: list[str] | None = None    # e.g. "T", "R", "P" per slot (analysis only; never shown to agents)
    clamped: dict[int, str] | None = None   # clamped_witness hybrid: slot -> fixed label (no model call)

    @property
    def N(self) -> int:
        return len(self.crops)


@dataclass
class Arm:
    name: str
    kind: str = "baseline"
    target: int | None = None
    window: tuple[int, int] | None = None   # slot indices [a, b); None = whole game
    h: float = 1.0                           # drop probability per eligible outgoing report
    slots: tuple[int, ...] | None = None     # explicit eligible slots (single-message omission)
    mask_seed: int = 0
    namespace: str = "main"                  # replicate namespace; A/A uses its own
    efficient: bool = False                  # skip the speaker call for known-dropped reports (logged)

    def eligible(self, block: Block) -> list[int]:
        if self.target is None or self.kind in ("baseline", "aa"):
            return []
        if self.slots is not None:
            return list(self.slots)
        a, b = self.window or (0, len(block.schedule))
        return [t for t in range(a, b) if block.schedule[t][0] == self.target]

    def dropped(self, block: Block) -> set[int]:
        """Independent uniform per (block, target, slot) shared across doses, so masks are nested in h."""
        out = set()
        for t in self.eligible(block):
            u = _uniform(f"{block.block_id}|{self.target}|{t}|{self.mask_seed}")
            if u < self.h:
                out.add(t)
        return out


def _uniform(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16) / 16**15


@dataclass
class BranchResult:
    block_id: str
    arm: str
    namespace: str
    events: list[dict] = field(default_factory=list)
    readouts: list[dict] = field(default_factory=list)


async def run_branch(block: Block, arm: Arm, backend: Backend, window_slots: int | None = None) -> BranchResult:
    S, N, H = len(block.schedule), block.N, block.H
    sched = block.schedule
    drop = arm.dropped(block)
    incoming: list[list[int]] = [[] for _ in range(N)]
    for t, (_, l) in enumerate(sched):
        incoming[l].append(t)
    delivered: list[asyncio.Future] = [asyncio.get_running_loop().create_future() for _ in range(S)]
    events: list[dict | None] = [None] * S
    ns = f"{block.block_id}|{arm.namespace}"

    async def memory_before(agent: int, t: int) -> list[str]:
        """Last H delivered lines among slots < t addressed to agent (scan backwards; FIFO semantics)."""
        idx = incoming[agent]
        k = bisect.bisect_left(idx, t) - 1
        out: list[str] = []
        while k >= 0 and len(out) < H:
            line = await delivered[idx[k]]
            if line is not None:
                out.append(line)
            k -= 1
        return out[::-1]

    async def slot(t: int) -> None:
        s, l = sched[t]
        try:
            mem = await memory_before(s, t)
            is_drop = t in drop
            if block.clamped and s in block.clamped:
                ans = Answer(block.clamped[s], env.TEMPLATE_REASON if block.m == 3 else None, True, meta={"clamped": True})
            elif is_drop and arm.efficient and arm.kind != "placeholder":
                ans = Answer(None, None, False, "skipped_dropped")
            else:
                ans = await backend(Request("report", block.m, block.candidates, mem, block.crops[s],
                                            block.pngs[block.crops[s]], ns, f"slot:{t}:agent:{s}"))
            line = ans.memory_entry()
            if is_drop:
                line_delivered = PLACEHOLDER if arm.kind == "placeholder" else None
            else:
                line_delivered = line
            events[t] = {"t": t, "round": t // N + 1, "speaker": s, "listener": l, "dropped": is_drop,
                         "delivered": line_delivered is not None, "valid": ans.valid, "error": ans.error,
                         "country": ans.country, "reason": ans.reason, "speaker_memory_len": len(mem)}
            delivered[t].set_result(line_delivered)
        except BaseException as exc:
            if not delivered[t].done():
                delivered[t].set_exception(exc)
            raise

    async def readout(r: int, agent: int) -> dict:
        t = r * N
        mem = await memory_before(agent, t)
        m = block.m if (r == 0 and block.m == 3) else 1   # upstream: m=3 schema at initialization, country-only after
        if block.clamped and agent in block.clamped:
            return {"round": r, "agent": agent, "country": block.clamped[agent], "valid": True, "error": None,
                    "memory_len": len(mem), "fifo_full": len(mem) == H, "clamped": True}
        ans = await backend(Request("readout", m, block.candidates, mem, block.crops[agent],
                                    block.pngs[block.crops[agent]], ns, f"readout:{r}:agent:{agent}"))
        return {"round": r, "agent": agent, "country": ans.country, "valid": ans.valid, "error": ans.error,
                "memory_len": len(mem), "fifo_full": len(mem) == H}

    window_slots = window_slots or 4 * N
    gate = asyncio.Semaphore(window_slots)
    tasks: list[asyncio.Task] = []
    readout_tasks: list[asyncio.Task] = []

    async def gated(t: int) -> None:
        try:
            await slot(t)
        finally:
            gate.release()

    rounds = set(block.readout_rounds)
    try:
        for t in range(S + 1):
            if t % N == 0 and (t // N) in rounds:
                readout_tasks += [asyncio.create_task(readout(t // N, a)) for a in range(N)]
            if t == S:
                break
            await gate.acquire()
            tasks.append(asyncio.create_task(gated(t)))
        await asyncio.gather(*tasks)
        readouts = await asyncio.gather(*readout_tasks)
    except BaseException:
        for task in tasks + readout_tasks:
            task.cancel()
        raise

    # FIFO occupancy and evictions, reconstructed from the delivery log (diagnostic for the memory side effect).
    occ = [0] * N
    for e in events:
        l = e["listener"]
        e["listener_fifo_before"] = occ[l]
        e["evicted"] = e["delivered"] and occ[l] == H
        if e["delivered"]:
            occ[l] = min(H, occ[l] + 1)
    return BranchResult(block.block_id, arm.name, arm.namespace, events, list(readouts))


def truth_share(readouts: list[dict], truth: str, round_: int, exclude: int | None = None) -> float:
    """Fraction of agents naming the truth at a readout round; invalid votes count as not-truth here and are
    reported separately (bounds are computed in analysis)."""
    rows = [r for r in readouts if r["round"] == round_ and r["agent"] != exclude]
    return float(np.mean([r["country"] == truth for r in rows]))


def plurality_score(readouts: list[dict], truth: str, round_: int) -> float:
    """1 if truth is the unique plurality, 1/k if it is in a k-way tie, else 0 (mean of uniform tie-breaking)."""
    from collections import Counter
    c = Counter(r["country"] for r in readouts if r["round"] == round_ and r["valid"])
    if not c:
        return float("nan")
    top = max(c.values())
    winners = [k for k, v in c.items() if v == top]
    return 1.0 / len(winners) if truth in winners else 0.0
