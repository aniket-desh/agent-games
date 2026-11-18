"""Replay-harness invariants with scripted backends (no provider calls)."""
import asyncio
import random

import pytest

from flagswarm import env
from flagswarm.game import Arm, Block, run_branch
from flagswarm.llm import Answer, Backend, CallLog, Request, ScriptedCopying


class Tracer(Backend):
    """Every report carries a unique reason naming its namespace and event, so memory contents can be traced."""
    source = "scripted"

    def __init__(self):
        super().__init__(None)
        self.requests: list[Request] = []

    def params(self):
        return {"model": "tracer"}

    async def _sample(self, req):
        self.requests.append(req)
        rng = random.Random(req.key(self.params()))
        label = req.candidates[rng.random() < 0.5]
        await asyncio.sleep(0)  # let other branches interleave
        return Answer(label, f"{req.namespace}/{req.event}/{rng.random():.6f}" if req.m == 3 else None, True)


def make_block(N=6, rounds=6, seed=5, H=8):
    cls = env.crop_classes("France")
    crops = [cls[i % len(cls)].hash for i in range(N)]
    return Block("blk", "France", "Peru", ["Peru", "France"], crops, {c.hash: c.png for c in cls},
                 env.generate_pair_schedule(N, N * rounds, seed), [0, rounds // 2, rounds], H=H)


def run(*coros):
    return asyncio.run(_gather(*coros))


async def _gather(*coros):
    return await asyncio.gather(*coros)


def report_memory(be, ns_suffix):
    """event -> memory for report requests of one branch namespace."""
    return {r.event: r.memory for r in be.requests if r.kind == "report" and r.namespace.endswith(ns_suffix)}


def test_drop_consumes_slot_appends_nothing_and_muted_agent_listens_and_votes():
    b = make_block()
    be = Tracer()
    target = b.schedule[0][0]
    base, muted = run(run_branch(b, Arm("base"), be), run_branch(b, Arm("mute", "mute_window", target=target), be))
    assert len(muted.events) == len(base.events) == len(b.schedule)
    drops = [e for e in muted.events if e["dropped"]]
    assert drops and all(e["speaker"] == target and not e["delivered"] for e in drops)
    assert len(drops) == sum(s == target for s, _ in b.schedule)
    # the muted target keeps receiving and still answers every readout
    assert any(e["listener"] == target and e["delivered"] for e in muted.events)
    assert all(r["valid"] for r in muted.readouts if r["agent"] == target)
    assert len([r for r in muted.readouts if r["agent"] == target]) == len(b.readout_rounds)


def test_memory_holds_only_last_8_delivered_reports_and_no_readouts():
    b = make_block(N=4, rounds=12)   # 48 slots, ~12 incoming per agent > H
    be = Tracer()
    res, = run(run_branch(b, Arm("mute", "mute_window", target=1, h=0.5, mask_seed=2), be))
    delivered_lines = {}
    for e in res.events:
        delivered_lines[e["t"]] = (e["listener"], f"{e['country']} | {e['reason']}" if e["delivered"] else None)
    for r in be.requests:
        agent = int(r.event.split(":")[-1])
        t = int(r.event.split(":")[1]) * (b.N if r.kind == "readout" else 1)
        expect = [line for k, (l, line) in sorted(delivered_lines.items()) if k < t and l == agent and line is not None][-8:]
        assert r.memory == expect
        assert all("readout" not in line for line in r.memory)   # readout answers never enter memory
    assert max(len(r.memory) for r in be.requests) == 8
    assert any(e["evicted"] for e in res.events)


def test_h0_target_placement_has_no_effect_with_shared_randomness():
    b = make_block()
    be = ScriptedCopying("France", "Peru", {h: 0.5 for h in b.pngs}, 0.5)
    base, *arms = run(run_branch(b, Arm("base"), be), *[run_branch(b, Arm(f"h0_{i}", "mute_window", target=i, h=0.0), be) for i in range(b.N)])
    for a in arms:
        assert [r["country"] for r in a.readouts] == [r["country"] for r in base.readouts]
    assert be.n_fresh == len(b.schedule) + b.N * len(b.readout_rounds)


def test_branches_share_until_fork_then_diverge():
    b = make_block(N=6, rounds=8)
    be = Tracer()
    target = 2
    base, muted = run(run_branch(b, Arm("base"), be), run_branch(b, Arm("mute", "mute_window", target=target), be))
    fork = min(e["t"] for e in muted.events if e["dropped"])
    for e0, e1 in zip(base.events[:fork + 1], muted.events[:fork + 1]):
        assert (e0["country"], e0["reason"]) == (e1["country"], e1["reason"])   # identical histories up to the first drop
    listener = b.schedule[fork][1]
    nxt = next(t for t in range(fork + 1, len(b.schedule)) if b.schedule[t][0] == listener)
    assert base.events[nxt]["speaker_memory_len"] == muted.events[nxt]["speaker_memory_len"] + 1 or base.events[nxt]["speaker_memory_len"] == 8
    # the next speaker after the fork received different histories, so its report is a fresh sample in each branch
    ev = f"slot:{nxt}:agent:{listener}"
    seen = [r.memory for r in be.requests if r.event == ev]
    assert len(seen) == 2 and seen[0] != seen[1]
    assert base.events[nxt]["reason"] != muted.events[nxt]["reason"]


def test_aa_namespaces_draw_independent_samples():
    b = make_block()
    be = Tracer()
    main, aa = run(run_branch(b, Arm("base"), be), run_branch(b, Arm("aa", "aa", namespace="aa"), be))
    assert be.n_shared == 0 and be.n_resumed == 0
    # the first slot has identical input (empty memory) in both, yet the answers are separate samples
    assert main.events[0]["reason"] != aa.events[0]["reason"]
    assert be.n_fresh == 2 * (len(b.schedule) + b.N * len(b.readout_rounds))


def test_call_log_resume_reproduces_without_new_calls(tmp_path):
    b = make_block()
    log = CallLog(tmp_path / "calls.jsonl")
    be = ScriptedCopying("France", "Peru", {h: 0.5 for h in b.pngs}, 0.5, log=log)
    first, = run(run_branch(b, Arm("mute", "mute_window", target=0), be))
    log.close()
    be2 = ScriptedCopying("France", "Peru", {h: 0.5 for h in b.pngs}, 0.5, log=CallLog(tmp_path / "calls.jsonl"))
    again, = run(run_branch(b, Arm("mute", "mute_window", target=0), be2))
    assert be2.n_fresh == 0
    assert again.readouts == first.readouts


def test_ambiguous_crop_requests_are_byte_identical_across_truths():
    """Leakage guard: a crop shared by both flags yields the same request payload whatever the hidden truth."""
    shared = set(c.hash for c in env.crop_classes("France")) & set(c.hash for c in env.crop_classes("Peru"))
    h = sorted(shared)[0]
    png_fr = next(c.png for c in env.crop_classes("France") if c.hash == h)
    png_pe = next(c.png for c in env.crop_classes("Peru") if c.hash == h)
    assert png_fr == png_pe
    r1 = Request("readout", 1, ["France", "Peru"], [], h, png_fr, "ns", "e")
    r2 = Request("readout", 1, ["France", "Peru"], [], h, png_pe, "ns", "e")
    assert r1.text() == r2.text() and r1.key({}) == r2.key({})
