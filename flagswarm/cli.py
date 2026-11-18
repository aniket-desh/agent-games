"""One entry point for every stage. Each live command prints a cost estimate first and respects the session cap.

  python -m flagswarm.cli offline                      # planted control + scaling curves (no provider calls)
  python -m flagswarm.cli preflight  --model gpt-6-luna
  python -m flagswarm.cli calibrate  --model gpt-6-luna   # full-flag control, discovery, confirmation
  python -m flagswarm.cli crop-only  --model gpt-6-luna
  python -m flagswarm.cli freeze-panel --model gpt-6-luna
  python -m flagswarm.cli qcal       --model gpt-6-luna   # listener games + probes + frozen predictor
  python -m flagswarm.cli manifest   --model gpt-6-luna --stage stage1 [--Ns 8 16]
  python -m flagswarm.cli manifest   --model gpt-6-luna --stage stage3_h32 --H 32 --no-aa   # memory variant
  python -m flagswarm.cli estimate   --model gpt-6-luna --stage stage1
  python -m flagswarm.cli run        --model gpt-6-luna --stage stage1 --max-usd 30 [--scripted]
  python -m flagswarm.cli resume     ...same as run: finished branches are skipped, logged calls reloaded
  python -m flagswarm.cli analyze    --model gpt-6-luna --stage stage1
  python -m flagswarm.cli spend
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from . import env


def measured_tokens(model: str) -> tuple[float, float]:
    """Mean billed tokens per call over every live call log of this model (falls back to the spec's assumption)."""
    import glob
    it = ot = n = 0
    for f in glob.glob(str(env.ROOT / "data" / "live" / model / "**" / "*calls*.jsonl"), recursive=True) + \
             glob.glob(str(env.ROOT / "data" / "live" / model / "**" / "calls" / "*.jsonl"), recursive=True):
        with open(f) as fh:
            for line in fh:
                try:
                    u = json.loads(line)["answer"]["usage"]
                except Exception:
                    continue
                if u.get("input_tokens"):
                    it += u["input_tokens"]; ot += u.get("output_tokens", 0); n += 1
    return (it / n, ot / n) if n else (800.0, 40.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--stage", default="stage1")
    ap.add_argument("--Ns", type=int, nargs="*", default=[8, 16])
    ap.add_argument("--blocks", type=int, default=32)
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=384)
    ap.add_argument("--block-concurrency", type=int, default=48)
    ap.add_argument("--H", type=int, default=8, help="FIFO length (stage3* manifests)")
    ap.add_argument("--no-aa", action="store_true", help="omit the A/A arm (stage3* manifests)")
    ap.add_argument("--scripted", action="store_true", help="run the manifest with the scripted copier (data/scripted/)")
    a = ap.parse_args()

    if a.cmd == "offline":
        from . import offline
        print(offline.run())
    elif a.cmd == "preflight":
        from .calibrate import preflight
        print(json.dumps(asyncio.run(preflight(a.model)), indent=1, default=str)[:3000])
    elif a.cmd == "calibrate":
        from .calibrate import calibrate
        r = asyncio.run(calibrate(a.model))
        print(json.dumps({k: r[k] for k in ("validity", "qualified", "spend")}, indent=1, default=str))
    elif a.cmd == "crop-only":
        from .calibrate import crop_only
        print(asyncio.run(crop_only(a.model))["spend"])
    elif a.cmd == "freeze-panel":
        from .calibrate import freeze_panel
        p = freeze_panel(a.model); print(p["controlled_study"], p["viable_cases"])
    elif a.cmd == "qcal":
        from .qcal import build_qcal_games, freeze_predictor, probe_pairs
        from .runner import run_manifest
        p = env.ROOT / "data" / "live" / a.model / "qcal_games" / "manifest.json"
        if not p.exists():
            build_qcal_games(a.model)
        asyncio.run(run_manifest(p, "live", max_usd=1.0))
        print(asyncio.run(probe_pairs(a.model)))
        print(json.dumps(freeze_predictor(a.model)["fits"], indent=1, default=str)[:2000])
    elif a.cmd == "manifest":
        from .manifest import build_stage1, build_stage3
        from .qcal import freeze_predictions
        if a.stage.startswith("stage3"):
            p = build_stage3(a.model, blocks=a.blocks, name=a.stage, H=a.H, aa=not a.no_aa)
        else:
            p = build_stage1(a.model, Ns=tuple(a.Ns), blocks_per_cell=a.blocks, name=a.stage)
        freeze_predictions(a.model, a.stage)
        print(p)
    elif a.cmd == "estimate":
        from .manifest import estimate, estimate_table
        it, ot = measured_tokens(a.model)
        man = json.loads((env.ROOT / "data" / "live" / a.model / a.stage / "manifest.json").read_text())
        print(json.dumps({"measured_tokens_per_call": [round(it, 1), round(ot, 1)], "stage": estimate(man, it, ot),
                          "spec_rows_repriced": estimate_table(it, ot)}, indent=1))
    elif a.cmd in ("run", "resume"):
        from .llm import ledger_total, remaining_budget
        from .manifest import estimate
        from .runner import run_manifest
        mpath = env.ROOT / "data" / "live" / a.model / a.stage / "manifest.json"
        man = json.loads(mpath.read_text())
        it, ot = measured_tokens(a.model)
        est = estimate(man, it, ot)
        print(f"estimate (projection, pre-sharing): {est}; session spend so far ${ledger_total():.2f}, "
              f"this run may spend ${remaining_budget(a.max_usd):.2f}")
        if a.scripted:
            panel = json.loads((env.ROOT / "data" / "live" / a.model / "panel.json").read_text())
            q = json.loads((env.ROOT / "data" / "live" / a.model / "predictor.json").read_text())["q"]
            p_crop = {case: {h: v["p_truth"] for h, v in rr.items()} for case, rr in panel["crop_only"].items()}
            out = env.ROOT / "data" / "scripted" / a.model / a.stage
            st = asyncio.run(run_manifest(mpath, "scripted", scripted={"p_crop": p_crop, "q": q}, out_dir=out,
                                          block_concurrency=a.block_concurrency))
        else:
            st = asyncio.run(run_manifest(mpath, "live", max_usd=a.max_usd, concurrency=a.concurrency,
                                          block_concurrency=a.block_concurrency))
        print(json.dumps(st, indent=1))
    elif a.cmd == "analyze":
        from .analyze import analyze
        d = env.ROOT / "data" / "live" / a.model / a.stage
        s = analyze(d)
        print(json.dumps({k: v for k, v in s.items() if k != "cells"}, indent=1, default=float)[:3000])
    elif a.cmd == "spend":
        from .llm import LEDGER
        print(LEDGER.read_text() if LEDGER.exists() else "no live spend yet")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
