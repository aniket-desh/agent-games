"""Flag Game environment pieces reused from the pinned upstream checkout (vendor/flag_game @ 546badf).

Reused unchanged: renderer (render_flag, image_to_png_bytes), crop enumeration (all_crop_boxes, scale_crop_box,
crop_image), prompt builders (system_prompt, interaction_text, probe_text), pair schedule (generate_pair_schedule),
country pools and canonicalization. This module only adds hashing, crop classes, and provenance records.
"""
from __future__ import annotations

import hashlib
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from nnd.flag_game import prompts as up_prompts
from nnd.flag_game.catalog import COUNTRY_POOLS, canonical_country_name, get_country_lookup, get_flag
from nnd.flag_game.crops import all_crop_boxes, crop_image, scale_crop_box
from nnd.flag_game.render import image_to_png_bytes, render_flag
from nnd.flag_game.runner import generate_pair_schedule

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "flag_game"
UPSTREAM_COMMIT = "546badfdb84440ec455d1a951e207b5357619e2f"

# Authors' geometry: 24 x 16 logical canvas, 6 x 4 crops, 25x rendering -> 150 x 100 px per crop.
CANVAS_W, CANVAS_H, TILE_W, TILE_H, SCALE = 24, 16, 6, 4, 25
DISCOVERY_CASES = [("France", "Peru"), ("Yemen", "Austria"), ("Ireland", "Italy"), ("Netherlands", "Luxembourg")]
DIRECTED_CASES = [c for a, b in DISCOVERY_CASES for c in ((a, b), (b, a))]
TRANSFER_POOL = "stripe_plus_real_triangle_28"
# Label-symmetric, identically worded one-sentence report (individual memory manipulation and clamped witnesses).
TEMPLATE_REASON = "Based on my crop, this is my best guess."


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@lru_cache(maxsize=None)
def full_flag(country: str) -> np.ndarray:
    return render_flag(get_flag(country), width=CANVAS_W * SCALE, height=CANVAS_H * SCALE)


@lru_cache(maxsize=None)
def full_flag_png(country: str) -> bytes:
    return image_to_png_bytes(full_flag(country))


@dataclass(frozen=True)
class CropClass:
    """One distinct crop image of a flag, with every logical position that renders to it."""
    country: str
    hash: str
    png: bytes
    positions: tuple[tuple[int, int], ...]  # (top, left) in logical units

    @property
    def multiplicity(self) -> int:
        return len(self.positions)


@lru_cache(maxsize=None)
def crop_classes(country: str) -> tuple[CropClass, ...]:
    """Enumerate all 6x4 positions on the 24x16 canvas, group identical pixel hashes, keep multiplicities."""
    img = full_flag(country)
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pngs: dict[str, bytes] = {}
    for box in all_crop_boxes(canvas_width=CANVAS_W, canvas_height=CANVAS_H, tile_width=TILE_W, tile_height=TILE_H):
        crop = crop_image(img, scale_crop_box(box, SCALE))
        png = image_to_png_bytes(crop)
        h = sha(png)
        buckets[h].append((box.top, box.left))
        pngs[h] = png
    return tuple(CropClass(country, h, pngs[h], tuple(pos)) for h, pos in sorted(buckets.items(), key=lambda kv: kv[1][0]))


def crop_png_for_position(country: str, top: int, left: int) -> tuple[str, bytes]:
    for c in crop_classes(country):
        if (top, left) in c.positions:
            return c.hash, c.png
    raise KeyError((country, top, left))


def n_positions() -> int:
    return (CANVAS_H - TILE_H + 1) * (CANVAS_W - TILE_W + 1)


def compatibility(hash_: str, candidates: list[str]) -> tuple[str, ...]:
    """Which candidate flags contain this exact crop image somewhere (pixel-identical)."""
    return tuple(c for c in candidates if any(cc.hash == hash_ for cc in crop_classes(c)))


def native_multiplicity(hash_: str, country: str) -> int:
    return next((c.multiplicity for c in crop_classes(country) if c.hash == hash_), 0)


def png_bytes_by_hash(hash_: str, countries: list[str]) -> bytes:
    for country in countries:
        for c in crop_classes(country):
            if c.hash == hash_:
                return c.png
    raise KeyError(hash_)


# ---------------------------------------------------------------------------------------------
# Prompts (upstream-faithful arm): guidance disabled, binary or 28-label candidate list.
# ---------------------------------------------------------------------------------------------

def system_text() -> str:
    return up_prompts.system_prompt()


def report_text(candidates: list[str], memory: list[str], m: int) -> str:
    return up_prompts.interaction_text(countries=candidates, memory_lines=memory, m=m, prompt_social_susceptibility=False)


def readout_text(candidates: list[str], memory: list[str], m: int) -> str:
    return up_prompts.probe_text(countries=candidates, memory_lines=memory, m=m, prompt_social_susceptibility=False)


def canonical(label: str, candidates: list[str]) -> str | None:
    c = canonical_country_name(label)
    return c if c in candidates else None


def transfer_pool() -> list[str]:
    return list(get_country_lookup(TRANSFER_POOL))


def upstream_provenance() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(UPSTREAM), *args], capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "HEAD")
    diff = git("diff", "HEAD")
    return {
        "upstream_repo": "https://github.com/lizardp1/flag_game",
        "upstream_commit": head,
        "upstream_commit_expected": UPSTREAM_COMMIT,
        "upstream_local_diff_sha256": sha(diff.encode()),
        "upstream_local_diff_empty": diff == "",
        "prompt_hashes": {
            "system": sha(system_text().encode()),
            "report_m3_empty": sha(report_text(["A", "B"], [], 3).encode()),
            "report_m1_empty": sha(report_text(["A", "B"], [], 1).encode()),
        },
    }


def source_snapshot() -> dict:
    """Hash of this package's source files, stored with every run so results map to code."""
    files = sorted((ROOT / "flagswarm").glob("*.py"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode()); h.update(f.read_bytes())
    return {"flagswarm_source_sha256": h.hexdigest(), "files": [f.name for f in files]}

