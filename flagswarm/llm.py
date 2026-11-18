"""Agent backends behind one interface: the live Responses adapter and the scripted copying agent.

Sharing rule: within ONE replicate namespace, two branches that request the same event (kind, slot/round, agent)
with the same full input (text, image hash, parameters) receive the one sampled answer. Nothing is memoized
across namespaces, so independent replicas and the A/A audit (its own namespace) draw independent samples.
The per-namespace call log is the checkpoint: on resume, logged answers are reloaded instead of re-billed.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import env

PRICES = {  # USD per 1M tokens, standard short-context uncached rates as given in the implementation prompt (2026-11-17)
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-terra": (1.00, 5.00),
}


@dataclass
class Request:
    kind: str                # "report" | "readout" | "probe" | "fullflag"
    m: int                   # 3: country + one-sentence reason; 1: country only
    candidates: list[str]    # ordered as presented
    memory: list[str]        # oldest -> newest received reports (already normalized "Country | reason")
    image_hash: str
    png: bytes
    namespace: str           # replicate namespace; sharing never crosses it
    event: str               # e.g. "slot:123:agent:4" or "readout:5:agent:4" or "probe:<id>:draw:3"

    def text(self) -> str:
        if self.kind == "report":
            return env.report_text(self.candidates, self.memory, self.m)
        return env.readout_text(self.candidates, self.memory, self.m)

    def key(self, params: dict) -> str:
        payload = json.dumps([self.namespace, self.event, self.kind, self.m, self.candidates, self.text(),
                              self.image_hash, params], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Answer:
    country: str | None
    reason: str | None
    valid: bool
    error: str | None = None
    usage: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def memory_entry(self) -> str | None:
        """Upstream normalized_memory_entry: 'Country | reason' (m=3) or 'Country' (m=1)."""
        if not self.valid:
            return None
        return f"{self.country} | {self.reason.strip()}" if self.reason else self.country

    def to_json(self) -> dict:
        return {"country": self.country, "reason": self.reason, "valid": self.valid, "error": self.error,
                "usage": self.usage, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "Answer":
        return cls(d["country"], d["reason"], d["valid"], d.get("error"), d.get("usage", {}), d.get("meta", {}))

    def slim(self) -> "Answer":
        """What later branches need from a cached answer; usage and metadata live in the call log on disk."""
        return Answer(self.country, self.reason, self.valid, self.error)


class BudgetExceeded(RuntimeError):
    pass


class Spend:
    """Running provider cost from reported usage; refuses new calls past max_usd."""

    def __init__(self, model: str, max_usd: float | None):
        self.model, self.max_usd = model, max_usd
        self.usd = 0.0
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0

    def add(self, usage: dict) -> None:
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        it, ot = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        self.calls += 1
        self.input_tokens += it
        self.output_tokens += ot
        self.cached_tokens += usage.get("cached_tokens", 0)
        # Cached input billed at the uncached rate here: an upper bound until cache pricing is confirmed.
        self.usd += (it * pin + ot * pout) / 1e6

    def check(self) -> None:
        if self.max_usd is not None and self.usd >= self.max_usd:
            raise BudgetExceeded(f"spend ${self.usd:.2f} reached cap ${self.max_usd:.2f}")

    def summary(self) -> dict:
        return {"model": self.model, "calls": self.calls, "usd": round(self.usd, 4), "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "cached_tokens": self.cached_tokens}


class CallLog:
    """Append-only JSONL of answered calls for one namespace group; doubles as the resume checkpoint."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, Answer] = {}
        if path.exists():
            with path.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn final line after a crash
                    self.done[rec["key"]] = Answer.from_json(rec["answer"]).slim()
        self._fh = path.open("a")

    def write(self, key: str, req: Request, ans: Answer) -> None:
        rec = {"key": key, "ns": req.namespace, "event": req.event, "kind": req.kind, "m": req.m,
               "image_hash": req.image_hash, "memory_len": len(req.memory), "answer": ans.to_json()}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class Backend:
    """Shared machinery: per-key dedup of in-flight requests, call log, spend accounting."""
    source = "abstract"

    def __init__(self, log: CallLog | None, spend: Spend | None = None):
        self.log = log
        self.spend = spend
        self._inflight: dict[str, asyncio.Future] = {}
        self.done: dict[str, Answer] = log.done if log is not None else {}
        self.n_fresh = 0
        self.n_shared = 0
        self.n_resumed = 0

    def params(self) -> dict:
        raise NotImplementedError

    async def _sample(self, req: Request) -> Answer:
        raise NotImplementedError

    async def __call__(self, req: Request) -> Answer:
        key = req.key(self.params())
        if key in self.done:
            self.n_resumed += 1   # finished earlier: same namespace and input (another branch, or a resumed log)
            return self.done[key]
        if key in self._inflight:
            self.n_shared += 1
            return await self._inflight[key]
        fut = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            ans = await self._sample(req)
        except BaseException as exc:
            fut.set_exception(exc)
            del self._inflight[key]
            raise
        self.n_fresh += 1
        if self.log is not None:
            self.log.write(key, req, ans)
        self.done[key] = ans.slim()   # memory: long games hold hundreds of thousands of cached answers
        del self._inflight[key]
        fut.set_result(ans)
        return ans


# ---------------------------------------------------------------------------------------------
# Live: OpenAI Responses API
# ---------------------------------------------------------------------------------------------

def report_schema(candidates: list[str], m: int) -> dict:
    props: dict[str, Any] = {"country": {"type": "string", "enum": list(candidates)}}
    if m == 3:
        props["reason"] = {"type": "string", "minLength": 1}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def schema_hash(candidates: list[str], m: int) -> str:
    return env.sha(json.dumps(report_schema(candidates, m), sort_keys=True).encode())


def load_openai_key() -> str:
    """OPENAI_API_KEY from the environment, else from /workspace/.env (never printed)."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for line in Path("/workspace/.env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENAI_API_KEY not found")


class OpenAIResponses(Backend):
    source = "live"

    def __init__(self, model: str, log: CallLog | None, spend: Spend, concurrency: int = 64,
                 max_output_tokens: int = 128, detail: str = "high", max_attempts: int = 8,
                 shared_from: "OpenAIResponses | None" = None):
        super().__init__(log, spend)
        import openai
        self.openai = openai
        self.model = model
        if shared_from is not None:   # one client and one concurrency limit across per-block call logs
            self.client, self.sem = shared_from.client, shared_from.sem
        else:
            self.client = openai.AsyncOpenAI(api_key=load_openai_key(), max_retries=0, timeout=120)
            self.sem = asyncio.Semaphore(concurrency)
        self.max_output_tokens = max_output_tokens
        self.detail = detail
        self.max_attempts = max_attempts
        self.failed_attempts = 0
        self.ratelimit_headers: dict = {}

    def params(self) -> dict:
        return {"model": self.model, "reasoning": {"effort": "none"}, "max_output_tokens": self.max_output_tokens,
                "detail": self.detail, "schema": "flag_report/strict"}

    def body(self, req: Request) -> dict:
        data_url = "data:image/png;base64," + base64.b64encode(req.png).decode("ascii")
        return dict(
            model=self.model,
            input=[
                {"role": "system", "content": env.system_text()},
                {"role": "user", "content": [
                    {"type": "input_text", "text": req.text()},
                    {"type": "input_image", "image_url": data_url, "detail": self.detail},
                ]},
            ],
            text={"format": {"type": "json_schema", "name": "flag_report", "strict": True,
                             "schema": report_schema(req.candidates, req.m)}},
            reasoning={"effort": "none"},
            max_output_tokens=self.max_output_tokens,
            store=False,
        )

    async def _sample(self, req: Request) -> Answer:
        transient = (self.openai.RateLimitError, self.openai.APIConnectionError, self.openai.APITimeoutError,
                     self.openai.InternalServerError)
        attempts = 0
        while True:
            self.spend.check()
            attempts += 1
            t0 = time.time()
            try:
                async with self.sem:
                    raw = await self.client.responses.with_raw_response.create(**self.body(req))
                resp = raw.parse()
                self.ratelimit_headers = {k: v for k, v in raw.headers.items() if k.startswith("x-ratelimit")}
                request_id = raw.headers.get("x-request-id")
                break
            except transient as exc:
                self.failed_attempts += 1
                if attempts >= self.max_attempts:
                    return Answer(None, None, False, f"transport:{type(exc).__name__}",
                                  meta={"attempts": attempts, "requested_model": self.model})
                await asyncio.sleep(min(60.0, 2 ** attempts) * (0.5 + random.random()))
            except self.openai.APIStatusError as exc:
                self.failed_attempts += 1
                if exc.status_code >= 500 and attempts < self.max_attempts:
                    await asyncio.sleep(min(60.0, 2 ** attempts) * (0.5 + random.random()))
                    continue
                return Answer(None, None, False, f"http{exc.status_code}:{str(exc)[:300]}",
                              meta={"attempts": attempts, "requested_model": self.model})
        u = resp.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cached_tokens": getattr(u.input_tokens_details, "cached_tokens", 0) or 0,
                 "reasoning_tokens": getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0}
        self.spend.add(usage)
        meta = {"requested_model": self.model, "returned_model": resp.model, "request_id": request_id,
                "response_id": resp.id, "created_at": resp.created_at, "t_request": t0, "latency_s": round(time.time() - t0, 3),
                "attempts": attempts, "status": resp.status, "sdk": self.openai.__version__,
                "image_bytes": len(req.png), "image_hash": req.image_hash,
                "schema_sha256": schema_hash(req.candidates, req.m)}
        return decode(resp, req, usage, meta)


def decode(resp: Any, req: Request, usage: dict, meta: dict) -> Answer:
    """Strict decoding: refusals, truncation, empty reasons, wrong keys and labels are invalid (never coerced)."""
    if resp.status == "incomplete":
        why = getattr(resp.incomplete_details, "reason", "unknown")
        return Answer(None, None, False, f"incomplete:{why}", usage, meta)
    text_parts = []
    for item in resp.output:
        for c in getattr(item, "content", None) or []:
            if c.type == "refusal":
                return Answer(None, None, False, "refusal", usage, meta)
            if c.type == "output_text":
                text_parts.append(c.text)
    raw = "".join(text_parts)
    meta = {**meta, "raw_len": len(raw)}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return Answer(None, None, False, "json", usage, {**meta, "raw": raw[:300]})
    want = {"country", "reason"} if req.m == 3 else {"country"}
    if set(obj) != want:
        return Answer(None, None, False, "keys", usage, {**meta, "raw": raw[:300]})
    country = obj["country"]
    if country not in req.candidates:
        return Answer(None, None, False, "label", usage, {**meta, "raw": raw[:300]})
    reason = obj.get("reason")
    if req.m == 3 and (not isinstance(reason, str) or not reason.strip()):
        return Answer(None, None, False, "empty_reason", usage, meta)
    if reason is not None:
        meta["reason_len"] = len(reason)
    return Answer(country, reason, True, None, usage, meta)


# ---------------------------------------------------------------------------------------------
# Scripted copying agent (harness tests and the "scripted" dataset; no provider call)
# ---------------------------------------------------------------------------------------------

class ScriptedCopying(Backend):
    """Stateless neutral copier rebuilt from local memory, mirroring the live call contract.

    For each received report, newest first, adopt its label with probability q; if none is adopted, answer
    from the crop: truth with probability p_crop[image_hash]. Frozen witnesses (q=0, p in {0,1}) always give
    their role label. Randomness is a hash of the request key, so identical requests in one namespace agree
    (as the shared-sample rule requires) while different namespaces are independent.
    """
    source = "scripted"

    def __init__(self, truth: str, rival: str, p_crop: dict[str, float], q: dict[str, float] | float,
                 log: CallLog | None = None, tag: str = "scripted-v1"):
        super().__init__(log, None)
        self.truth, self.rival = truth, rival
        self.p_crop, self.q, self.tag = p_crop, q, tag

    def params(self) -> dict:
        return {"model": self.tag}

    async def _sample(self, req: Request) -> Answer:
        rng = random.Random(req.key(self.params()))
        q = self.q[req.image_hash] if isinstance(self.q, dict) else self.q
        label = None
        for line in reversed(req.memory):
            if rng.random() < q:
                label = line.split(" | ", 1)[0]
                break
        if label is None:
            label = self.truth if rng.random() < self.p_crop[req.image_hash] else self.rival
        reason = f"scripted report for {label}" if req.m == 3 else None
        return Answer(label, reason, True, None, {"input_tokens": 0, "output_tokens": 0}, {"returned_model": self.tag})


# ---------------------------------------------------------------------------------------------
# Cross-run spend ledger (the $150 session cap holds across separate commands)
# ---------------------------------------------------------------------------------------------

LEDGER = env.ROOT / "data" / "live" / "spend_ledger.json"
SESSION_CAP_USD = 150.0
SAFETY_MARGIN_USD = 10.0


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(r["usd"] for r in json.loads(LEDGER.read_text())["runs"])


def ledger_record(run: str, spend: Spend, note: str = "") -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"cap_usd": SESSION_CAP_USD, "runs": []}
    data["runs"].append({"run": run, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **spend.summary(), "note": note})
    data["total_usd"] = round(sum(r["usd"] for r in data["runs"]), 4)
    LEDGER.write_text(json.dumps(data, indent=1))


def remaining_budget(requested: float | None = None) -> float:
    """What a new run may spend: the smaller of its own cap and what the session cap leaves after the margin."""
    left = SESSION_CAP_USD - SAFETY_MARGIN_USD - ledger_total()
    return max(0.0, left if requested is None else min(requested, left))
