#!/usr/bin/env python3
"""
STELLA runner for the PhosphoAtlas Agent Benchmark.

Wraps STELLA (Jin et al. 2025, arXiv:2507.02004; a smolagents multi-agent system:
manager CodeAgent + dev / critic / tool_creation agents, models via OpenRouter) so it
can be run under the benchmark's prompts and produce the standard contribution files:
atlas.json, run_log.json (with exact token_usage), run.log, prompt.txt, scores/.

Run from the agent-pa-benchmark repo root, in a Python 3.11 env with STELLA's deps:

  # 0. Harness self-test (no STELLA, no smolagents, no API calls)
  python contributions/swechchha_stella/agent_runner.py --dry-run \
      --condition pipeline_guided --mode tools --output /tmp/stella_dry

  # 1. Wiring check (imports STELLA, registers tools, prints models; no LLM calls)
  python contributions/swechchha_stella/agent_runner.py --stella-repo ../STELLA \
      --wiring-check --mode live --output /tmp/stella_wiring

  # 2. Cheap smoke test (real API, few steps)
  python contributions/swechchha_stella/agent_runner.py --stella-repo ../STELLA \
      --condition naive --mode live --max-steps 4 --timeout-min 10 --output /tmp/stella_smoke

  # 3. Real runs (one folder per condition)
  python contributions/swechchha_stella/agent_runner.py --stella-repo ../STELLA \
      --condition naive --mode live --output contributions/swechchha_stella_default_naive

  # 4. Iterative (3 rounds with scorer feedback)
  python contributions/swechchha_stella/agent_runner.py --stella-repo ../STELLA \
      --condition naive --mode live --iterative --output contributions/swechchha_stella_default_naive

Modes:
  live   Harness web tools (bench_web_fetch, fetch_and_parse_db - same as claude_runner)
         plus all of STELLA's own tools. STELLA downloads PSP / SIGNOR / UniProt itself.
  tools  Local DatabaseTools API (list_databases, query_by_kinase, ...) - the tool names
         pipeline_guided.txt refers to. Needs databases/ files.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import urllib.request
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))

GOLD_PATH = BENCH_ROOT / "gold_standard" / "parsed" / "phosphoatlas_gold.json"
PROMPTS = {
    "naive": "agents/prompts/naive.txt",
    "paper_informed": "agents/prompts/paper_informed.txt",
    "pipeline_guided": "agents/prompts/pipeline_guided.txt",
}
MAX_TOOL_CALLS = 5000  # config.yaml -> run.max_tool_calls
# Strings that should never appear in agent-written code / commands (gold / other runs / creds).
LEAK_PATTERNS = ["phosphoatlas_gold", "gold_standard", "contributions/", BENCH_ROOT.name, str(BENCH_ROOT)]
# STELLA defaults (stella_core.py). Used only when OpenRouter does not report per-call cost.
# Check https://openrouter.ai/models before reporting; override with --price-in/--price-out.
FALLBACK_PRICING = {
    "x-ai/grok-4": (3.0, 15.0),
    "google/gemini-2.5-pro": (1.25, 10.0),
    "anthropic/claude-sonnet-4": (3.0, 15.0),
    "anthropic/claude-opus-5.5": (4.0, 20.0),  # OpenRouter list prices, checked 2026-11-24
    "anthropic/claude-sonnet-5.5": (2.0, 10.0),
    "x-ai/grok-4.3": (1.25, 2.5),
}
# NOTE (2026-11-24): STELLA's hard-coded x-ai/grok-4 (deprecated, 404) and anthropic/claude-sonnet-4
# (no OpenRouter endpoints) no longer run; override them with --preset, --backbone or --manager/--critic-model.

# Models hard-coded in STELLA's stella_core.py (published configuration)
STELLA_ORIGINAL_MODELS = {
    "manager_agent": "x-ai/grok-4",
    "dev_agent": "google/gemini-2.5-pro",
    "critic_agent": "anthropic/claude-sonnet-4",
    "tool_creation_agent": "anthropic/claude-sonnet-4",
}
RETIRED_MODELS = {  # why an original model had to be replaced (checked on OpenRouter 2026-11-24)
    "x-ai/grok-4": "deprecated on OpenRouter (HTTP 404); xAI recommends x-ai/grok-4.3",
    "anthropic/claude-sonnet-4": "listed on OpenRouter but no provider endpoints",
}
# Named model configurations; recorded in run_log.json -> model_config
PRESETS = {
    "opus55": {
        "label": "Option A",
        "description": "All four STELLA agents on Claude Opus 5.5 (backbone held fixed)",
        "backbone": "anthropic/claude-opus-5.5",
    },
    "updated_default": {
        "label": "Option B",
        "description": "STELLA's published model mix, with retired models swapped for the current "
                       "model of the same family (grok-4 -> grok-4.3, claude-sonnet-4 -> claude-sonnet-5.5)",
        "manager_model": "x-ai/grok-4.3",
        "critic_model": "anthropic/claude-sonnet-5.5",
    },
}


def expected_models(args) -> dict:
    """Models the flags will produce (used in --dry-run, where STELLA isn't loaded)."""
    m = dict(STELLA_ORIGINAL_MODELS)
    if args.backbone:
        m = {k: args.backbone for k in m}
    if args.manager_model:
        m["manager_agent"] = args.manager_model
    if args.dev_model:
        m["dev_agent"] = args.dev_model
    if args.critic_model:
        m["critic_agent"] = m["tool_creation_agent"] = args.critic_model
    return m


def model_config(args, used: dict) -> dict:
    """What was run vs. STELLA's published configuration, and why anything differs."""
    preset = PRESETS.get(args.preset or "", {})
    subs = []
    for agent, orig in STELLA_ORIGINAL_MODELS.items():
        new = used.get(agent, orig)
        if new != orig:
            subs.append({"agent": agent, "original": orig, "used": new,
                         "reason": RETIRED_MODELS.get(orig, "chosen by experiment design (backbone / override)")})
    return {
        "preset": args.preset or "custom",
        "preset_label": preset.get("label", "custom flags"),
        "preset_description": preset.get("description", "models set with --backbone / --*-model flags"),
        "stella_original_models": STELLA_ORIGINAL_MODELS,
        "models_used": used,
        "substitutions": subs,
        "matches_published_stella": not subs,
        "internal_llm_calls_note": "STELLA's tool-selection call (new_tools/llm.py) and the model inside "
                                   "new_tools/database_tools.py stay on google/gemini-2.5-pro in every preset; "
                                   "see token_usage.by_caller.stella_internal",
        "flags": {"backbone": args.backbone, "manager_model": args.manager_model, "dev_model": args.dev_model,
                  "critic_model": args.critic_model, "max_output_tokens": args.max_output_tokens},
    }
DB_TOOL_NAMES = ["list_databases", "list_kinases", "list_substrates", "get_stats", "search",
                 "query_by_kinase", "query_by_substrate", "query_by_site", "query_all_dbs"]
HARNESS_NAMES = {"bench_web_fetch", "fetch_and_parse_db", "ingest_entries_file", "submit_atlas", *DB_TOOL_NAMES}

# ─────────────────────────────────────────────────────────────────────────────
# Prompt adapter (appended identically to every condition; logged in run_log)
# ─────────────────────────────────────────────────────────────────────────────

LIVE_DOC = """  bench_web_fetch(url) -> dict        fetch a URL (text, truncated to 15,000 chars; gzip decompressed)
  fetch_and_parse_db(url, source_name) -> dict   download a dataset and auto-parse it into the atlas"""
TOOLS_DOC = """  list_databases(), list_kinases(db, offset, limit), list_substrates(db, offset, limit),
  get_stats(db), search(db, keyword, limit), query_by_kinase(db, gene),
  query_by_substrate(db, gene), query_by_site(db, gene, site), query_all_dbs(gene)"""
COMMON_DOC = """  ingest_entries_file(path, source_name) -> dict   add entries from a JSON/CSV/TSV file you or dev_agent wrote
  submit_atlas(additional_entries, strategy_summary) -> dict   submit the final atlas (call once, at the end)"""
HARNESS_NOTE = """
=== HARNESS NOTE (STELLA adapter; identical across conditions) ===
Your output is scored automatically. Besides your usual tools you have these harness tools:
{tool_doc}
{common_doc}
Only entries that reach the harness (fetch_and_parse_db, ingest_entries_file, or
submit_atlas(additional_entries=<python list>)) are scored. The atlas is too large to paste
into a message: build it in Python or in a file and hand it over with these tools.
If the task mentions tools you do not have, use the tools you do have.
After submit_atlas returns status "accepted", finish with final_answer(<short strategy summary>)."""


def build_prompt(condition: str, mode: str, feedback: str = "") -> tuple:
    base = (BENCH_ROOT / PROMPTS[condition]).read_text().strip()
    note = HARNESS_NOTE.format(tool_doc=TOOLS_DOC if mode == "tools" else LIVE_DOC, common_doc=COMMON_DOC)
    return base + feedback + "\n" + note, base, note


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark imports (reuse the benchmark's own web tools, accumulator and scorer)
# ─────────────────────────────────────────────────────────────────────────────

def import_benchmark() -> types.SimpleNamespace:
    try:
        import anthropic  # noqa: F401
    except ImportError:  # claude_runner only needs it for ClaudeAgent, which we don't use
        sys.modules["anthropic"] = types.ModuleType("anthropic")
    from agents.claude_runner import EntryAccumulator, tool_web_fetch, tool_fetch_and_parse
    return types.SimpleNamespace(EntryAccumulator=EntryAccumulator, tool_web_fetch=tool_web_fetch,
                                 tool_fetch_and_parse=tool_fetch_and_parse)


# ─────────────────────────────────────────────────────────────────────────────
# Instrumentation: tracker, actor attribution, tool / agent / LLM / HTTP logging
# ─────────────────────────────────────────────────────────────────────────────

def _short(x, n: int = 300) -> str:
    try:
        s = x if isinstance(x, str) else json.dumps(x, default=str)
    except Exception:
        s = str(x)
    return s if len(s) <= n else s[:n] + f"...(+{len(s) - n} chars)"


def _failure_text(result) -> Optional[str]:
    """Soft failures: STELLA tools return '❌ ...' strings, benchmark tools return {'error': ...}."""
    if isinstance(result, dict) and result.get("error"):
        return str(result["error"])[:500]
    if isinstance(result, str) and result.lstrip().startswith(("❌", "Error", "ERROR")):
        return result.strip()[:500]
    return None


class Tracker:
    def __init__(self, accumulator_cls):
        self.acc = accumulator_cls()
        self.lock = threading.RLock()
        self.t0 = time.time()
        self.submissions: list = []
        self.strategy = ""
        self.calls: Counter = Counter()
        self.events: list = []
        self.failures: list = []
        self.llm_calls: list = []
        self.http: list = []
        self.audit_chunks: list = []
        self.stop_reason: Optional[str] = None
        self.on_budget = None

    def elapsed(self) -> float:
        return round(time.time() - self.t0, 2)

    def record(self, kind: str, name: str, args, result=None, error: Optional[str] = None, dur: float = 0.0):
        with self.lock:
            self.calls[name] += 1
            self.audit_chunks.append(_short(args, 20000))
            ev = {"t": self.elapsed(), "kind": kind, "name": name, "actor": _actor(),
                  "args": _short(args, 300), "dur_s": round(dur, 2)}
            fail = error or _failure_text(result)
            if fail:
                ev["failed"] = True
                self.failures.append({"t": ev["t"], "kind": kind, "name": name, "args": _short(args, 500), "error": fail})
            else:
                ev["result"] = _short(result, 200)
            self.events.append(ev)
            total = sum(self.calls.values())
        if total >= MAX_TOOL_CALLS and self.on_budget:
            self.on_budget()

    def record_llm(self, requested_model: str, resp, dur: float, source: str = "openai_client"):
        usage = getattr(resp, "usage", None)
        pin = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        pout = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        cached = 0
        cost = None
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
            cost = getattr(usage, "cost", None)
            if cost is None:
                cost = (getattr(usage, "model_extra", None) or {}).get("cost")
        actor = _actor()
        with self.lock:
            self.llm_calls.append({
                "call": len(self.llm_calls) + 1, "t": self.elapsed(),
                "caller": "stella_internal" if actor.startswith("tool:") else actor, "actor": actor,
                "model": requested_model, "served_model": str(getattr(resp, "model", "") or ""),
                "input_tokens": pin, "output_tokens": pout, "cache_read_tokens": cached,
                "reported_cost_usd": float(cost) if cost is not None else None,
                "usage_missing": usage is None, "latency_s": round(dur, 2), "source": source,
            })

    def agent_turns(self) -> int:
        return sum(1 for c in self.llm_calls if c["caller"] != "stella_internal")


_CURRENT: dict = {"tracker": None, "B": None, "db": None, "patched": False}
_ACTOR = threading.local()


def _actor() -> str:
    stack = getattr(_ACTOR, "stack", None)
    return stack[-1] if stack else "stella_internal"  # worker threads (parallel tools) land here


@contextmanager
def acting(name: str):
    if not hasattr(_ACTOR, "stack"):
        _ACTOR.stack = []
    _ACTOR.stack.append(name)
    try:
        yield
    finally:
        _ACTOR.stack.pop()


def T() -> Tracker:
    return _CURRENT["tracker"]


def patch_http():
    """Log every HTTP request made through requests / urllib (URL, status or error).
    LLM traffic (openai -> httpx) is not included."""
    def rec(method, url, status=None, error=None, nbytes=None):
        tr = T()
        if tr is None:
            return
        with tr.lock:
            tr.http.append({"method": method, "url": str(url)[:300], "status": status,
                            "error": (str(error)[:200] if error else None), "bytes": nbytes,
                            "actor": _actor(), "t": tr.elapsed()})
    try:
        import requests
        orig_req = requests.sessions.Session.request

        def logged_request(self, method, url, *a, **k):
            try:
                r = orig_req(self, method, url, *a, **k)
                rec(method, url, r.status_code, nbytes=len(r.content) if k.get("stream") is not True else None)
                return r
            except Exception as e:
                rec(method, url, error=e)
                raise
        requests.sessions.Session.request = logged_request
    except ImportError:
        pass
    orig_open = urllib.request.urlopen

    def logged_urlopen(url, *a, **k):
        u = getattr(url, "full_url", url)
        try:
            r = orig_open(url, *a, **k)
            rec("GET", u, getattr(r, "status", None))
            return r
        except Exception as e:
            rec("GET", u, error=e)
            raise
    urllib.request.urlopen = logged_urlopen


def install_smolagents_instrumentation():
    """One interception point each for STELLA tool calls, sub-agent calls and LLM calls."""
    from smolagents import agents as sm_agents
    from smolagents import tools as sm_tools

    orig_tool_call = sm_tools.Tool.__call__

    def tool_call(self, *args, **kwargs):
        name = getattr(self, "name", type(self).__name__)
        if T() is None or name == "final_answer" or name in HARNESS_NAMES:  # harness tools self-log
            return orig_tool_call(self, *args, **kwargs)
        t = time.time()
        with acting(f"tool:{name}"):
            try:
                out = orig_tool_call(self, *args, **kwargs)
            except Exception as e:
                T().record("stella_tool", name, kwargs or args, error=f"{type(e).__name__}: {e}", dur=time.time() - t)
                raise
        T().record("stella_tool", name, kwargs or args, result=out, dur=time.time() - t)
        return out

    sm_tools.Tool.__call__ = tool_call

    orig_agent_call = sm_agents.MultiStepAgent.__call__

    def agent_call(self, task, **kwargs):
        name = getattr(self, "name", None) or type(self).__name__
        if T() is None:
            return orig_agent_call(self, task, **kwargs)
        t = time.time()
        with acting(name):
            try:
                out = orig_agent_call(self, task, **kwargs)
            except Exception as e:
                T().record("managed_agent", f"agent:{name}", task, error=f"{type(e).__name__}: {e}", dur=time.time() - t)
                raise
        T().record("managed_agent", f"agent:{name}", task, result=out, dur=time.time() - t)
        return out

    sm_agents.MultiStepAgent.__call__ = agent_call

    # Fallback: smolagents' own per-call token_usage, used only when the openai-level
    # hook below did not record the call (e.g. openai moved/renamed Completions).
    try:
        from smolagents import models as sm_models
        for cls_name in ("OpenAIServerModel", "OpenAIModel", "LiteLLMModel"):
            cls = getattr(sm_models, cls_name, None)
            if cls is None or getattr(cls.generate, "_pa_wrapped", False):
                continue

            def gen(self, *a, _orig=cls.generate, **k):
                tr = T()
                n0 = len(tr.llm_calls) if tr else 0
                t = time.time()
                msg = _orig(self, *a, **k)
                tr = T()
                if tr is not None and len(tr.llm_calls) == n0:
                    tu = getattr(msg, "token_usage", None)
                    tr.record_llm(str(getattr(self, "model_id", "?")), types.SimpleNamespace(
                        usage=None if tu is None else types.SimpleNamespace(
                            prompt_tokens=getattr(tu, "input_tokens", 0),
                            completion_tokens=getattr(tu, "output_tokens", 0)),
                        model=""), time.time() - t, source="smolagents_token_usage")
                return msg
            gen._pa_wrapped = True
            cls.generate = gen
    except Exception as e:
        print(f"[WARN] could not wrap smolagents models for token fallback: {e}")

    try:
        from openai.resources.chat import Completions
    except Exception as e:
        print(f"[WARN] openai Completions hook unavailable ({e}); using smolagents token_usage fallback.")
        return
    orig_create = Completions.create

    def create(self, *args, **kwargs):
        if "openrouter" in str(getattr(getattr(self, "_client", None), "base_url", "")):
            extra = dict(kwargs.get("extra_body") or {})
            extra.setdefault("usage", {"include": True})  # ask OpenRouter for per-call cost
            kwargs["extra_body"] = extra
        t = time.time()
        resp = orig_create(self, *args, **kwargs)
        if T() is not None:
            try:
                T().record_llm(str(kwargs.get("model", "?")), resp, time.time() - t)
            except Exception:
                pass
        return resp

    Completions.create = create


# ─────────────────────────────────────────────────────────────────────────────
# Harness functions (plain Python; wrapped as smolagents tools for real runs)
# ─────────────────────────────────────────────────────────────────────────────

def _logged(name):
    def deco(fn):
        def wrapper(*a, **k):
            t = time.time()
            try:
                out = fn(*a, **k)
            except Exception as e:
                T().record("harness", name, k or a, error=f"{type(e).__name__}: {e}", dur=time.time() - t)
                raise
            T().record("harness", name, k or a, result=out, dur=time.time() - t)
            return out
        wrapper.__name__ = name
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return deco


def _add_entries(entries, default_source: str) -> tuple:
    """Add entry dicts to the atlas, keeping each entry's own supporting_databases
    (claude_runner labels all direct entries 'agent_direct'; that would hide multi-DB support)."""
    added, skipped = 0, 0
    for e in entries or []:
        if not isinstance(e, dict):
            skipped += 1
            continue
        k, s, p = e.get("kinase_gene", ""), e.get("substrate_gene", ""), e.get("phospho_site", "")
        if not (k and s and p):
            skipped += 1
            continue
        dbs = e.get("supporting_databases") or [e.get("source") or default_source]
        if isinstance(dbs, str):
            dbs = [d.strip() for d in dbs.split(";") if d.strip()] or [default_source]
        for db in dbs:
            T().acc.add(str(k), str(s), str(p), str(e.get("substrate_uniprot", "") or ""),
                        str(e.get("heptameric_peptide", "") or ""), str(db))
        added += 1
    return added, skipped


@_logged("bench_web_fetch")
def h_bench_web_fetch(url: str) -> dict:
    return _CURRENT["B"].tool_web_fetch(url)


@_logged("fetch_and_parse_db")
def h_fetch_and_parse_db(url: str, source_name: Optional[str] = None) -> dict:
    with T().lock:
        return _CURRENT["B"].tool_fetch_and_parse(url, T().acc, source_name or "")


@_logged("ingest_entries_file")
def h_ingest_entries_file(path: str, source_name: Optional[str] = None) -> dict:
    import csv
    p = Path(path).expanduser()
    p = p if p.is_absolute() else Path.cwd() / p
    if not p.exists():
        return {"error": f"file not found: {p}"}
    try:
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                data = data.get("atlas") or data.get("entries") or []
        else:
            delim = "\t" if p.suffix.lower() in (".tsv", ".tab", ".txt") else ","
            with open(p, newline="") as fh:
                data = list(csv.DictReader(fh, delimiter=delim))
    except Exception as e:
        return {"error": f"could not parse {p.name}: {e}"}
    with T().lock:
        added, skipped = _add_entries(data, source_name or p.stem)
        return {"file": str(p), "entries_added": added, "entries_skipped": skipped, "atlas_size": T().acc.size}


@_logged("submit_atlas")
def h_submit_atlas(additional_entries: Optional[list] = None, strategy_summary: Optional[str] = None) -> dict:
    with T().lock:
        added, skipped = _add_entries(additional_entries, "agent_direct")
        if strategy_summary:
            T().strategy = str(strategy_summary)
        T().submissions.append({"additional_entries_added": added, "invalid_dropped": skipped,
                                "atlas_size": T().acc.size, "t": T().elapsed()})
        return {"status": "accepted", "additional_entries_added": added, "invalid_dropped": skipped,
                "entries_received": T().acc.size,
                "message": f"Atlas submitted with {T().acc.size} deduplicated entries."}


def make_db_funcs() -> dict:
    db = _CURRENT["db"]
    return {name: _logged(name)(getattr(db, name)) for name in DB_TOOL_NAMES}


def harness_funcs(mode: str) -> dict:
    funcs = {"ingest_entries_file": h_ingest_entries_file, "submit_atlas": h_submit_atlas}
    if mode == "tools":
        funcs.update(make_db_funcs())
    else:
        funcs.update({"bench_web_fetch": h_bench_web_fetch, "fetch_and_parse_db": h_fetch_and_parse_db})
    return funcs


def to_smolagents_tools(mode: str) -> list:
    from smolagents import Tool, tool

    @tool
    def bench_web_fetch(url: str) -> dict:
        """Benchmark tool: fetch a URL and return its text content, truncated to 15,000 characters (gzip is decompressed). Use it to read documentation, API responses and download pages. Returns a dict with url, content_type, length, content and truncated, or an error.

        Args:
            url: Full http(s) URL to fetch.
        """
        return h_bench_web_fetch(url=url)

    @tool
    def fetch_and_parse_db(url: str, source_name: Optional[str] = None) -> dict:
        """Benchmark tool: download a dataset (TSV, CSV, JSON or gzipped), auto-parse kinase-substrate-phosphosite entries and add them to the harness atlas (deduplicated by kinase|substrate|site). Recognises PhosphoSitePlus-style TSV columns (GENE, SUB_GENE, SUB_MOD_RSD, SUB_ACC_ID, SITE_+/-7_AA) or a JSON list of entry dicts. Returns entries parsed/added and the current atlas size.

        Args:
            url: URL of the dataset to download.
            source_name: Database name recorded in supporting_databases, e.g. 'PhosphoSitePlus', 'SIGNOR', 'UniProt'.
        """
        return h_fetch_and_parse_db(url=url, source_name=source_name)

    @tool
    def ingest_entries_file(path: str, source_name: Optional[str] = None) -> dict:
        """Benchmark tool: add atlas entries from a file in the working directory, e.g. one dev_agent produced. Accepts a JSON list of entry dicts, or a CSV/TSV whose header includes kinase_gene, substrate_gene, phospho_site (optional: heptameric_peptide, substrate_uniprot, supporting_databases separated by ';'). Returns counts added/skipped and the current atlas size.

        Args:
            path: Path to the JSON, CSV or TSV file (relative to the working directory or absolute).
            source_name: Database name used when an entry has no supporting_databases field.
        """
        return h_ingest_entries_file(path=path, source_name=source_name)

    @tool
    def submit_atlas(additional_entries: Optional[list] = None, strategy_summary: Optional[str] = None) -> dict:
        """Benchmark tool: submit the final phosphorylation atlas. Call once, when curation is finished. Entries already gathered with fetch_and_parse_db / ingest_entries_file are included automatically. Pass entries you built yourself as additional_entries: a Python list of dicts with kinase_gene, substrate_gene, phospho_site and optionally heptameric_peptide, substrate_uniprot, supporting_databases (list of str). Returns the number of deduplicated entries accepted.

        Args:
            additional_entries: Optional list of entry dicts to add before submission.
            strategy_summary: Brief description of the curation strategy used.
        """
        return h_submit_atlas(additional_entries=additional_entries, strategy_summary=strategy_summary)

    tools = [ingest_entries_file, submit_atlas]
    if mode == "live":
        return [bench_web_fetch, fetch_and_parse_db] + tools

    # tools mode: wrap DatabaseTools from its own OpenAI-style definitions
    from databases.tools import DatabaseTools
    funcs = make_db_funcs()
    ok_types = {"string", "integer", "number", "boolean", "array", "object", "any"}
    for d in DatabaseTools.get_tool_definitions():
        fn = d.get("function", d)
        if fn["name"] not in funcs:
            continue
        params = fn.get("parameters", {}) or {}
        required = set(params.get("required", []))
        inputs = {}
        for pname, spec in (params.get("properties") or {}).items():
            t = spec.get("type", "string")
            t = t[0] if isinstance(t, list) else t
            inputs[pname] = {"type": t if t in ok_types else "any", "description": spec.get("description", pname)}
            if pname not in required:
                inputs[pname]["nullable"] = True

        def forward(self, **kwargs):
            return funcs[self.name](**{k: v for k, v in kwargs.items() if v is not None})

        cls = type(f"LocalDB_{fn['name']}", (Tool,), {
            "name": fn["name"], "description": "Local benchmark database tool: " + fn.get("description", ""),
            "inputs": inputs, "output_type": "object",
            "skip_forward_signature_validation": True, "forward": forward,
        })
        tools.append(cls())
    return tools


# ─────────────────────────────────────────────────────────────────────────────
# STELLA setup (once per process; iterative rounds share it -> self-evolution persists)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_workdir(args) -> Path:
    """Fresh scratch dir OUTSIDE the repo: keeps the benchmark .env (PSP credentials) and
    gold standard out of STELLA's cwd, and STELLA's ./new_tools writes out of both repos."""
    wd = Path(args.workdir).resolve() if args.workdir else Path(tempfile.mkdtemp(prefix="stella_pa_"))
    (wd / "agent_outputs").mkdir(parents=True, exist_ok=True)
    if args.stella_repo and (Path(args.stella_repo) / "new_tools").is_dir():
        shutil.copytree(Path(args.stella_repo) / "new_tools", wd / "new_tools", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        (wd / "new_tools").mkdir(exist_ok=True)
    return wd


def snapshot_tools(wd: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (wd / "new_tools").glob("*.py")}


def setup_stella(args, workdir: Path) -> dict:
    repo = Path(args.stella_repo)
    env_file = repo / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    except ImportError:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("[ERROR] OPENROUTER_API_KEY not set (export it or put it in STELLA/.env)")

    install_smolagents_instrumentation()
    patch_http()
    _CURRENT["patched"] = True
    os.chdir(workdir)  # STELLA writes ./new_tools and ./agent_outputs relative to cwd
    sys.path.insert(0, str(repo))
    import stella_core as sc  # creates models, connects PubMed MCP via uvx

    if args.backbone:
        for m in (sc.grok_model, sc.gemini_model, sc.claude_model):
            m.model_id = args.backbone
    if args.max_output_tokens:  # cap per-call output (OpenRouter otherwise reserves the model's maximum)
        for m in (sc.grok_model, sc.gemini_model, sc.claude_model):
            if isinstance(getattr(m, "kwargs", None), dict):
                m.kwargs["max_tokens"] = args.max_output_tokens
    if args.manager_model:
        sc.grok_model.model_id = args.manager_model
    if args.dev_model:
        sc.gemini_model.model_id = args.dev_model
    if args.critic_model:
        sc.claude_model.model_id = args.critic_model

    harness = to_smolagents_tools(args.mode)
    existing = {getattr(t, "name", None) for t in sc.manager_tool_management}
    for t in harness:
        if t.name in existing:
            print(f"[WARN] tool name clash with a STELLA tool, skipping harness tool {t.name}")
            continue
        sc.manager_tool_management.append(t)  # initialize_stella() reads this list

    # initialize_stella() loads STELLA's custom prompts only when use_template=True
    use_template = args.prompts == "stella" or args.memory != "off"
    if not sc.initialize_stella(use_template=use_template, use_mem0=(args.memory == "mem0")) or sc.manager_agent is None:
        sys.exit("[ERROR] stella_core.initialize_stella() failed - see output above")
    if args.memory == "off":
        sc.global_memory_manager = None  # keep custom prompts, disable cross-run memory
        sc.use_templates = False

    ingest = next(t for t in harness if t.name == "ingest_entries_file")
    tools = sc.dev_agent.tools  # let dev_agent hand files back to the harness
    if isinstance(tools, dict):
        tools.setdefault(ingest.name, ingest)
    elif isinstance(tools, list) and ingest not in tools:
        tools.append(ingest)

    mgr = sc.manager_agent
    mgr.max_steps = args.max_steps
    import smolagents
    return {"sc": sc, "mgr": mgr, "smolagents_version": getattr(smolagents, "__version__", "?")}


def stella_models(sc) -> dict:
    return {"manager_agent": getattr(sc.grok_model, "model_id", "?"),
            "dev_agent": getattr(sc.gemini_model, "model_id", "?"),
            "critic_agent": getattr(sc.claude_model, "model_id", "?"),
            "tool_creation_agent": getattr(sc.claude_model, "model_id", "?")}


def tool_names(agent) -> list:
    tools = getattr(agent, "tools", {}) or {}
    return sorted(tools.keys() if isinstance(tools, dict) else [getattr(t, "name", str(t)) for t in tools])


# ─────────────────────────────────────────────────────────────────────────────
# Agent execution
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def write(self, s):
        self.a.write(s)
        try:
            self.b.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        self.a.flush()
        self.b.flush()

    def __getattr__(self, item):
        return getattr(self.a, item)


@contextmanager
def tee_to(path: Path):
    fh = open(path, "w", encoding="utf-8")
    old = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = _Tee(old[0], fh), _Tee(old[1], fh)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old
        fh.close()


def summarize_steps(agent, limit: int = 1500) -> tuple:
    """Compact manager trace for run_log + full code text for the leakage audit."""
    out, code_text = [], []
    for s in getattr(getattr(agent, "memory", None), "steps", []) or []:
        d = {"type": type(s).__name__}
        for attr in ("step_number", "model_output", "code_action", "observations", "error", "plan"):
            v = getattr(s, attr, None)
            if v not in (None, ""):
                v = str(v)
                if attr in ("model_output", "code_action"):
                    code_text.append(v)
                d[attr] = v if len(v) <= limit else v[:limit] + f"...(+{len(v) - limit} chars)"
        timing = getattr(s, "timing", None)
        dur = getattr(timing, "duration", None) if timing is not None else getattr(s, "duration", None)
        if dur is not None:
            d["duration_s"] = round(dur, 2)
        tcs = getattr(s, "tool_calls", None)
        if tcs:
            d["tool_calls"] = [getattr(tc, "name", str(tc)) for tc in tcs]
        out.append(d)
    return out, "\n".join(code_text)


def run_stella(args, prompt: str, ctx: dict, log_path: Path) -> dict:
    sc, mgr = ctx["sc"], ctx["mgr"]
    agents = [mgr, sc.dev_agent, sc.critic_agent, sc.tool_creation_agent]

    def stop(reason: str):
        if T().stop_reason is None:
            T().stop_reason = reason
            print(f"\n[RUNNER] stopping agents: {reason}")
        for a in agents:
            try:
                a.interrupt()  # checked by smolagents between steps
            except Exception:
                pass

    T().on_budget = lambda: stop("max_tool_calls")
    timer = threading.Timer(args.timeout_min * 60, lambda: stop("timeout"))
    timer.daemon = True
    final, stop_reason = None, None
    with tee_to(log_path):
        print(f"[SETUP] models={json.dumps(stella_models(sc))} max_steps={args.max_steps} "
              f"timeout={args.timeout_min}m memory={args.memory} prompts={args.prompts}")
        timer.start()
        try:
            with acting("manager_agent"):
                final = mgr.run(prompt)
        except KeyboardInterrupt:  # manual stop still produces atlas/run_log/scores
            stop_reason = "interrupted"
        except Exception as e:  # API errors, interrupts, ...
            stop_reason = f"error: {type(e).__name__}: {e}"
            print("\n[RUNNER ERROR]\n" + traceback.format_exc())
        finally:
            timer.cancel()
    if T().stop_reason:
        stop_reason = T().stop_reason
    elif stop_reason is None:
        stop_reason = "max_steps" if (getattr(mgr, "step_number", 0) or 0) > args.max_steps else "final_answer"
    trace, code_text = summarize_steps(mgr)
    return {"stop_reason": stop_reason, "final_answer": final, "trace": trace, "agent_text": code_text,
            "manager_steps": getattr(mgr, "step_number", None)}


def run_dry(args, prompt: str, funcs: dict, log_path: Path) -> dict:
    """Stub agent: exercises harness tools + submit + scoring without STELLA or an LLM."""
    lines = [f"[DRY RUN] prompt chars={len(prompt)}"]
    entries = []
    if "query_by_kinase" in funcs:
        for k in funcs["list_kinases"]("psp", 0, 5)["kinases"]:
            for e in funcs["query_by_kinase"]("psp", k)["entries"]:
                entries.append({kk: e.get(kk) for kk in ["kinase_gene", "substrate_gene", "phospho_site",
                                                         "heptameric_peptide", "substrate_uniprot"]}
                               | {"supporting_databases": ["PSP"]})
    else:
        lines.append(f"[DRY RUN] fetch(bad url) -> {funcs['bench_web_fetch']('http://127.0.0.1:9/none')}")
        entries = [{"kinase_gene": "CDK1", "substrate_gene": "RB1", "phospho_site": "S807",
                    "supporting_databases": ["PSP"]}]
    Path("agent_outputs/dry_signor.json").write_text(json.dumps(
        [{"kinase_gene": "CDK1", "substrate_gene": "RB1", "phospho_site": "S807", "supporting_databases": ["SIGNOR"]}]))
    lines.append(f"[DRY RUN] ingest -> {funcs['ingest_entries_file']('agent_outputs/dry_signor.json', 'SIGNOR')}")
    lines.append(f"[DRY RUN] submit -> {funcs['submit_atlas'](entries, 'dry-run stub')}")
    log_path.write_text("\n".join(lines) + "\n")
    return {"stop_reason": "dry_run", "final_answer": None, "trace": [], "agent_text": "", "manager_steps": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring + logging
# ─────────────────────────────────────────────────────────────────────────────

def score(entries: list, scores_dir: Path) -> dict:
    from evaluation.scorer import load_gold, score_atlas, score_per_kinase
    scores_dir.mkdir(parents=True, exist_ok=True)
    gold = load_gold(str(GOLD_PATH))
    s = score_atlas(entries, gold)
    cl = s["column_level"]
    summary = dict(s)
    summary["column_level"] = {k: v for k, v in cl.items() if k != "peptide_mismatches"}
    (scores_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (scores_dir / "per_kinase.json").write_text(json.dumps(score_per_kinase(entries, gold), indent=2))
    (scores_dir / "peptide_mismatches.json").write_text(json.dumps(cl.get("peptide_mismatches", []), indent=2))
    return summary


def token_summary(tracker: Tracker, args) -> dict:
    by_caller, by_model = {}, {}
    total_cost, n_rep, n_est, n_none = 0.0, 0, 0, 0
    for c in tracker.llm_calls:
        if c["reported_cost_usd"] is not None:
            cost = c["reported_cost_usd"]
            n_rep += 1
        else:
            pi, po = FALLBACK_PRICING.get(c["model"], (None, None))
            pi = args.price_in if args.price_in is not None else pi
            po = args.price_out if args.price_out is not None else po
            if pi is None or po is None:
                cost, n_none = 0.0, n_none + 1
            else:
                cost, n_est = c["input_tokens"] * pi / 1e6 + c["output_tokens"] * po / 1e6, n_est + 1
        total_cost += cost
        for key, table in ((c["caller"], by_caller), (c["model"], by_model)):
            b = table.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            b["calls"] += 1
            b["input_tokens"] += c["input_tokens"]
            b["output_tokens"] += c["output_tokens"]
            b["cost_usd"] = round(b["cost_usd"] + cost, 6)
    tin = sum(c["input_tokens"] for c in tracker.llm_calls)
    tout = sum(c["output_tokens"] for c in tracker.llm_calls)
    source = ("none" if not tracker.llm_calls else "openrouter_reported" if n_rep and not (n_est or n_none)
              else "pricing_table" if n_est and not (n_rep or n_none) else "mixed")
    return {
        "total_input_tokens": tin,
        "total_output_tokens": tout,
        "total_tokens": tin + tout,
        "cache_read_input_tokens": sum(c["cache_read_tokens"] for c in tracker.llm_calls),
        "api_calls": len(tracker.llm_calls),
        "estimated_cost_usd": round(total_cost, 4),
        "cost_source": source,
        "calls_with_reported_cost": n_rep, "calls_priced_from_table": n_est, "calls_unpriced": n_none,
        "calls_missing_usage": sum(1 for c in tracker.llm_calls if c["usage_missing"]),
        "by_caller": by_caller,
        "by_model": by_model,
        "per_call": tracker.llm_calls,
        "note": "Exact counts from every OpenAI-compatible chat.completions call (all STELLA agents + "
                "STELLA-internal LLM calls, e.g. tool selection and database_tools).",
    }


def leakage_audit(text: str, mode: str) -> dict:
    hits = sorted({p for p in LEAK_PATTERNS if p and p in text})
    db_hits = mode == "live" and str(BENCH_ROOT / "databases") in text
    return {"patterns_checked": LEAK_PATTERNS, "hits": hits, "local_db_access_in_live_mode": db_hits,
            "clean": not hits and not db_hits}


def git_rev(path, *git_args) -> str:
    if not path:
        return "?"
    try:
        return subprocess.check_output(["git", "-C", str(path), *(git_args or ("rev-parse", "--short", "HEAD"))],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "?"


def run_one(args, ctx: dict, out: Path, feedback: str = "", round_num: int = 1) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "atlas.json").unlink(missing_ok=True)
    workdir = ctx["workdir"]
    tracker = Tracker(_CURRENT["B"].EntryAccumulator)
    _CURRENT["tracker"] = tracker
    prompt, base, note = build_prompt(args.condition, args.mode, feedback)
    (out / "prompt.txt").write_text(prompt)  # exact task sent to STELLA's manager
    print(f"[SETUP] condition={args.condition} mode={args.mode} round={round_num}")
    print(f"[SETUP] output={out}  workdir={workdir}")

    tools_before = snapshot_tools(workdir)
    t_start = time.time()
    os.chdir(workdir)
    if args.dry_run:
        res = run_dry(args, prompt, harness_funcs(args.mode), out / "run.log")
    else:
        res = run_stella(args, prompt, ctx, out / "run.log")
    os.chdir(BENCH_ROOT)
    elapsed = time.time() - t_start

    # Atlas = everything that reached the harness accumulator (+ a list returned as final_answer)
    if tracker.submissions:
        atlas_source = "submit_atlas"
    else:
        if isinstance(res["final_answer"], list):
            with tracker.lock:
                _add_entries(res["final_answer"], "final_answer")
        atlas_source = "accumulated_without_submit" if tracker.acc.size else "none"
    entries = tracker.acc.finalize()
    (out / "atlas.json").write_text(json.dumps(entries, indent=2))
    summary = score(entries, out / "scores")

    # Self-evolution artifacts + files STELLA produced
    after = snapshot_tools(workdir)
    evolved = sorted(n for n, h in after.items() if tools_before.get(n) != h)
    if evolved:
        (out / "self_evolved_tools").mkdir(exist_ok=True)
        for n in evolved:
            shutil.copy2(workdir / "new_tools" / n, out / "self_evolved_tools" / n)
    produced = [{"file": str(p.relative_to(workdir)), "bytes": p.stat().st_size}
                for p in sorted(workdir.rglob("*")) if p.is_file() and p.stat().st_mtime >= t_start
                and "new_tools" not in p.parts and "__pycache__" not in p.parts]
    script_text = "\n".join(p.read_text(errors="replace")[:200000] for p in workdir.rglob("*")
                            if p.is_file() and p.suffix in (".py", ".sh", ".R") and p.stat().st_mtime >= t_start)

    db_counts = Counter(db for e in entries for db in (e.get("supporting_databases") or []))
    hosts = Counter(h["url"].split("/")[2] if "://" in h["url"] else h["url"] for h in tracker.http)
    http_fail = [h for h in tracker.http if h["error"] or (h["status"] and h["status"] >= 400)]
    sc = ctx.get("sc")
    run_log = {
        "agent": ("STELLA (" + ", ".join(f"{k}={v}" for k, v in stella_models(sc).items()) + ")") if sc else "STELLA (dry run)",
        "framework": "STELLA (smolagents: manager CodeAgent + dev/critic/tool_creation agents)",
        "stella_commit": git_rev(args.stella_repo),
        "stella_remote": git_rev(args.stella_repo, "remote", "get-url", "origin"),
        "benchmark_commit": git_rev(BENCH_ROOT),
        "smolagents_version": ctx.get("smolagents_version", "n/a"),
        "python": platform.python_version(),
        "model": args.backbone or (stella_models(sc)["manager_agent"] if sc else "n/a"),
        "models": stella_models(sc) if sc else {},
        "model_config": model_config(args, stella_models(sc) if sc else expected_models(args)),
        "condition": args.condition if round_num == 1 else f"{args.condition}_round{round_num}",
        "mode": args.mode,
        "round": round_num,
        "prompt_file": PROMPTS[args.condition],
        "prompt_chars": len(prompt),
        "harness_note": note,
        "feedback_included": bool(feedback),
        "stella_config": {"memory": args.memory, "prompts": args.prompts, "max_steps": args.max_steps,
                          "max_output_tokens": args.max_output_tokens},
        "budget": {"timeout_min": args.timeout_min, "max_steps": args.max_steps, "max_tool_calls": MAX_TOOL_CALLS},
        "stop_reason": res["stop_reason"],
        "strategy": tracker.strategy,
        "final_answer": _short(res["final_answer"], 4000) if res["final_answer"] is not None else None,
        "atlas_source": atlas_source,
        "submissions": tracker.submissions,
        "databases_accessed": sorted(db_counts),
        "raw_counts": dict(db_counts),
        "merged_atlas": len(entries),
        "unique_kinases": len({e["kinase_gene"] for e in entries}),
        "unique_substrates": len({e["substrate_gene"] for e in entries}),
        "multi_db_entries": sum(1 for e in entries if len(e.get("supporting_databases") or []) >= 2),
        "tool_calls": sum(tracker.calls.values()),
        "agent_turns": tracker.agent_turns(),
        "manager_steps": res["manager_steps"],
        "function_calls": dict(tracker.calls.most_common()),
        "failures": tracker.failures,
        "http_hosts": dict(hosts),
        "http_failures": http_fail,
        "http_requests": tracker.http,
        "self_evolved_tools": evolved,
        "agent_output_files": produced,
        "elapsed_seconds": round(elapsed, 1),
        "token_usage": token_summary(tracker, args),
        "leakage_audit": leakage_audit("\n".join(tracker.audit_chunks) + res["agent_text"] + script_text, args.mode),
        "tool_events": tracker.events,
        "trace": res["trace"],
    }
    (out / "run_log.json").write_text(json.dumps(run_log, indent=2, default=str))
    me = Path(__file__).resolve()
    if round_num == 1 and me.parent != out.resolve():
        shutil.copy2(me, out / "agent_runner.py")  # provenance
    ov, tu = summary["overview"], run_log["token_usage"]
    print(f"[DONE] stop={res['stop_reason']} atlas={len(entries)} ({atlas_source}) "
          f"F1={ov['f1']} P={ov['precision']} R={ov['recall']} kinases={ov['kinases_found']} "
          f"failures={len(tracker.failures)} http_failures={len(http_fail)}")
    print(f"[COST] {tu['total_tokens']:,} tokens over {tu['api_calls']} LLM calls ≈ ${tu['estimated_cost_usd']} "
          f"({tu['cost_source']})")
    if not run_log["leakage_audit"]["clean"]:
        print(f"[WARN] leakage audit hits: {run_log['leakage_audit']}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run STELLA on the PhosphoAtlas benchmark")
    ap.add_argument("--condition", choices=list(PROMPTS), default="naive")
    ap.add_argument("--mode", choices=["live", "tools"], default="live")
    ap.add_argument("--output", required=True)
    ap.add_argument("--stella-repo", default=os.getenv("STELLA_REPO"), help="Path to STELLA clone (loads its .env)")
    ap.add_argument("--preset", choices=list(PRESETS), default=None,
                    help="opus55 = Option A (all agents Opus 5.5); updated_default = Option B (grok-4.3 + "
                         "gemini-2.5-pro + sonnet-5.5). Explicit --backbone/--*-model flags still win.")
    ap.add_argument("--backbone", default=None, help="One OpenRouter model id for ALL STELLA agents (holds LLM fixed)")
    ap.add_argument("--manager-model", default=None, help="Override manager_agent model (STELLA default x-ai/grok-4)")
    ap.add_argument("--dev-model", default=None, help="Override dev_agent model (default google/gemini-2.5-pro)")
    ap.add_argument("--critic-model", default=None, help="Override critic + tool_creation model (default anthropic/claude-sonnet-4)")
    ap.add_argument("--max-output-tokens", type=int, default=None,
                    help="Cap max_tokens per LLM call (e.g. 8192); logged in run_log.stella_config")
    ap.add_argument("--max-steps", type=int, default=30, help="manager_agent max steps (smolagents default: 20)")
    ap.add_argument("--timeout-min", type=float, default=60, help="Wall-clock budget per round (config.yaml: 60)")
    ap.add_argument("--memory", choices=["off", "kb", "mem0"], default="off",
                    help="STELLA cross-run memory; 'off' keeps runs independent (kb/mem0 need path patches)")
    ap.add_argument("--prompts", choices=["stella", "default"], default="stella",
                    help="STELLA's prompts/Stella_prompt_modified.yaml or smolagents defaults")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens if OpenRouter reports no cost")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens if OpenRouter reports no cost")
    ap.add_argument("--iterative", action="store_true", help="3 rounds with scorer feedback")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workdir", default=None, help="Agent scratch dir (default: fresh temp dir outside repo)")
    ap.add_argument("--dry-run", action="store_true", help="Stub agent; tests harness without STELLA/API")
    ap.add_argument("--wiring-check", action="store_true", help="Import + configure STELLA, print tools; no LLM calls")
    args = ap.parse_args()

    if args.preset:  # fill model flags from the preset unless set explicitly
        for key in ("backbone", "manager_model", "dev_model", "critic_model"):
            if getattr(args, key) is None and key in PRESETS[args.preset]:
                setattr(args, key, PRESETS[args.preset][key])
        print(f"[SETUP] preset={args.preset} ({PRESETS[args.preset]['label']}): {PRESETS[args.preset]['description']}")
    if args.stella_repo:
        args.stella_repo = Path(args.stella_repo).expanduser().resolve()
    if not args.dry_run:
        if not args.stella_repo or not (args.stella_repo / "stella_core.py").exists():
            ap.error("--stella-repo must point to a STELLA clone (containing stella_core.py)")
    if args.condition == "pipeline_guided" and args.mode == "live":
        print("[NOTE] pipeline_guided names DatabaseTools tools (list_databases, query_by_kinase, ...). "
              "In live mode they don't exist; use --mode tools and a *_tools folder for that comparison.")
    out = Path(args.output).resolve()

    _CURRENT["B"] = import_benchmark()
    if args.mode == "tools":
        from databases.tools import DatabaseTools
        _CURRENT["db"] = DatabaseTools(str(BENCH_ROOT / "databases"))
    if args.dry_run:
        patch_http()
    workdir = prepare_workdir(args)
    ctx = {"workdir": workdir}
    if not args.dry_run:
        ctx.update(setup_stella(args, workdir))
        os.chdir(BENCH_ROOT)
    if args.wiring_check:
        sc, mgr = ctx["sc"], ctx["mgr"]
        print("\n[WIRING] models:", json.dumps(stella_models(sc), indent=2))
        print("[WIRING] manager tools:", tool_names(mgr))
        print("[WIRING] managed agents:", sorted(getattr(mgr, "managed_agents", {}) or {}))
        print("[WIRING] dev_agent tools:", tool_names(sc.dev_agent))
        print("[WIRING] smolagents:", ctx["smolagents_version"], "| python:", platform.python_version())
        print("[WIRING] gold:", "found" if GOLD_PATH.exists() else f"MISSING ({GOLD_PATH})", "| workdir:", workdir)
        return

    if not args.iterative:
        run_one(args, ctx, out)
        return

    # Iterative: same process + workdir, so tools STELLA created in round 1 stay available.
    from experiments.iterative_refinement import generate_feedback
    rows, blocks, feedback = [], [], ""
    for r in range(1, args.rounds + 1):
        rdir = out if r == 1 else out / f"round{r}"
        if r > 1:
            prev = (out if r == 2 else out / f"round{r-1}") / "scores" / "summary.json"
            blocks.append(generate_feedback(prev, r - 1))
            feedback = (f"\n\nYou are on Round {r} of an iterative curation process. "
                        f"Below is feedback from your previous round(s). Use it to improve.\n" + "".join(blocks))
        s = run_one(args, ctx, rdir, feedback, r)
        rows.append({"round": r, **s["overview"], **{k: s["atlas_level"][k] for k in
                     ["total_agent_entries", "total_gold_entries", "true_positives", "false_positives", "false_negatives"]}})
    (out / "iterative_comparison.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
