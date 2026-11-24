#!/usr/bin/env python3
"""
Claude Fable 5.2 runner for the PhosphoAtlas Agent Benchmark (direct Anthropic Messages API).

Built on the same harness as contributions/swechchha_stella/agent_runner.py, minus STELLA: one
Claude model (claude-fable-5-2) in a tool-use loop, called straight through the Anthropic API.
Produces the standard contribution files: atlas.json, run_log.json (exact token_usage from every
API response), run.log, prompt.txt, scores/.

Run from the agent-pa-benchmark repo root (Python >= 3.10, `pip install anthropic`):

  # 0. Harness self-test (no API key, no LLM calls)
  python contributions/swechchha_fable52/agent_runner.py --dry-run \
      --condition pipeline_guided --mode tools --output /tmp/fable_dry

  # 1. Wiring check (checks key + model id with the free count_tokens endpoint; no generation)
  export ANTHROPIC_API_KEY=sk-ant-...
  python contributions/swechchha_fable52/agent_runner.py --wiring-check \
      --condition pipeline_guided --mode tools --output /tmp/fable_wiring

  # 2. Cheap smoke test (real API, few turns)
  python contributions/swechchha_fable52/agent_runner.py --condition naive --mode live \
      --max-turns 4 --timeout-min 10 --output /tmp/fable_smoke

  # 3. Real runs (one folder per condition)
  python contributions/swechchha_fable52/agent_runner.py --condition naive --mode live \
      --output contributions/swechchha_fable52_naive
  python contributions/swechchha_fable52/agent_runner.py --condition pipeline_guided --mode tools \
      --output contributions/swechchha_fable52_pipeline_guided_tools

  # 4. Iterative (3 rounds with scorer feedback)
  python contributions/swechchha_fable52/agent_runner.py --condition naive --mode live --iterative \
      --output contributions/swechchha_fable52_naive_iterative

  # Any other prompt
  python contributions/swechchha_fable52/agent_runner.py --prompt-file my_prompt.txt --mode tools \
      --output /tmp/fable_custom

Modes:
  live   claude_runner's web tools (web_search, web_fetch, fetch_and_parse_db). Claude downloads
         PSP / SIGNOR / UniProt itself.
  tools  Local DatabaseTools API (list_databases, query_by_kinase, ...) - the tool names
         pipeline_guided.txt refers to. Needs databases/ files.

run_python (on by default, --no-code-tool to disable):
  STELLA's manager is a CodeAgent, so it can loop over 400+ kinases inside one step and pass a ~16k
  entry list to submit_atlas as a Python object. A plain tool-use loop cannot: every entry would
  have to be written out as JSON tokens. run_python gives Claude the same ability - a persistent
  Python namespace in a scratch dir outside the repo, with every harness tool available as a
  function. Calls made from code are logged like direct tool calls. open() on paths inside the
  benchmark repo is refused, and all code is included in the leakage audit.
"""
import argparse
import ast
import io
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import urllib.request
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
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
DEFAULT_MODEL = "claude-fable-5-2"
MAX_TOOL_CALLS = 5000  # config.yaml -> run.max_tool_calls
LEAK_PATTERNS = ["phosphoatlas_gold", "gold_standard", "contributions/", BENCH_ROOT.name, str(BENCH_ROOT)]
# USD per 1M tokens: (input, output, cache_read, cache_write_5m).
# claude-fable-5-2 input/output = $8 / $40 per third-party price trackers (checked 2026-11-24);
# cache rates assume Anthropic's usual 0.1x read / 1.25x write. Confirm on
# https://docs.claude.com/en/docs/about-claude/pricing before reporting; override with --price-*.
PRICING = {
    "claude-fable-5-2": (8.0, 40.0, 0.80, 10.0),
    "claude-opus-5-5": (5.0, 25.0, 0.50, 6.25),
    "claude-sonnet-5-5": (3.0, 15.0, 0.30, 3.75),
}

DB_TOOL_NAMES = ["list_databases", "list_kinases", "list_substrates", "get_stats", "search",
                 "query_by_kinase", "query_by_substrate", "query_by_site", "query_all_dbs"]
LIVE_TOOL_NAMES = ["web_search", "web_fetch", "fetch_and_parse_db"]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt adapter (appended identically to every condition; logged in run_log)
# ─────────────────────────────────────────────────────────────────────────────

LIVE_DOC = """  web_search(query) -> dict          search the web (titles + URLs)
  web_fetch(url) -> dict             fetch a URL (text, truncated to 15,000 chars; gzip decompressed)
  fetch_and_parse_db(url, source_name) -> dict   download a dataset and auto-parse it into the atlas"""
TOOLS_DOC = """  list_databases(), list_kinases(db, offset, limit), list_substrates(db, offset, limit),
  get_stats(db), search(db, keyword, limit), query_by_kinase(db, gene),
  query_by_substrate(db, gene), query_by_site(db, gene, site), query_all_dbs(gene)"""
COMMON_DOC = """  ingest_entries_file(path, source_name) -> dict   add entries from a JSON/CSV/TSV file you wrote
  submit_atlas(additional_entries, strategy_summary) -> dict   submit the final atlas (call once, at the end)"""
CODE_DOC = """  run_python(code) -> stdout        run Python in a persistent namespace (variables survive between
                                     calls). Every tool above is callable there as a plain function
                                     returning a dict, e.g. query_by_kinase(db="psp", gene="CDK1"),
                                     and submit_atlas(additional_entries=my_list, strategy_summary="...")."""
HARNESS_NOTE = """
=== HARNESS NOTE (Claude API adapter; identical across conditions) ===
Your output is scored automatically. You have these tools:
{tool_doc}
{common_doc}{code_doc}
Only entries that reach the harness (fetch_and_parse_db, ingest_entries_file, or
submit_atlas(additional_entries=...)) are scored. {size_hint}
If the task mentions tools you do not have, use the tools you do have.
When submit_atlas returns status "accepted", reply with a short strategy summary and stop."""


def build_prompt(args, feedback: str = "") -> tuple:
    src = Path(args.prompt_file) if args.prompt_file else BENCH_ROOT / PROMPTS[args.condition]
    base = src.read_text().strip()
    note = HARNESS_NOTE.format(
        tool_doc=TOOLS_DOC if args.mode == "tools" else LIVE_DOC, common_doc=COMMON_DOC,
        code_doc=("\n" + CODE_DOC) if args.code_tool else "",
        size_hint=("The atlas is too large to write out in a message: build it in run_python and pass the "
                   "Python list to submit_atlas there." if args.code_tool else
                   "The atlas is too large to write out in one message: prefer fetch_and_parse_db / "
                   "ingest_entries_file for bulk data."))
    return base + feedback + "\n" + note, base, note, str(src.relative_to(BENCH_ROOT) if src.is_relative_to(BENCH_ROOT) else src)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark imports (reuse the benchmark's own web tools, accumulator and scorer)
# ─────────────────────────────────────────────────────────────────────────────

def import_benchmark() -> types.SimpleNamespace:
    try:
        import anthropic  # noqa: F401
    except ImportError:  # only the dry run gets this far without it
        sys.modules["anthropic"] = types.ModuleType("anthropic")
    from agents.claude_runner import (EntryAccumulator, get_tool_definitions, tool_fetch_and_parse,
                                      tool_web_fetch, tool_web_search)
    return types.SimpleNamespace(EntryAccumulator=EntryAccumulator, tool_web_search=tool_web_search,
                                 tool_web_fetch=tool_web_fetch, tool_fetch_and_parse=tool_fetch_and_parse,
                                 claude_tool_defs=get_tool_definitions)


# ─────────────────────────────────────────────────────────────────────────────
# Instrumentation: tracker, tool / LLM / HTTP logging
# ─────────────────────────────────────────────────────────────────────────────

def _short(x, n: int = 300) -> str:
    try:
        s = x if isinstance(x, str) else json.dumps(x, default=str)
    except Exception:
        s = str(x)
    return s if len(s) <= n else s[:n] + f"...(+{len(s) - n} chars)"


def _failure_text(result) -> Optional[str]:
    if isinstance(result, dict) and result.get("error"):
        return str(result["error"])[:500]
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
        self.code_cells: list = []
        self.stop_reason: Optional[str] = None
        self.max_tool_calls = 0
        self.via = "direct"  # "direct" or "run_python" (who called the tool)

    def elapsed(self) -> float:
        return round(time.time() - self.t0, 2)

    def record(self, name: str, args, result=None, error: Optional[str] = None, dur: float = 0.0):
        with self.lock:
            self.calls[name] += 1
            self.audit_chunks.append(_short(args, 20000))
            ev = {"t": self.elapsed(), "name": name, "via": self.via, "args": _short(args, 300),
                  "dur_s": round(dur, 2)}
            fail = error or _failure_text(result)
            if fail:
                ev["failed"] = True
                self.failures.append({"t": ev["t"], "name": name, "via": self.via,
                                      "args": _short(args, 500), "error": fail})
            else:
                ev["result"] = _short(result, 200)
            self.events.append(ev)
            total = sum(self.calls.values())
        if self.max_tool_calls and total >= self.max_tool_calls and self.stop_reason is None:
            self.stop_reason = "max_tool_calls"
            print(f"\n[RUNNER] stopping: max_tool_calls ({self.max_tool_calls}) reached")

    def record_llm(self, model: str, resp, dur: float, turn: int):
        u = getattr(resp, "usage", None)
        g = (lambda k: int(getattr(u, k, 0) or 0)) if u is not None else (lambda k: 0)
        with self.lock:
            self.llm_calls.append({
                "call": len(self.llm_calls) + 1, "turn": turn, "t": self.elapsed(), "model": model,
                "served_model": str(getattr(resp, "model", "") or ""),
                "input_tokens": g("input_tokens"), "output_tokens": g("output_tokens"),
                "cache_read_tokens": g("cache_read_input_tokens"),
                "cache_write_tokens": g("cache_creation_input_tokens"),
                "stop_reason": getattr(resp, "stop_reason", None),
                "usage_missing": u is None, "latency_s": round(dur, 2),
            })


_CURRENT: dict = {"tracker": None, "B": None, "db": None}


def T() -> Tracker:
    return _CURRENT["tracker"]


def patch_http():
    """Log every HTTP request made through requests / urllib (URL, status or error).
    Anthropic API traffic (httpx) is not included."""
    def rec(method, url, status=None, error=None, nbytes=None):
        tr = T()
        if tr is None:
            return
        with tr.lock:
            tr.http.append({"method": method, "url": str(url)[:300], "status": status,
                            "error": (str(error)[:200] if error else None), "bytes": nbytes,
                            "via": tr.via, "t": tr.elapsed()})
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


# ─────────────────────────────────────────────────────────────────────────────
# Harness functions (called by Claude directly, or from run_python)
# ─────────────────────────────────────────────────────────────────────────────

def _logged(name):
    def deco(fn):
        def wrapper(*a, **k):
            if T().stop_reason:
                return {"error": f"run is stopping ({T().stop_reason}); call submit_atlas now"} \
                    if name != "submit_atlas" else fn(*a, **k)
            t = time.time()
            try:
                out = fn(*a, **k)
            except Exception as e:
                T().record(name, k or a, error=f"{type(e).__name__}: {e}", dur=time.time() - t)
                raise
            T().record(name, k or a, result=out, dur=time.time() - t)
            return out
        wrapper.__name__ = name
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return deco


def _add_entries(entries, default_source: str) -> tuple:
    """Add entry dicts, keeping each entry's own supporting_databases
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


@_logged("web_search")
def h_web_search(query: str, max_results: int = 10) -> dict:
    return _CURRENT["B"].tool_web_search(query, max_results)


@_logged("web_fetch")
def h_web_fetch(url: str) -> dict:
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
                                "atlas_size": T().acc.size, "via": T().via, "t": T().elapsed()})
        return {"status": "accepted", "additional_entries_added": added, "invalid_dropped": skipped,
                "entries_received": T().acc.size,
                "message": f"Atlas submitted with {T().acc.size} deduplicated entries."}


def harness_funcs(mode: str) -> dict:
    funcs = {"ingest_entries_file": h_ingest_entries_file, "submit_atlas": h_submit_atlas}
    if mode == "tools":
        db = _CURRENT["db"]
        funcs.update({n: _logged(n)(getattr(db, n)) for n in DB_TOOL_NAMES})
    else:
        funcs.update({"web_search": h_web_search, "web_fetch": h_web_fetch,
                      "fetch_and_parse_db": h_fetch_and_parse_db})
    return funcs


# ─────────────────────────────────────────────────────────────────────────────
# run_python: persistent Python namespace (STELLA-CodeAgent parity)
# ─────────────────────────────────────────────────────────────────────────────

class PySandbox:
    """Not a security sandbox: runs in-process in a scratch dir outside the repo. open() on the
    benchmark repo is refused; everything else is caught by the leakage audit."""

    def __init__(self, funcs: dict, timeout_s: float, max_out: int):
        self.timeout_s, self.max_out = timeout_s, max_out
        real_open = open

        def guarded_open(file, *a, **k):
            try:
                rp = Path(file).expanduser().resolve()
            except Exception:
                return real_open(file, *a, **k)
            if rp.is_relative_to(BENCH_ROOT):
                raise PermissionError(f"access to {rp} is not allowed in this benchmark")
            return real_open(file, *a, **k)
        self.ns = {"__name__": "__agent__", "open": guarded_open, **funcs}

    def run(self, code: str, budget_s: float) -> str:
        buf = io.StringIO()
        limit = max(1.0, min(self.timeout_s, budget_s))

        def on_alarm(signum, frame):
            raise TimeoutError(f"run_python cell exceeded {limit:.0f}s")
        use_alarm = threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM")
        if use_alarm:
            old = signal.signal(signal.SIGALRM, on_alarm)
            signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                tree = ast.parse(code, "<run_python>", "exec")
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, "<run_python>", "exec"), self.ns)
                if last is not None:  # echo the last expression, like a notebook cell
                    val = eval(compile(ast.Expression(last.value), "<run_python>", "eval"), self.ns)
                    if val is not None:
                        print(repr(val))
        except BaseException as e:  # incl. TimeoutError / SystemExit from agent code
            if isinstance(e, KeyboardInterrupt):
                raise
            buf.write("\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)[-6:]))
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)
        out = buf.getvalue() or "(no output)"
        return out if len(out) <= self.max_out else out[:self.max_out] + f"\n...(output truncated, {len(out)} chars)"


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic tool definitions
# ─────────────────────────────────────────────────────────────────────────────

INGEST_DEF = {
    "name": "ingest_entries_file",
    "description": "Add atlas entries from a file in the working directory. Accepts a JSON list of entry dicts, or a "
                   "CSV/TSV whose header includes kinase_gene, substrate_gene, phospho_site (optional: heptameric_peptide, "
                   "substrate_uniprot, supporting_databases separated by ';'). Returns counts added/skipped and atlas size.",
    "input_schema": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the JSON/CSV/TSV file (relative to the working directory or absolute)"},
        "source_name": {"type": "string", "description": "Database name used when an entry has no supporting_databases"}},
        "required": ["path"]},
}
RUN_PYTHON_DEF = {
    "name": "run_python",
    "description": "Execute Python code in a persistent namespace (variables, imports and functions persist across calls). "
                   "Every other tool is available inside as a plain Python function returning a dict. print() output and "
                   "the value of the last expression are returned (truncated). Use it to loop over many queries, build the "
                   "atlas as a list of dicts, and call submit_atlas(additional_entries=atlas, strategy_summary=...).",
    "input_schema": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source to run"}},
                     "required": ["code"]},
}


def tool_definitions(args) -> list:
    claude_defs = {d["name"]: d for d in _CURRENT["B"].claude_tool_defs()}
    if args.mode == "tools":
        from databases.tools import DatabaseTools
        defs = [{"name": f["function"]["name"], "description": f["function"]["description"],
                 "input_schema": f["function"]["parameters"]}
                for f in DatabaseTools.get_tool_definitions() if f["function"]["name"] in DB_TOOL_NAMES]
    else:
        defs = [claude_defs[n] for n in LIVE_TOOL_NAMES]
    defs += [INGEST_DEF, claude_defs["submit_atlas"]]
    if args.code_tool:
        defs.append(RUN_PYTHON_DEF)
    defs = json.loads(json.dumps(defs))  # private copy before adding cache_control
    defs[-1]["cache_control"] = {"type": "ephemeral"}
    return defs


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


def dispatch(name: str, inp: dict, funcs: dict, sandbox: Optional[PySandbox], deadline: float):
    if name == "run_python" and sandbox is not None:
        code = str(inp.get("code", ""))
        t = time.time()
        T().via = "run_python"
        try:
            out = sandbox.run(code, deadline - time.time())
        finally:
            T().via = "direct"
        T().code_cells.append(code)
        failed = "Traceback (most recent call last)" in out
        T().record("run_python", {"code": code}, result=None if failed else out,
                   error=out[-500:] if failed else None, dur=time.time() - t)
        return out
    fn = funcs.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**inp)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _set_cache_mark(messages: list):
    """Keep one cache breakpoint on the newest user turn (tools + system hold the other two)."""
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if isinstance(b, dict):
                    b.pop("cache_control", None)
    last = messages[-1]
    if last["role"] == "user":
        if isinstance(last["content"], str):
            last["content"] = [{"type": "text", "text": last["content"]}]
        last["content"][-1]["cache_control"] = {"type": "ephemeral"}


def _shrink_old_results(messages: list, keep_last: int, max_chars: int = 800) -> int:
    """Context guard: shorten tool results older than the last `keep_last` user turns."""
    idx = [i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], list)]
    n = 0
    for i in idx[:-keep_last] if keep_last else idx:
        for b in messages[i]["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_result" and isinstance(b.get("content"), str) \
                    and len(b["content"]) > max_chars:
                b["content"] = b["content"][:max_chars] + "...(older result shortened by harness)"
                n += 1
    return n


def call_claude(client, kw: dict, tries: int = 6):
    import anthropic
    for attempt in range(tries):
        try:
            with client.messages.stream(**kw) as s:
                return s.get_final_message()
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            wait = min(120, 15 * (attempt + 1))
            print(f"[API] {type(e).__name__}; retrying in {wait}s")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", 0) == 529 and attempt < tries - 1:  # overloaded
                time.sleep(min(120, 20 * (attempt + 1)))
                continue
            raise
    raise RuntimeError(f"API failed after {tries} attempts")


def run_claude(args, prompt: str, funcs: dict, log_path: Path) -> dict:
    import anthropic
    client = anthropic.Anthropic(max_retries=3)
    tools = tool_definitions(args)
    sandbox = PySandbox(funcs, args.code_timeout_s, args.max_result_chars) if args.code_tool else None
    system = [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
    messages: list = [{"role": "user", "content": "Begin."}]
    deadline = time.time() + args.timeout_min * 60
    trace, final_text, stop_reason, turn, nudges, grace_turns = [], None, None, 0, 0, 0
    with tee_to(log_path):
        print(f"[SETUP] model={args.model} max_turns={args.max_turns} timeout={args.timeout_min}m "
              f"code_tool={args.code_tool} tools={[t['name'] for t in tools]}")
        try:
            for turn in range(1, args.max_turns + 1):
                if T().stop_reason:
                    break
                if time.time() >= deadline:
                    T().stop_reason = "timeout"
                    break
                if T().llm_calls and T().llm_calls[-1]["input_tokens"] + T().llm_calls[-1]["cache_read_tokens"] \
                        + T().llm_calls[-1]["cache_write_tokens"] > args.context_limit:
                    print(f"[RUNNER] context guard: shortened {_shrink_old_results(messages, 6)} old tool results")
                _set_cache_mark(messages)
                kw = dict(kw_base(args, system, tools), messages=messages)
                t = time.time()
                resp = call_claude(client, kw)
                T().record_llm(args.model, resp, time.time() - t, turn)
                c = T().llm_calls[-1]
                print(f"[TOKENS] turn {turn}: in={c['input_tokens']:,} cache_r={c['cache_read_tokens']:,} "
                      f"cache_w={c['cache_write_tokens']:,} out={c['output_tokens']:,} stop={resp.stop_reason}")

                texts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
                uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
                if texts:
                    final_text = "\n".join(texts)
                    print(f"[CLAUDE] {_short(final_text, 400)}")
                messages.append({"role": "assistant", "content": resp.content})
                step = {"turn": turn, "stop_reason": resp.stop_reason, "text": _short(final_text, 1500) if texts else "",
                        "tool_calls": [u.name for u in uses]}

                if not uses:
                    trace.append(step)
                    if resp.stop_reason == "max_tokens":
                        messages.append({"role": "user", "content": "Continue."})
                        continue
                    if not T().submissions and nudges < args.nudges:
                        nudges += 1
                        messages.append({"role": "user", "content": "You have not called submit_atlas yet. "
                                         "Continue the task and submit the atlas when finished."})
                        continue
                    stop_reason = "end_turn"
                    break

                results = []
                for u in uses:
                    t = time.time()
                    out = dispatch(u.name, dict(u.input or {}), funcs, sandbox, deadline)
                    s = out if isinstance(out, str) else json.dumps(out, default=str)
                    if len(s) > args.max_result_chars:
                        s = s[:args.max_result_chars] + f"...(truncated, {len(s)} chars total)"
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": s})
                    print(f"[TOOL] {u.name}({_short(u.input, 120)}) -> {len(s)} chars ({time.time() - t:.1f}s)")
                    if u.name == "run_python":
                        print(f"[PY] {_short(s, 400)}")
                step["tool_results_chars"] = [len(r["content"]) for r in results]
                trace.append(step)
                messages.append({"role": "user", "content": results})
            else:
                stop_reason = "max_turns"
            budget_hit = T().stop_reason or (stop_reason if stop_reason == "max_turns" else None)
            if budget_hit and not T().submissions and args.grace_turns:
                T().stop_reason = T().stop_reason or budget_hit  # disables every tool except submit_atlas
                grace_turns = finalize_turns(args, client, kw_base(args, system, tools), messages, funcs, sandbox,
                                             trace, turn, budget_hit)
                if T().submissions:
                    final_text = trace[-1].get("text") or final_text
        except KeyboardInterrupt:  # manual stop still produces atlas/run_log/scores
            stop_reason = "interrupted"
        except Exception as e:
            stop_reason = f"error: {type(e).__name__}: {e}"
            print("\n[RUNNER ERROR]\n" + traceback.format_exc())
    if T().stop_reason:
        stop_reason = T().stop_reason
    if T().submissions and stop_reason in (None, "end_turn"):
        stop_reason = "submitted"
    return {"stop_reason": stop_reason or "unknown", "final_answer": final_text, "trace": trace,
            "turns": turn + grace_turns, "grace_turns_used": grace_turns,
            "namespace": dict(sandbox.ns) if sandbox else {}}


def kw_base(args, system, tools) -> dict:
    kw = dict(model=args.model, max_tokens=args.max_tokens, system=system, tools=tools)
    if args.thinking_budget:
        kw["thinking"] = {"type": "enabled", "budget_tokens": args.thinking_budget}
    return kw


def finalize_turns(args, client, kw: dict, messages: list, funcs: dict, sandbox, trace: list, turn: int,
                   reason: str) -> int:
    """Budget ran out before submit_atlas: tell Claude, give it --grace-turns turns to submit.
    Data tools now return an error; run_python still runs, so it can submit variables it already has."""
    note = (f"[HARNESS] The run budget is exhausted ({reason}). Data tools are now disabled. Call submit_atlas "
            f"now with the entries you already have (from run_python, pass your Python list as "
            f"additional_entries). Do not start new queries.")
    last = messages[-1]
    if last["role"] == "user" and isinstance(last["content"], list):
        last["content"].append({"type": "text", "text": note})
    else:
        messages.append({"role": "user", "content": note})
    print(f"\n[RUNNER] {note}")
    used = 0
    for g in range(1, args.grace_turns + 1):
        _set_cache_mark(messages)
        t = time.time()
        resp = call_claude(client, dict(kw, messages=messages))
        used += 1
        T().record_llm(args.model, resp, time.time() - t, turn + g)
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        texts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        trace.append({"turn": turn + g, "grace": True, "stop_reason": resp.stop_reason,
                      "text": _short("\n".join(texts), 1500), "tool_calls": [u.name for u in uses]})
        if not uses:
            break
        results = []
        for u in uses:
            out = dispatch(u.name, dict(u.input or {}), funcs, sandbox, time.time() + 300)
            s = out if isinstance(out, str) else json.dumps(out, default=str)
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": s[:args.max_result_chars]})
            print(f"[GRACE TOOL] {u.name}({_short(u.input, 120)}) -> {_short(s, 300)}")
        messages.append({"role": "user", "content": results})
        if T().submissions:
            break
    return used


def _atlas_like(v) -> Optional[list]:
    if isinstance(v, dict):
        v = list(v.values())
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], dict) and "kinase_gene" in v[0]:
        return list(v)
    return None


def recover_atlas(namespace: dict) -> tuple:
    """If Claude never submitted, take the largest atlas-like variable in its run_python state.
    Also looks one level into dicts of lists, e.g. raw = {"psp": [...], "signor": [...]} (concatenated)."""
    best, best_name = None, None
    for name, v in list(namespace.items()):
        if name.startswith("__"):
            continue
        cands = [(name, _atlas_like(v))]
        if isinstance(v, dict):
            parts = [(k, _atlas_like(x)) for k, x in v.items() if isinstance(x, (list, tuple, dict))]
            parts = [(k, p) for k, p in parts if p]
            if parts:
                cands.append((f"{name}[*]", [e for _, p in parts for e in p]))
        for cname, cand in cands:
            if cand and (best is None or len(cand) > len(best)):
                best, best_name = cand, cname
    return best, best_name


def run_dry(args, prompt: str, funcs: dict, log_path: Path) -> dict:
    """Stub agent: exercises tool defs, harness tools, run_python, submit and scoring - no API."""
    lines = [f"[DRY RUN] prompt chars={len(prompt)} tools={[t['name'] for t in tool_definitions(args)]}"]
    sandbox = PySandbox(funcs, args.code_timeout_s, args.max_result_chars)
    deadline = time.time() + 120
    if args.mode == "tools":
        code = ("atlas = []\nfor k in list_kinases(db='psp', offset=0, limit=5)['kinases']:\n"
                "    for e in query_by_kinase(db='psp', gene=k)['entries']:\n"
                "        atlas.append({**{c: e.get(c) for c in ['kinase_gene','substrate_gene','phospho_site',"
                "'heptameric_peptide','substrate_uniprot']}, 'supporting_databases': ['PSP']})\nlen(atlas)")
    else:
        lines.append(f"[DRY RUN] web_fetch(bad url) -> {_short(funcs['web_fetch']('http://127.0.0.1:9/none'))}")
        code = "atlas = [{'kinase_gene':'CDK1','substrate_gene':'RB1','phospho_site':'S807','supporting_databases':['PSP']}]\nlen(atlas)"
    lines.append(f"[DRY RUN] run_python(build) -> {dispatch('run_python', {'code': code}, funcs, sandbox, deadline).strip()}")
    leak = f"open({str(GOLD_PATH)!r}).read(10)"
    lines.append(f"[DRY RUN] run_python(open gold) -> {_short(dispatch('run_python', {'code': leak}, funcs, sandbox, deadline).strip().splitlines()[-1])}")
    Path("dry_signor.json").write_text(json.dumps(
        [{"kinase_gene": "CDK1", "substrate_gene": "RB1", "phospho_site": "S807", "supporting_databases": ["SIGNOR"]}]))
    lines.append(f"[DRY RUN] ingest -> {dispatch('ingest_entries_file', {'path': 'dry_signor.json', 'source_name': 'SIGNOR'}, funcs, sandbox, deadline)}")
    sub = "submit_atlas(additional_entries=atlas, strategy_summary='dry-run stub')"
    lines.append(f"[DRY RUN] run_python(submit) -> {dispatch('run_python', {'code': sub}, funcs, sandbox, deadline).strip()}")
    log_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return {"stop_reason": "dry_run", "final_answer": None, "trace": [], "turns": 0, "namespace": dict(sandbox.ns)}


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


def prices(args) -> tuple:
    p = list(PRICING.get(args.model, (None, None, None, None)))
    for i, override in enumerate((args.price_in, args.price_out, args.price_cache_read, args.price_cache_write)):
        if override is not None:
            p[i] = override
    if p[2] is None and p[0] is not None:
        p[2] = p[0] * 0.1
    if p[3] is None and p[0] is not None:
        p[3] = p[0] * 1.25
    return tuple(p)


def token_summary(tracker: Tracker, args) -> dict:
    pi, po, pr, pw = prices(args)
    cost = None
    if pi is not None and po is not None:
        cost = 0.0
        for c in tracker.llm_calls:
            c["cost_usd"] = round((c["input_tokens"] * pi + c["output_tokens"] * po
                                   + c["cache_read_tokens"] * pr + c["cache_write_tokens"] * pw) / 1e6, 6)
            cost += c["cost_usd"]
    uncached = sum(c["input_tokens"] for c in tracker.llm_calls)
    cr = sum(c["cache_read_tokens"] for c in tracker.llm_calls)
    cw = sum(c["cache_write_tokens"] for c in tracker.llm_calls)
    tout = sum(c["output_tokens"] for c in tracker.llm_calls)
    return {
        "total_input_tokens": uncached + cr + cw,
        "total_output_tokens": tout,
        "total_tokens": uncached + cr + cw + tout,
        "uncached_input_tokens": uncached,
        "cache_read_input_tokens": cr,
        "cache_creation_input_tokens": cw,
        "api_calls": len(tracker.llm_calls),
        "estimated_cost_usd": round(cost, 4) if cost is not None else None,
        "cost_source": "pricing_table" if cost is not None else "unpriced (pass --price-in/--price-out)",
        "pricing_usd_per_mtok": {"input": pi, "output": po, "cache_read": pr, "cache_write": pw},
        "calls_missing_usage": sum(1 for c in tracker.llm_calls if c["usage_missing"]),
        "per_call": tracker.llm_calls,
        "note": "Exact counts from every Anthropic Messages API response. total_input_tokens = uncached + "
                "cache_read + cache_creation (Anthropic reports these separately; OpenRouter's prompt_tokens "
                "includes all three, so this matches the STELLA runs).",
    }


def leakage_audit(text: str, mode: str) -> dict:
    hits = sorted({p for p in LEAK_PATTERNS if p and p in text})
    db_hits = str(BENCH_ROOT / "databases") in text
    return {"patterns_checked": LEAK_PATTERNS, "hits": hits, "local_db_path_in_agent_code": db_hits,
            "clean": not hits and not db_hits}


def git_rev(path, *git_args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), *(git_args or ("rev-parse", "--short", "HEAD"))],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "?"


def anthropic_version() -> str:
    try:
        import anthropic
        return getattr(anthropic, "__version__", "?")
    except Exception:
        return "not installed"


def run_one(args, workdir: Path, out: Path, feedback: str = "", round_num: int = 1) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "atlas.json").unlink(missing_ok=True)
    tracker = Tracker(_CURRENT["B"].EntryAccumulator)
    tracker.max_tool_calls = args.max_tool_calls
    _CURRENT["tracker"] = tracker
    prompt, base, note, prompt_file = build_prompt(args, feedback)
    (out / "prompt.txt").write_text(prompt)  # exact system prompt sent to Claude
    print(f"[SETUP] model={args.model} condition={args.condition} mode={args.mode} round={round_num}")
    print(f"[SETUP] output={out}  workdir={workdir}")

    funcs = harness_funcs(args.mode)
    t_start = time.time()
    os.chdir(workdir)
    try:
        res = run_dry(args, prompt, funcs, out / "run.log") if args.dry_run else \
            run_claude(args, prompt, funcs, out / "run.log")
    finally:
        os.chdir(BENCH_ROOT)
    elapsed = time.time() - t_start

    recovered_from = None
    if tracker.submissions:
        atlas_source = "submit_atlas"
    elif tracker.acc.size:
        atlas_source = "accumulated_without_submit"
    else:
        cand, recovered_from = recover_atlas(res.get("namespace") or {})
        if cand:
            with tracker.lock:
                _add_entries(cand, "recovered")
        atlas_source = f"recovered_from_namespace:{recovered_from}" if tracker.acc.size else "none"
    entries = tracker.acc.finalize()
    (out / "atlas.json").write_text(json.dumps(entries, indent=2))
    summary = score(entries, out / "scores")

    produced = [{"file": str(p.relative_to(workdir)), "bytes": p.stat().st_size}
                for p in sorted(workdir.rglob("*")) if p.is_file() and p.stat().st_mtime >= t_start
                and "__pycache__" not in p.parts]
    script_text = "\n".join(p.read_text(errors="replace")[:200000] for p in workdir.rglob("*")
                            if p.is_file() and p.suffix in (".py", ".sh", ".R") and p.stat().st_mtime >= t_start)
    if tracker.code_cells:
        (out / "code_cells.py").write_text("".join(f"# ── cell {i} ──\n{c}\n\n" for i, c in enumerate(tracker.code_cells, 1)))

    db_counts = Counter(db for e in entries for db in (e.get("supporting_databases") or []))
    hosts = Counter(h["url"].split("/")[2] if "://" in h["url"] else h["url"] for h in tracker.http)
    http_fail = [h for h in tracker.http if h["error"] or (h["status"] and h["status"] >= 400)]
    run_log = {
        "agent": f"Claude API tool-use loop ({args.model})",
        "framework": "Anthropic Messages API (direct; streaming; prompt caching)"
                     + (" + run_python code tool" if args.code_tool else ""),
        "benchmark_commit": git_rev(BENCH_ROOT),
        "anthropic_sdk_version": anthropic_version(),
        "python": platform.python_version(),
        "model": args.model,
        "served_models": sorted({c["served_model"] for c in tracker.llm_calls if c["served_model"]}),
        "condition": args.condition if round_num == 1 else f"{args.condition}_round{round_num}",
        "mode": args.mode,
        "round": round_num,
        "prompt_file": prompt_file,
        "prompt_chars": len(prompt),
        "harness_note": note,
        "feedback_included": bool(feedback),
        "api_config": {"max_tokens": args.max_tokens, "thinking_budget": args.thinking_budget or "off",
                       "code_tool": args.code_tool, "code_timeout_s": args.code_timeout_s,
                       "max_result_chars": args.max_result_chars, "context_limit": args.context_limit,
                       "nudges": args.nudges},
        "budget": {"timeout_min": args.timeout_min, "max_turns": args.max_turns,
                   "max_tool_calls": args.max_tool_calls or "off", "grace_turns": args.grace_turns},
        "stop_reason": res["stop_reason"],
        "grace_turns_used": res.get("grace_turns_used", 0),
        "strategy": tracker.strategy,
        "final_answer": _short(res["final_answer"], 4000) if res["final_answer"] else None,
        "atlas_source": atlas_source,
        "submissions": tracker.submissions,
        "databases_accessed": sorted(db_counts),
        "raw_counts": dict(db_counts),
        "merged_atlas": len(entries),
        "unique_kinases": len({e["kinase_gene"] for e in entries}),
        "unique_substrates": len({e["substrate_gene"] for e in entries}),
        "multi_db_entries": sum(1 for e in entries if len(e.get("supporting_databases") or []) >= 2),
        "tool_calls": sum(tracker.calls.values()),
        "tool_calls_direct": sum(1 for e in tracker.events if e["via"] == "direct"),
        "tool_calls_from_code": sum(1 for e in tracker.events if e["via"] == "run_python"),
        "agent_turns": res["turns"],
        "function_calls": dict(tracker.calls.most_common()),
        "failures": tracker.failures,
        "http_hosts": dict(hosts),
        "http_failures": http_fail,
        "http_requests": tracker.http,
        "agent_output_files": produced,
        "elapsed_seconds": round(elapsed, 1),
        "token_usage": token_summary(tracker, args),
        "leakage_audit": leakage_audit("\n".join(tracker.audit_chunks) + "\n".join(tracker.code_cells) + script_text,
                                       args.mode),
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
    print(f"[COST] {tu['total_tokens']:,} tokens over {tu['api_calls']} API calls ≈ ${tu['estimated_cost_usd']} "
          f"({tu['cost_source']})")
    if not run_log["leakage_audit"]["clean"]:
        print(f"[WARN] leakage audit hits: {run_log['leakage_audit']}")
    return summary


def wiring_check(args, workdir: Path):
    import anthropic
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("[ERROR] ANTHROPIC_API_KEY not set")
    _CURRENT["tracker"] = Tracker(_CURRENT["B"].EntryAccumulator)
    prompt, *_ = build_prompt(args)
    tools = tool_definitions(args)
    n = anthropic.Anthropic().messages.count_tokens(
        model=args.model, system=prompt, tools=tools, messages=[{"role": "user", "content": "Begin."}])
    print("[WIRING] model:", args.model, "| anthropic SDK:", anthropic_version(), "| python:", platform.python_version())
    print("[WIRING] tools:", [t["name"] for t in tools])
    print(f"[WIRING] first-turn input tokens (count_tokens, no generation): {n.input_tokens:,}")
    print("[WIRING] pricing (USD/Mtok in, out, cache_r, cache_w):", prices(args))
    print("[WIRING] gold:", "found" if GOLD_PATH.exists() else f"MISSING ({GOLD_PATH})", "| workdir:", workdir)


def main():
    ap = argparse.ArgumentParser(description="Run Claude (Fable 5.2 by default) on the PhosphoAtlas benchmark")
    ap.add_argument("--condition", choices=list(PROMPTS), default="naive")
    ap.add_argument("--prompt-file", default=None, help="Use this prompt instead of the condition's prompt")
    ap.add_argument("--mode", choices=["live", "tools"], default="live")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Anthropic model id (default {DEFAULT_MODEL})")
    ap.add_argument("--max-tokens", type=int, default=32000, help="max_tokens per API call")
    ap.add_argument("--thinking-budget", type=int, default=0, help="Extended thinking budget tokens (0 = off)")
    ap.add_argument("--no-code-tool", dest="code_tool", action="store_false", help="Disable run_python")
    ap.add_argument("--code-timeout-s", type=float, default=900, help="Wall-clock limit per run_python cell")
    ap.add_argument("--max-result-chars", type=int, default=30000, help="Truncate each tool result to N chars")
    ap.add_argument("--context-limit", type=int, default=600000,
                    help="If a turn's input exceeds this many tokens, shorten old tool results")
    ap.add_argument("--max-turns", type=int, default=100, help="Max API calls per round")
    ap.add_argument("--max-tool-calls", type=int, default=0,
                    help="Stop after N tool calls incl. those made from run_python (0 = off, default). Off for the "
                         f"same reason as the STELLA runner: config.yaml's {MAX_TOOL_CALLS} assumes one call per turn, "
                         "but pipeline_guided in tools mode needs ~7,600 DB queries when looped in code. "
                         "Budget = --max-turns + --timeout-min")
    ap.add_argument("--grace-turns", type=int, default=2,
                    help="When a budget (turns / timeout / tool calls) runs out, give Claude this many extra turns "
                         "to call submit_atlas; other tools are disabled then (0 = stop immediately)")
    ap.add_argument("--timeout-min", type=float, default=60, help="Wall-clock budget per round (config.yaml: 60)")
    ap.add_argument("--nudges", type=int, default=0,
                    help="Times to remind Claude to submit if it stops without submit_atlas (default 0)")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--price-cache-read", type=float, default=None, help="USD per 1M cache-read tokens")
    ap.add_argument("--price-cache-write", type=float, default=None, help="USD per 1M cache-write tokens")
    ap.add_argument("--iterative", action="store_true", help="3 rounds with scorer feedback")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workdir", default=None, help="Agent scratch dir (default: fresh temp dir outside repo)")
    ap.add_argument("--dry-run", action="store_true", help="Stub agent; tests harness without API")
    ap.add_argument("--wiring-check", action="store_true", help="Check key/model/tools with count_tokens; no generation")
    args = ap.parse_args()

    if args.thinking_budget and args.thinking_budget >= args.max_tokens:
        ap.error("--thinking-budget must be smaller than --max-tokens")
    if args.condition == "pipeline_guided" and args.mode == "live" and not args.prompt_file:
        print("[NOTE] pipeline_guided names DatabaseTools tools (list_databases, query_by_kinase, ...). "
              "In live mode they don't exist; use --mode tools and a *_tools folder for that comparison.")
    if not (args.dry_run or args.wiring_check) and not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("[ERROR] ANTHROPIC_API_KEY not set")
    out = Path(args.output).resolve()

    _CURRENT["B"] = import_benchmark()
    if args.mode == "tools":
        from databases.tools import DatabaseTools
        _CURRENT["db"] = DatabaseTools(str(BENCH_ROOT / "databases"))
    patch_http()
    workdir = Path(args.workdir).resolve() if args.workdir else Path(tempfile.mkdtemp(prefix="fable_pa_"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(BENCH_ROOT)

    if args.wiring_check:
        wiring_check(args, workdir)
        return
    if not args.iterative:
        run_one(args, workdir, out)
        return

    # Iterative: fresh conversation each round, same workdir (files from earlier rounds stay available).
    from experiments.iterative_refinement import generate_feedback
    rows, blocks, feedback = [], [], ""
    for r in range(1, args.rounds + 1):
        rdir = out if r == 1 else out / f"round{r}"
        if r > 1:
            prev = (out if r == 2 else out / f"round{r-1}") / "scores" / "summary.json"
            blocks.append(generate_feedback(prev, r - 1))
            feedback = (f"\n\nYou are on Round {r} of an iterative curation process. "
                        f"Below is feedback from your previous round(s). Use it to improve.\n" + "".join(blocks))
        s = run_one(args, workdir, rdir, feedback, r)
        rows.append({"round": r, **s["overview"], **{k: s["atlas_level"][k] for k in
                     ["total_agent_entries", "total_gold_entries", "true_positives", "false_positives", "false_negatives"]}})
    (out / "iterative_comparison.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
