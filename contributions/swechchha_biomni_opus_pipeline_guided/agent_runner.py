#!/usr/bin/env python3
"""
Biomni A1 runner for the PhosphoAtlas Agent Benchmark.

Wraps Stanford's Biomni A1 agent (a code-executing LangGraph agent) so it can
be run under the benchmark's prompts and produce the standard contribution
files: atlas.json, run_log.json (with exact token_usage), run.log, scores/.

Run from the agent-pa-benchmark repo root, inside the Biomni conda env
(Python >= 3.11, `pip install -e ../Biomni` or pass --biomni-repo):

  # 0. Harness self-test (no Biomni, no API calls)
  python contributions/swechchha_biomni_a1/agent_runner.py --dry-run \
      --condition pipeline_guided --mode tools --output /tmp/biomni_dry

  # 1. Cheap smoke test (a few turns, real API)
  python contributions/swechchha_biomni_a1/agent_runner.py \
      --biomni-repo ../Biomni --llm claude-sonnet-4-6 --condition naive \
      --mode live --max-turns 4 --output /tmp/biomni_smoke

  # 2. Real runs (one folder per condition)
  python contributions/swechchha_biomni_a1/agent_runner.py \
      --biomni-repo ../Biomni --llm claude-sonnet-4-6 --condition naive \
      --mode live --output contributions/swechchha_biomni_sonnet_naive

  # 3. Iterative (3 rounds with scorer feedback)
  python contributions/swechchha_biomni_a1/agent_runner.py \
      --biomni-repo ../Biomni --llm claude-sonnet-4-6 --condition naive \
      --mode live --iterative --output contributions/swechchha_biomni_sonnet_naive

Modes:
  live   Agent gets web_search() + its own Python (requests/pandas) to download
         PSP / SIGNOR / UniProt itself. Matches claude_runner / live_runner runs.
  tools  Agent gets the local DatabaseTools API as Python functions
         (list_databases, query_by_kinase, ...). Matches the tool names that
         pipeline_guided.txt refers to. Needs databases/ files.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from collections import Counter
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))

GOLD_PATH = BENCH_ROOT / "gold_standard" / "parsed" / "phosphoatlas_gold.json"
PROMPTS = {
    "naive": "agents/prompts/naive.txt",
    "paper_informed": "agents/prompts/paper_informed.txt",
    "pipeline_guided": "agents/prompts/pipeline_guided.txt",
}
# Strings that should never appear in agent-written code (gold / other runs).
LEAK_PATTERNS = ["phosphoatlas_gold", "gold_standard", "contributions/", BENCH_ROOT.name, str(BENCH_ROOT)]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt adapter (appended identically to every condition; logged in run_log)
# ─────────────────────────────────────────────────────────────────────────────

SUBMIT_DOC = """  submit_atlas(entries, strategy_summary="") -> dict
      entries: list of dicts with at least kinase_gene, substrate_gene, phospho_site
      (optional: heptameric_peptide, substrate_uniprot, supporting_databases)."""

TOOLS_DOC = """  list_databases() -> dict
  list_kinases(db, offset=0, limit=100) -> dict
  list_substrates(db, offset=0, limit=100) -> dict
  get_stats(db) -> dict
  search(db, keyword, limit=50) -> dict
  query_by_kinase(db, gene) -> dict
  query_by_substrate(db, gene) -> dict
  query_by_site(db, gene, site) -> dict
  query_all_dbs(gene) -> dict"""

LIVE_DOC = """  web_search(query, max_results=10) -> dict   (DuckDuckGo titles + URLs)
  You may also use Python (requests, urllib, gzip, pandas, ...) to download and parse data from URLs."""

HARNESS_NOTE = """

=== HARNESS NOTE (Biomni adapter; identical across conditions) ===
You act by writing code in <execute> blocks. The tools referred to above are
already defined as Python functions in your namespace:
{tool_doc}
{submit_doc}
The atlas is too large to paste into a message: build it in Python and call
submit_atlas(...) from an <execute> block. After it returns status "accepted",
reply with a short <solution> describing your strategy."""


def build_prompt(condition: str, mode: str, feedback: str = "") -> tuple[str, str, str]:
    base = (BENCH_ROOT / PROMPTS[condition]).read_text().strip()
    note = HARNESS_NOTE.format(tool_doc=TOOLS_DOC if mode == "tools" else LIVE_DOC, submit_doc=SUBMIT_DOC)
    return base + feedback + note, base, note


# ─────────────────────────────────────────────────────────────────────────────
# Instrumentation: token usage, tool calls, HTTP requests
# ─────────────────────────────────────────────────────────────────────────────

class Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.llm_calls: list[dict] = []
        self.tool_calls: Counter = Counter()
        self.tool_errors: list[dict] = []
        self.http: list[dict] = []
        self.submissions: list[dict] = []
        self.atlas: list | None = None
        self.strategy = ""

    def agent_turns(self) -> int:
        return sum(1 for c in self.llm_calls if c["caller"] == "agent")


# Patches are installed once per process and always write to the *current* round's
# tracker (Biomni modules cache get_llm at import, so per-round re-patching would leak).
_CURRENT: dict = {"tracker": None, "patched": False, "strip_temperature": False}

TEMPERATURE_PATCH = ("Model rejects the `temperature` parameter; Biomni's default (0.7) is removed from "
                     "the agent LLM and from every LLM Biomni creates internally, so the API default is used.")


def make_usage_handler(caller: str):
    from langchain_core.callbacks import BaseCallbackHandler

    class UsageHandler(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            tracker = _CURRENT["tracker"]
            if tracker is None:
                return
            um, model = {}, ""
            try:
                msg = response.generations[0][0].message
                um = dict(getattr(msg, "usage_metadata", None) or {})
                model = (getattr(msg, "response_metadata", {}) or {}).get("model", "") or (
                    getattr(msg, "response_metadata", {}) or {}).get("model_name", "")
            except Exception:
                pass
            if not um:  # fallback for providers that only fill llm_output
                u = (response.llm_output or {}).get("usage") or (response.llm_output or {}).get("token_usage") or {}
                um = {"input_tokens": u.get("input_tokens", u.get("prompt_tokens", 0)),
                      "output_tokens": u.get("output_tokens", u.get("completion_tokens", 0))}
            det = um.get("input_token_details") or {}
            with tracker.lock:
                tracker.llm_calls.append({
                    "call": len(tracker.llm_calls) + 1,
                    "caller": caller,
                    "model": model,
                    "input_tokens": int(um.get("input_tokens", 0) or 0),
                    "output_tokens": int(um.get("output_tokens", 0) or 0),
                    "cache_read_tokens": int(det.get("cache_read", 0) or 0),
                    "cache_creation_tokens": int(det.get("cache_creation", 0) or 0),
                    "t": round(time.time(), 2),
                })

    return UsageHandler()


def patch_http():
    """Log every HTTP request the agent's code makes (URL, status / error)."""
    def rec(method, url, status=None, error=None, nbytes=None):
        tracker = _CURRENT["tracker"]
        if tracker is None:
            return
        with tracker.lock:
            tracker.http.append({"method": method, "url": str(url)[:300], "status": status,
                                 "error": (str(error)[:200] if error else None), "bytes": nbytes,
                                 "t": round(time.time(), 2)})
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


def make_namespace_tools(tracker: Tracker, mode: str, out_dir: Path, databases_dir: Path) -> dict:
    """Functions injected into Biomni's persistent Python namespace."""

    def counted(name, fn):
        def wrapper(*a, **k):
            with tracker.lock:
                tracker.tool_calls[name] += 1
            try:
                return fn(*a, **k)
            except Exception as e:
                with tracker.lock:
                    tracker.tool_errors.append({"tool": name, "args": repr((a, k))[:200], "error": str(e)[:300]})
                raise
        wrapper.__name__ = name
        wrapper.__doc__ = fn.__doc__
        return wrapper

    def submit_atlas(entries, strategy_summary=""):
        """Submit the completed atlas (list of dicts). Last submission wins."""
        if not isinstance(entries, list):
            return {"status": "rejected", "error": "entries must be a list of dicts"}
        clean, bad = [], 0
        for e in entries:
            e = clean_entry(e) if isinstance(e, dict) else None
            if e and e.get("kinase_gene") and e.get("substrate_gene") and e.get("phospho_site"):
                clean.append(e)
            else:
                bad += 1
        keys = {(str(e["kinase_gene"]).upper(), str(e["substrate_gene"]).upper(), str(e["phospho_site"]).upper())
                for e in clean}
        tracker.atlas, tracker.strategy = clean, str(strategy_summary or "")
        tracker.submissions.append({"entries": len(clean), "invalid_dropped": bad,
                                    "unique_triplets": len(keys), "t": round(time.time(), 2)})
        with open(out_dir / "atlas.json", "w") as f:
            json.dump(clean, f, indent=2)
        return {"status": "accepted", "entries_received": len(clean), "invalid_dropped": bad,
                "unique_triplets": len(keys)}

    funcs = {"submit_atlas": counted("submit_atlas", submit_atlas)}
    if mode == "tools":
        from databases.tools import DatabaseTools
        db = DatabaseTools(str(databases_dir))
        for name in ["list_databases", "list_kinases", "list_substrates", "get_stats", "search",
                     "query_by_kinase", "query_by_substrate", "query_by_site", "query_all_dbs"]:
            funcs[name] = counted(name, getattr(db, name))
    else:
        from agents.live_runner import web_search
        funcs["web_search"] = counted("web_search", web_search)
    return funcs


# ─────────────────────────────────────────────────────────────────────────────
# Agent execution
# ─────────────────────────────────────────────────────────────────────────────

PREFILL_PATCH = ("Biomni stores <observation> results as assistant messages, so requests end on an "
                 "assistant turn (= prefill), which newer Claude models reject. The runner re-sends "
                 "observations as user turns and appends 'Continue.' if a request still ends on an "
                 "assistant turn. Biomni source is unchanged.")


def fix_prefill(messages):
    """Make sure a chat request never ends with an assistant message."""
    if not isinstance(messages, list):
        return messages
    from langchain_core.messages import AIMessage, HumanMessage
    out = []
    for m in messages:
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.lstrip().startswith("<observation>"):
            out.append(HumanMessage(content=m.content))
        else:
            out.append(m)
    if out and isinstance(out[-1], AIMessage):
        out.append(HumanMessage(content="Continue."))
    return out


class NoPrefillLLM:
    """Proxy around Biomni's chat model: fixes message roles, delegates everything else."""

    def __init__(self, llm):
        self._llm = llm

    def invoke(self, messages, *a, **k):
        messages = fix_prefill(messages)
        try:
            return self._llm.invoke(messages, *a, **k)
        except Exception as e:  # e.g. Opus 5.5: "`temperature` is deprecated for this model."
            if "temperature" not in str(e) or getattr(self._llm, "temperature", None) is None:
                raise
            print(f"[PATCH] model rejected temperature={self._llm.temperature}; retrying without it")
            _CURRENT["strip_temperature"] = True
            self._llm.temperature = None
            return self._llm.invoke(messages, *a, **k)

    def __getattr__(self, name):
        return getattr(self._llm, name)


STR_FIELDS = ("kinase_gene", "substrate_gene", "phospho_site", "heptameric_peptide", "substrate_uniprot")


def clean_entry(e: dict) -> dict:
    """Make agent-built entries JSON/scorer-safe: pandas NaN/None -> "", numpy -> python, dbs -> list."""
    out = {}
    for k, v in e.items():
        if hasattr(v, "tolist") and not isinstance(v, (str, list, dict)):
            v = v.tolist()  # numpy scalars / arrays
        if v is None or (isinstance(v, float) and v != v):
            v = ""
        if k == "supporting_databases":
            if isinstance(v, str):
                v = [x.strip() for x in v.replace(";", ",").replace("|", ",").split(",") if x.strip()]
            elif isinstance(v, (list, tuple, set)):
                v = [str(x) for x in v if x is not None and not (isinstance(x, float) and x != x) and str(x)]
            else:
                v = []
        elif k in STR_FIELDS:
            v = str(v).strip()
        out[str(k)] = v
    return out


def recover_atlas(namespace: dict) -> list | None:
    """If the agent never called submit_atlas, take the largest atlas-like list in its namespace."""
    best = None
    for v in list(namespace.values()):
        if isinstance(v, list) and v and isinstance(v[0], dict) and "kinase_gene" in v[0]:
            if best is None or len(v) > len(best):
                best = v
    return best


def run_biomni(args, prompt: str, tracker: Tracker, funcs: dict, workdir: Path, log_path: Path) -> dict:
    # Biomni reads ./.env at import time -> import from a clean cwd so the benchmark's
    # .env (PSP credentials) is not loaded into the agent's environment.
    os.chdir(workdir)
    import biomni.llm as bllm

    if not _CURRENT["patched"]:
        orig_get_llm = bllm.get_llm
        internal_handler = make_usage_handler("biomni_internal")

        def get_llm_tracked(*a, **k):
            llm = orig_get_llm(*a, **k)
            try:
                llm.callbacks = [internal_handler]
                if _CURRENT["strip_temperature"]:
                    llm.temperature = None
            except Exception:
                pass
            return llm
        bllm.get_llm = get_llm_tracked  # covers database-query tools that build their own LLM
        patch_http()
        _CURRENT["patched"] = True

    from biomni.agent import A1
    from biomni.config import default_config
    from biomni.tool import support_tools

    default_config.llm = args.llm
    if args.source:
        default_config.source = args.source
    if args.temperature is not None:
        default_config.temperature = args.temperature

    support_tools._persistent_namespace.clear()
    agent = A1(path=str(workdir / "biomni_root"), llm=args.llm, source=args.source,
               use_tool_retriever=not args.no_retriever, timeout_seconds=args.exec_timeout,
               expected_data_lake_files=[])
    agent.llm.callbacks = [make_usage_handler("agent")]
    if args.omit_temperature:
        _CURRENT["strip_temperature"] = True
    if _CURRENT["strip_temperature"]:
        agent.llm.temperature = None
    agent.llm = NoPrefillLLM(agent.llm)            # see PREFILL_PATCH
    agent._custom_functions = dict(funcs)          # re-injected before every <execute>
    support_tools._persistent_namespace.update(funcs)

    t0, stop, exec_blocks, exec_errors = time.time(), "solution", 0, []
    with open(log_path, "w") as log:
        try:
            for step in agent.go_stream(prompt):
                out = step.get("output", "")
                log.write(out + "\n")
                log.flush()
                if "<observation>" in out:
                    exec_blocks += 1
                    obs = out.split("<observation>", 1)[1][:400]
                    if obs.lstrip().startswith("Error") or "Traceback" in obs or "TIMEOUT" in obs:
                        exec_errors.append(obs)  # heuristic: REPL errors / timeouts
                if time.time() - t0 > args.timeout_min * 60:
                    stop = "timeout"
                    break
                if tracker.agent_turns() >= args.max_turns:
                    stop = "max_turns"
                    break
                if tracker.atlas is not None and args.stop_on_submit:
                    stop = "submitted"
                    break
        except KeyboardInterrupt:  # manual stop still produces atlas/run_log/scores
            stop = "interrupted"
        except Exception as e:  # API errors, GraphRecursionError, ...
            stop = f"error: {type(e).__name__}: {e}"
            log.write("\n[RUNNER ERROR]\n" + traceback.format_exc())

    agent_text = "\n".join(str(x) for x in getattr(agent, "log", []))
    return {"stop_reason": stop, "exec_blocks": exec_blocks, "exec_errors": exec_errors,
            "namespace": support_tools._persistent_namespace, "agent_text": agent_text,
            "biomni_version": getattr(__import__("biomni.version", fromlist=["__version__"]), "__version__", "?"),
            "system_prompt_chars": len(getattr(agent, "system_prompt", ""))}


def run_dry(args, prompt: str, tracker: Tracker, funcs: dict, workdir: Path, log_path: Path) -> dict:
    """Stub agent: exercises tools + submit + scoring without Biomni or an LLM."""
    entries = []
    if "query_by_kinase" in funcs:
        for k in funcs["list_kinases"]("psp", 0, 5)["kinases"]:
            for e in funcs["query_by_kinase"]("psp", k)["entries"]:
                entries.append({kk: e.get(kk) for kk in ["kinase_gene", "substrate_gene", "phospho_site",
                                                         "heptameric_peptide", "substrate_uniprot"]}
                               | {"supporting_databases": ["PSP"]})
    else:
        entries = [{"kinase_gene": "CDK1", "substrate_gene": "RB1", "phospho_site": "S807"}]
    res = funcs["submit_atlas"](entries, "dry-run stub")
    log_path.write_text(f"[DRY RUN] prompt chars={len(prompt)}\n[DRY RUN] submit -> {res}\n")
    return {"stop_reason": "dry_run", "exec_blocks": 0, "exec_errors": [], "namespace": {},
            "agent_text": "", "biomni_version": "n/a", "system_prompt_chars": 0}


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
    from agents.live_runner import MODEL_PRICING
    key = args.pricing or next((k for k in ["opus", "sonnet", "haiku", "gpt-5", "gpt-4o", "gemini-pro"]
                                if k in args.llm.lower()), None)
    p = dict(MODEL_PRICING.get(key, {"input": 0.0, "output": 0.0, "label": "unknown"}))
    if args.price_in is not None:
        p["input"] = args.price_in
    if args.price_out is not None:
        p["output"] = args.price_out
    tin = sum(c["input_tokens"] for c in tracker.llm_calls)
    tout = sum(c["output_tokens"] for c in tracker.llm_calls)
    by_caller = {}
    for c in tracker.llm_calls:
        b = by_caller.setdefault(c["caller"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        b["calls"] += 1
        b["input_tokens"] += c["input_tokens"]
        b["output_tokens"] += c["output_tokens"]
    return {
        "total_input_tokens": tin,
        "total_output_tokens": tout,
        "total_tokens": tin + tout,
        "cache_read_input_tokens": sum(c["cache_read_tokens"] for c in tracker.llm_calls),
        "cache_creation_input_tokens": sum(c["cache_creation_tokens"] for c in tracker.llm_calls),
        "api_calls": len(tracker.llm_calls),
        "estimated_cost_usd": round(tin * p["input"] / 1e6 + tout * p["output"] / 1e6, 4),
        "pricing": {"key": key, "input_per_M": p["input"], "output_per_M": p["output"]},
        "by_caller": by_caller,
        "per_call": tracker.llm_calls,
        "note": "Exact counts from LangChain usage_metadata on every LLM call (agent + Biomni-internal).",
    }


def leakage_audit(text: str, mode: str) -> dict:
    hits = sorted({p for p in LEAK_PATTERNS if p and p in text})
    db_hits = mode == "live" and str(BENCH_ROOT / "databases") in text
    return {"patterns_checked": LEAK_PATTERNS, "hits": hits, "local_db_access_in_live_mode": db_hits,
            "clean": not hits and not db_hits}


def git_rev(path: Path | None) -> str:
    if not path:
        return "?"
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def run_one(args, out_dir: Path, feedback: str = "", round_num: int = 1) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "atlas.json").unlink(missing_ok=True)
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="biomni_pa_")) / f"round{round_num}"
    workdir.mkdir(parents=True, exist_ok=True)

    tracker = Tracker()
    _CURRENT["tracker"] = tracker
    prompt, base, note = build_prompt(args.condition, args.mode, feedback)
    funcs = make_namespace_tools(tracker, args.mode, out_dir, BENCH_ROOT / "databases")
    (out_dir / "prompt.txt").write_text(prompt)  # exact user prompt sent to Biomni

    print(f"[SETUP] condition={args.condition} mode={args.mode} llm={args.llm} round={round_num}")
    print(f"[SETUP] output={out_dir}  workdir={workdir}")
    t0 = time.time()
    runner = run_dry if args.dry_run else run_biomni
    res = runner(args, prompt, tracker, funcs, workdir, out_dir / "run.log")
    os.chdir(BENCH_ROOT)
    elapsed = time.time() - t0

    atlas_source = "submit_atlas"
    entries = tracker.atlas
    if entries is None:
        entries = [clean_entry(e) for e in (recover_atlas(res["namespace"]) or []) if isinstance(e, dict)]
        entries = [e for e in entries if e.get("kinase_gene") and e.get("substrate_gene") and e.get("phospho_site")]
        atlas_source = "recovered_from_namespace" if entries else "none"
        (out_dir / "atlas.json").write_text(json.dumps(entries, indent=2))

    db_counts = Counter(db for e in entries for db in (e.get("supporting_databases") or []))
    hosts = Counter(h["url"].split("/")[2] if "://" in h["url"] else h["url"] for h in tracker.http)
    http_fail = [h for h in tracker.http if h["error"] or (h["status"] and h["status"] >= 400)]

    run_log = {
        "agent": f"Biomni A1 ({args.llm})",
        "framework": "Biomni A1",
        "biomni_version": res["biomni_version"],
        "biomni_commit": git_rev(Path(args.biomni_repo) if args.biomni_repo else None),
        "benchmark_commit": git_rev(BENCH_ROOT),
        "model": args.llm,
        "condition": args.condition if round_num == 1 else f"{args.condition}_round{round_num}",
        "mode": args.mode,
        "round": round_num,
        "prompt_file": PROMPTS[args.condition],
        "prompt_chars": len(prompt),
        "harness_note": note,
        "feedback_included": bool(feedback),
        "biomni_system_prompt_chars": res["system_prompt_chars"],
        "biomni_patches": ([] if args.dry_run else [PREFILL_PATCH])
                          + ([TEMPERATURE_PATCH] if _CURRENT["strip_temperature"] else []),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "biomni_config": {"use_tool_retriever": not args.no_retriever, "exec_timeout_s": args.exec_timeout,
                          "temperature": ("omitted (API default)" if _CURRENT["strip_temperature"]
                                          else (args.temperature if args.temperature is not None else 0.7)),
                          "data_lake": "skipped (expected_data_lake_files=[])"},
        "budget": {"timeout_min": args.timeout_min, "max_turns": args.max_turns},
        "stop_reason": res["stop_reason"],
        "strategy": tracker.strategy,
        "atlas_source": atlas_source,
        "submissions": tracker.submissions,
        "databases_accessed": sorted(db_counts),
        "raw_counts": dict(db_counts),
        "merged_atlas": len(entries),
        "unique_kinases": len({e["kinase_gene"] for e in entries}),
        "unique_substrates": len({e["substrate_gene"] for e in entries}),
        "multi_db_entries": sum(1 for e in entries if len(e.get("supporting_databases") or []) >= 2),
        "tool_calls": res["exec_blocks"],
        "agent_turns": tracker.agent_turns(),
        "function_calls": dict(tracker.tool_calls),
        "function_errors": tracker.tool_errors,
        "code_execution_errors": res["exec_errors"],
        "http_hosts": dict(hosts),
        "http_failures": http_fail,
        "http_requests": tracker.http,
        "elapsed_seconds": round(elapsed, 1),
        "token_usage": token_summary(tracker, args),
        "leakage_audit": leakage_audit(res["agent_text"], args.mode),
    }
    # Write run_log BEFORE scoring so token usage is never lost if scoring fails.
    (out_dir / "run_log.json").write_text(json.dumps(run_log, indent=2, default=str))

    me = Path(__file__).resolve()
    if round_num == 1 and me.parent != out_dir.resolve():
        shutil.copy2(me, out_dir / "agent_runner.py")

    try:
        summary = score(entries, out_dir / "scores")
    except Exception as e:
        run_log["score_error"] = f"{type(e).__name__}: {e}"
        (out_dir / "run_log.json").write_text(json.dumps(run_log, indent=2, default=str))
        print(f"[ERROR] scoring failed ({run_log['score_error']}); atlas.json and run_log.json are saved. "
              f"Fix, then re-score with --rescore-only.")
        raise

    ov, tu = summary["overview"], run_log["token_usage"]
    print(f"[DONE] stop={res['stop_reason']} atlas={len(entries)} ({atlas_source}) "
          f"F1={ov['f1']} P={ov['precision']} R={ov['recall']} kinases={ov['kinases_found']}")
    print(f"[COST] {tu['total_tokens']:,} tokens over {tu['api_calls']} LLM calls ≈ ${tu['estimated_cost_usd']}")
    if not run_log["leakage_audit"]["clean"]:
        print(f"[WARN] leakage audit hits: {run_log['leakage_audit']}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run Biomni A1 on the PhosphoAtlas benchmark")
    ap.add_argument("--condition", choices=list(PROMPTS), default="naive")
    ap.add_argument("--mode", choices=["live", "tools"], default="live")
    ap.add_argument("--output", required=True)
    ap.add_argument("--llm", default="claude-sonnet-4-6")
    ap.add_argument("--source", default=None, help="Biomni LLM source (Anthropic, OpenAI, ...); auto if omitted")
    ap.add_argument("--biomni-repo", default=None, help="Path to Biomni clone (adds to sys.path, loads its .env)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--omit-temperature", action="store_true",
                    help="Never send temperature (auto-enabled if the model rejects it)")
    ap.add_argument("--max-turns", type=int, default=50, help="Max agent LLM turns (claude_runner default: 50)")
    ap.add_argument("--timeout-min", type=float, default=60, help="Wall-clock budget (config.yaml: 60)")
    ap.add_argument("--exec-timeout", type=int, default=600, help="Per <execute> block timeout (s)")
    ap.add_argument("--no-retriever", action="store_true", help="Disable Biomni tool retriever")
    ap.add_argument("--stop-on-submit", action="store_true", help="Stop as soon as submit_atlas succeeds")
    ap.add_argument("--pricing", default=None, help="Key in agents/live_runner.MODEL_PRICING")
    ap.add_argument("--price-in", type=float, default=None, help="Override USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="Override USD per 1M output tokens")
    ap.add_argument("--iterative", action="store_true", help="3 rounds with scorer feedback")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workdir", default=None, help="Agent scratch dir (default: fresh temp dir outside repo)")
    ap.add_argument("--dry-run", action="store_true", help="Stub agent; tests harness without Biomni/API")
    ap.add_argument("--rescore-only", action="store_true",
                    help="Don't run the agent: clean <output>/atlas.json and (re)write scores/")
    args = ap.parse_args()

    if args.biomni_repo:
        repo = Path(args.biomni_repo).resolve()
        sys.path.insert(0, str(repo))
        try:
            from dotenv import load_dotenv
            load_dotenv(repo / ".env", override=False)
        except ImportError:
            pass
    out = Path(args.output).resolve()

    if args.rescore_only:
        raw = (out / "atlas.json").read_text()
        data = json.loads(raw.replace(": NaN", ": null"))  # json.dump writes pandas NaN as bare NaN
        data = data.get("atlas", data.get("entries", data)) if isinstance(data, dict) else data
        entries = [clean_entry(e) for e in data if isinstance(e, dict)]
        entries = [e for e in entries if e.get("kinase_gene") and e.get("substrate_gene") and e.get("phospho_site")]
        (out / "atlas.json").write_text(json.dumps(entries, indent=2))
        ov = score(entries, out / "scores")["overview"]
        rl_path = out / "run_log.json"
        rl = json.loads(rl_path.read_text()) if rl_path.exists() else {
            "agent": f"Biomni A1 ({args.llm})", "framework": "Biomni A1", "model": args.llm,
            "condition": args.condition, "mode": args.mode,
            "token_usage": {"note": "UNAVAILABLE: original run crashed before run_log.json was written; "
                                    "see Anthropic console usage for this run's time window."}}
        rl.update({"merged_atlas": len(entries), "rescored": True, "score_error": None})
        rl_path.write_text(json.dumps(rl, indent=2, default=str))
        print(f"[RESCORE] atlas={len(entries)} F1={ov['f1']} P={ov['precision']} R={ov['recall']} "
              f"kinases={ov['kinases_found']}")
        return

    if not args.iterative:
        run_one(args, out)
        return

    from experiments.iterative_refinement import generate_feedback
    rows, blocks, feedback = [], [], ""
    for r in range(1, args.rounds + 1):
        rdir = out if r == 1 else out / f"round{r}"
        if r > 1:
            prev = (out if r == 2 else out / f"round{r-1}") / "scores" / "summary.json"
            blocks.append(generate_feedback(prev, r - 1))
            feedback = (f"\n\nYou are on Round {r} of an iterative curation process. "
                        f"Below is feedback from your previous round(s). Use it to improve.\n" + "".join(blocks))
        s = run_one(args, rdir, feedback, r)
        rows.append({"round": r, **s["overview"], **{k: s["atlas_level"][k] for k in
                     ["total_agent_entries", "total_gold_entries", "true_positives", "false_positives", "false_negatives"]}})
    (out / "iterative_comparison.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
