# Running STELLA on the PhosphoAtlas Benchmark

How to run and test **STELLA** (Jin et al. 2025, arXiv:2507.02004; local clone `../STELLA`, fork `asfarasimconcerned/STELLA`) under this benchmark's prompts, and log the results in the standard contribution format. The runner is `agent_runner.py` in this folder. Nothing is written into the STELLA repo.

---

## 1. The integration problem

| | agent-pa-benchmark expects | STELLA provides |
|---|---|---|
| Entry point | A script that runs one condition and exits | A Gradio chat UI only (`stella_core.py`, `start_stella_web.py`); `initialize_stella()` builds the agents without the UI |
| Agents | One model in a tool loop | smolagents **manager CodeAgent** (`x-ai/grok-4`) driving `dev_agent` (`gemini-2.5-pro`; has shell, scripts and pip), `critic_agent` and `tool_creation_agent` (`claude-sonnet-4`). All models go through OpenRouter |
| Output | `atlas.json` | Free text. The manager's Python sandbox can't `open()` files or import `gzip`/`io`, so dev_agent does the heavy parsing and saves files |
| Token log | `run_log.json -> token_usage` (required) | Nothing. Some LLM calls also happen inside tools (`new_tools/llm.py` for tool selection, and a model inside `database_tools.py`) |
| Budget | 60 min, 5000 tool calls (`config.yaml`) | Manager `max_steps` (smolagents default 20) |
| Side effects | Stay inside your contribution folder | Writes `./new_tools/*.py` (self-evolution) and `./agent_outputs/`, relative to the working directory |

`agent_runner.py` bridges each row:

1. **Harness tools are registered with STELLA before `initialize_stella()`**, so the manager's prompt lists them.
   - **live mode** adds the same tools as `claude_runner.py`: `bench_web_fetch` and `fetch_and_parse_db`. They are imported from it, not copied.
   - **tools mode** wraps `DatabaseTools` directly from its own `get_tool_definitions()`.
   - Both modes add `ingest_entries_file`, so dev_agent can hand back a JSON/CSV/TSV file, and `submit_atlas(additional_entries=<python list>)`. Because the manager is a CodeAgent, it can pass a list of ~16k entries as a Python object.
   - Entries keep their own `supporting_databases`. `claude_runner` labels direct entries `agent_direct`, which would hide multi-DB support.
2. **Exact token usage.** `openai...Completions.create` is patched once. Every STELLA call goes through it: all four agents, the tool-selection LLM and the model in `database_tools`.
   - Each call is logged with `caller` (manager / dev / critic / tool_creation / `stella_internal`) and `model`.
   - The runner asks OpenRouter to return per-call cost (`usage.include`), so `estimated_cost_usd` is the billed amount whenever it is reported (`cost_source`).
3. **Every tool and sub-agent call** is logged through `smolagents.Tool.__call__` and `MultiStepAgent.__call__`, with arguments, duration and the error if one occurred. STELLA tools return `"❌ ..."` strings instead of raising, so the runner counts those as failures too. Every `requests`/`urllib` request is also logged (`http_*`).
4. **Budget.** A timer calls `agent.interrupt()` on all four agents at `--timeout-min`, and the runner does the same once the run reaches 5000 tool calls. The manager's step limit is `--max-steps` (default 30). Files are still written when a run stops early.
5. **Working directory.** STELLA runs in a fresh temp directory **outside** both repos, seeded with a copy of STELLA's `new_tools/`.
   - Any tool STELLA writes is copied into `self_evolved_tools/` in the output folder.
   - STELLA's own `load_dotenv()` resolves to `../STELLA/.env`, so it never loads the benchmark `.env` with the PSP credentials.
6. **Harness note.** One fixed block is appended to every prompt and recorded verbatim in `run_log.json -> harness_note`. It is the only change to the canonical prompts, and `prompt.txt` holds the exact text sent.

---

## 2. Setup (one time)

```bash
# STELLA's asset/runtime.txt pins Python 3.11 (the bundled stella_venv is 3.14 and empty: don't use it)
cd ../STELLA && python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -r requirements.txt
# imported by STELLA but missing from its requirements.txt:
pip install python-dotenv pyyaml biopython langchain-openai tqdm rapidfuzz networkx pymed googlesearch-python
pip install fpdf2 anthropic          # benchmark side (report; claude_runner import - stubbed if absent)
brew install uv                      # optional: STELLA starts a PubMed MCP server with `uvx` at import
# ../STELLA/.env must contain OPENROUTER_API_KEY (the runner loads it via --stella-repo)
```

**Silent degradation to check first.** `new_tools/database_tools.py` holds `query_uniprot`, `query_kegg` and similar tools. It imports `Bio`, `langchain_openai` and `tqdm` at the top. If any of these is missing, STELLA's tool loader skips the whole module without telling you, and the run goes ahead without them. Install them before scoring a run.

**Tools mode** also needs the local `databases/` files (PSP, SIGNOR, UniProt). Your clone already has them.

---

## 3. Testing ladder (run in this order, from the benchmark repo root)

| Step | Command | What it proves | Cost |
|---|---|---|---|
| 0. Harness self-test | `python contributions/swechchha_stella/agent_runner.py --dry-run --mode tools --condition pipeline_guided --output /tmp/stella_dry` | Tool wiring, `ingest`/`submit`, scorer, `run_log.json` schema. Needs no STELLA and no API | $0 |
| 1. Wiring check | `... --stella-repo ../STELLA --wiring-check --mode live --output /tmp/stella_wiring` | STELLA imports, the harness tools appear in the manager tool list, models are as expected. Makes no LLM calls | $0 |
| 2. Smoke test | `... --stella-repo ../STELLA --condition naive --mode live --max-steps 4 --timeout-min 10 --output /tmp/stella_smoke` | The API key works, `token_usage.by_caller` is non-zero, `cost_source` is `openrouter_reported` | Cents |
| 3. Cost check | Read `/tmp/stella_smoke/run_log.json -> token_usage.by_caller` and `by_model` | Cost per manager step. Multiply by `--max-steps` to estimate a full run before you commit to it | — |
| 4. One full run | `... --condition naive --mode live --output contributions/swechchha_stella_default_naive` | End to end: a real atlas plus scores | 1 run |
| 5. All conditions | Repeat with `paper_informed`, and with `pipeline_guided --mode tools` into a `_tools` folder | The main comparison | 3 runs |
| 6. Iterative | `... --condition naive --iterative --output contributions/swechchha_stella_default_naive` | Writes `round2/`, `round3/` and `iterative_comparison.json` | 3 runs |
| 7. Repeats (optional) | Re-run with a `_run2` / `_run3` suffix | `config.yaml` asks for `runs_per_condition: 3`. STELLA's models aren't deterministic | 3× |

Steps 0, 2, 5 and 6 already work end to end with stub agents. See §7.

---

## 4. Experimental design decisions (decide before the real runs)

1. **Backbone LLM = the variable you hold fixed.** STELLA's default mixes three models (Grok-4, Gemini 2.5 Pro, Claude Sonnet 4), so it can't be compared directly with any other row.
   - Run it twice. Once as published (folders `swechchha_stella_default_*`), and once with `--backbone <id>`, which puts all four agents on one model. Use the same backbone as your Biomni and Claude Code runs (e.g. Sonnet 4.6; check the exact OpenRouter id), in folders `swechchha_stella_sonnet_*`.
   - Even with `--backbone`, STELLA's internal tool-selection call (`new_tools/llm.py`) stays on `gemini-2.5-pro`. It shows up as `by_caller.stella_internal`; report it.
2. **Mode: `live` vs `tools`.** Use the same split as Biomni.
   - `live` (default): STELLA keeps all its own tools (web search, `extract_url_content`, `query_uniprot`, and shell access through dev_agent) plus the harness web tools.
   - `tools`: STELLA gets the local DatabaseTools API, so the tool names in `pipeline_guided` exist. Label those folders `_tools`.
   - Don't mix modes within one comparison table.
3. **Counting "tool calls".** `run_log.json` records four counts:
   - `tool_calls`: every tool and sub-agent invocation. `report.py` displays this one.
   - `manager_steps`: the manager's code steps.
   - `agent_turns`: LLM calls made by the agents.
   - `function_calls`: the count per tool.

   One dev_agent call can run a whole download script, so state which count you compare.
4. **Memory and self-evolution.**
   - `--memory off` (the default) keeps runs independent. STELLA's KB/Mem0 paths are hard-coded to `/home/ubuntu/agent_outputs`, so patch them before using `kb` or `mem0`.
   - Self-evolution is preserved where it matters. In `--iterative` mode all rounds share one process and one working directory, so tools created in round 1 stay available in rounds 2 and 3. That condition is the real test of the "self-evolving" claim. Compare `self_evolved_tools` across rounds.
5. **Leakage control (important).** dev_agent can run arbitrary shell commands.
   - The working directory is outside the repo, but an absolute path still reaches `gold_standard/`. So `leakage_audit` searches the manager's code, every tool argument (including shell commands) and every script written during the run for gold-standard or repo paths. In `live` mode it also flags reads of local `databases/`.
   - The runner prints `[WARN]` on any hit. Treat a run with a hit as invalid. For airtight runs, use a machine where the benchmark repo isn't mounted.
6. **Failure logging (`todo.md` asks for this).**
   - `failures` lists every tool or sub-agent error and every `❌` result, with its arguments.
   - `http_failures` and `http_hosts` show which databases were reachable at run time, e.g. PSP login pages or SIGNOR timeouts.

---

## 5. Logging results in this repo

**Folder naming:** `contributions/swechchha_stella_<backbone>_<condition>[_tools]/`, e.g. `swechchha_stella_default_naive`, `swechchha_stella_sonnet_pipeline_guided_tools`.

```
contributions/swechchha_stella_default_naive/
├── agent_runner.py          # copied in automatically (provenance)
├── prompt.txt               # exact prompt sent (condition + harness note [+ feedback])
├── atlas.json
├── run_log.json             # token_usage + commits + failures + http + leakage audit + trace
├── run.log                  # full STELLA console trace
├── scores/{summary,per_kinase,peptide_mismatches}.json
├── self_evolved_tools/      # only if STELLA wrote new tools
└── round2/, round3/, iterative_comparison.json   # only with --iterative
```

**Key `run_log.json` fields:** `model`, `models`, `framework`, `stella_commit`, `benchmark_commit`, `condition`, `mode`, `stop_reason`, `atlas_source`, `token_usage{total_*, by_caller, by_model, per_call, estimated_cost_usd, cost_source}`, `tool_calls`, `agent_turns`, `manager_steps`, `function_calls`, `failures`, `http_failures`, `self_evolved_tools`, `leakage_audit`, `elapsed_seconds`.

**Report caveat.** As with the Biomni and Claude Code folders, `evaluation/report.py` labels unknown folders by their folder name, with condition **`naive`**. Ask Hui Lin to add rows like these, or flag the change clearly in your PR:

```python
"swechchha_stella_default_naive":           ("STELLA (Grok-4/Gemini/Sonnet)", "Agent framework", "naive"),
"swechchha_stella_default_paper_informed":  ("STELLA (Grok-4/Gemini/Sonnet)", "Agent framework", "paper_informed"),
"swechchha_stella_sonnet_pipeline_guided_tools": ("STELLA (Sonnet 4.6)", "Agent framework", "pipeline_guided"),
```

Do not commit `.env`, `databases/*` or the temp working directory.

---

## 6. Interpreting STELLA runs (what to look at)

- **`stop_reason`:**
  - `final_answer`: the manager finished on its own.
  - `max_steps` / `timeout` / `max_tool_calls`: a budget ran out.
  - `error: ...`: API errors, or an interrupt that landed mid-step.
- **`atlas_source`:**
  - `submit_atlas`: the normal case.
  - `accumulated_without_submit`: entries reached the harness, but the manager never submitted. Report it, but flag it.
  - `none`: nothing was produced. Still commit the folder; failed runs are data.
- **`by_caller`:** a large `stella_internal` share means STELLA's tool-selection and database-tool LLM calls are a real part of its cost. That is a framework overhead worth reporting.
- **`function_calls` ∋ `agent:tool_creation_agent`, `self_evolved_tools`:** these show whether STELLA actually used self-evolution on this task.
- **Precision vs. the gold standard:** pulling PSP + UniProt in full gives ~0.8 precision and ~0.88 recall (`claude_sonnet_pipeline_guided`). Low recall with `http_failures` on phosphosite.org usually means the PSP login wall, not the agent.

---

## 7. What has been verified and what hasn't

- **Verified in the sandbox (Python 3.10), against this repo's real modules** (`claude_runner`, `live_runner`, `DatabaseTools`, `evaluation.scorer`, `experiments.iterative_refinement`, real gold standard):
  - `--dry-run` works in both modes and with `--iterative`. Round prompts build up the feedback from earlier rounds.
  - `scores/summary.json` is identical to running `python -m evaluation.scorer` directly on the same atlas.
  - The leakage audit catches gold-standard paths, and local-DB access in live mode.
- **Verified with mock `smolagents` / `openai` / `stella_core` modules**, covering the full real code path:
  - The harness tools register with STELLA, dev_agent receives `ingest_entries_file`, and tools mode wraps DatabaseTools.
  - `--backbone` reaches all four agents.
  - Token counts and OpenRouter-reported cost are attributed to `manager_agent` vs `dev_agent`.
  - Soft `❌` failures and HTTP failures are logged, `self_evolved_tools` is captured, and the timeout interrupts the agents.
  - An agent error still produces `atlas.json`, `run_log.json` and `scores/`.
- **Not verified:** a real STELLA + smolagents + OpenRouter run. The sandbox has no Python 3.11 and no access to install packages, so step 1 (the wiring check) is the first real test.
  - If `token_usage` is empty there, your `openai` version has moved `Completions`. Check `[WARN] openai not importable`.
  - If OpenRouter doesn't return `usage.cost`, `cost_source` falls back to `pricing_table`. Verify those prices.
- `interrupt()` is only checked between agent steps, so one long dev_agent script can overrun `--timeout-min`.
