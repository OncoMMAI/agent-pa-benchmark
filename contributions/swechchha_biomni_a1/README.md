# Running Biomni A1 on the PhosphoAtlas Benchmark

How to run and test Stanford's **Biomni A1** agent (`../Biomni`, v0.0.8, commit `400c1f3`) under this benchmark's prompts and log results in the standard contribution format. The runner is `agent_runner.py` in this folder.

---

## 1. The integration problem

| | agent-pa-benchmark expects | Biomni A1 provides |
|---|---|---|
| Action style | Function calling (`list_databases`, `query_by_kinase`, `submit_atlas`, `web_search`, ...) | Writes code in `<execute>` blocks that run in a persistent Python REPL, then ends with `<solution>` |
| Output | `atlas.json` (list of `{kinase_gene, substrate_gene, phospho_site, ...}`) | Free text. Output is capped at `max_tokens=8192`, so a ~16k-entry atlas can't be pasted |
| Token log | `run_log.json -> token_usage` (required) | No token accounting |
| Budget | 60 min (`config.yaml`), 50 turns (`claude_runner` default) | Only `recursion_limit=500` plus a 600 s timeout on each code block |
| Prompt slot | System prompt = condition prompt | Biomni has its own large system prompt. Your task goes in as the **user** message |

`agent_runner.py` bridges each row:

1. **Tools become Python functions.** Before the run, the benchmark tools are injected into Biomni's REPL namespace (`biomni.tool.support_tools._persistent_namespace` and `agent._custom_functions`). I did not use `agent.add_tool()`: it calls the LLM to write a schema for every tool, which costs tokens and isn't deterministic.
2. **`submit_atlas(entries, strategy_summary)`** is a Python function. It validates the entries and writes `atlas.json` directly. If the agent never calls it, the runner takes the largest atlas-like list left in the REPL namespace and logs `atlas_source: "recovered_from_namespace"`.
3. **Exact token usage.** A LangChain callback is attached to the agent's LLM and, by patching `biomni.llm.get_llm`, to every LLM that Biomni tools create internally (for example `query_uniprot`). Totals are split into `by_caller: agent / biomni_internal`.
4. **Budget.** The runner iterates `agent.go_stream()` and stops on `--timeout-min` (60), `--max-turns` (50) or Ctrl-C. Files are still written when it stops early.
5. **Prefill fix.** Biomni stores each code result (`<observation>`) as an *assistant* message, so the next API request ends on an assistant turn. Claude Sonnet 4.6 rejects that with a 400 error ("does not support assistant message prefill"). The runner wraps the agent's LLM so observations are sent as user turns. Biomni's source is unchanged, and the fix is recorded in `run_log.json -> biomni_patches`.
6. **Harness note.** One fixed block is appended to every prompt and recorded verbatim in `run_log.json -> harness_note`. It tells the agent the tools are Python functions and that it must call `submit_atlas()`. This is the only change to the canonical prompts.

---

## 2. Setup (one time)

```bash
# Biomni needs Python >= 3.11. The light env is enough: skip the 30 GB / 10 h full E1 setup.
cd ../Biomni/biomni_env && conda env create -f environment.yml && conda activate biomni_e1
cd .. && pip install -e .               # local clone, so the commit is pinned and logged
pip install fpdf2                        # only needed for evaluation.report
# ../Biomni/.env must contain ANTHROPIC_API_KEY (the runner loads it via --biomni-repo)
```

**Data lake.** The runner passes `expected_data_lake_files=[]`, which skips the 11 GB download. `biomni/env_desc.py` lists no phosphorylation, kinase-substrate, SIGNOR or OmniPath data, so skipping it doesn't take away a hidden source.

**Tools mode** also needs the local `databases/` files (PSP, SIGNOR, UniProt). They are already present in your clone.

---

## 3. Testing ladder (run in this order, from the benchmark repo root)

| Step | Command | What it proves | Cost |
|---|---|---|---|
| 0. Harness self-test | `python contributions/swechchha_biomni_a1/agent_runner.py --dry-run --mode tools --condition pipeline_guided --output /tmp/biomni_dry` | Tool wiring, `submit_atlas`, scorer, `run_log.json` schema. Needs no Biomni and no API | $0 |
| 1. Smoke test | `... --biomni-repo ../Biomni --llm claude-sonnet-4-6 --condition naive --mode live --max-turns 4 --output /tmp/biomni_smoke` | Biomni imports, the API key works, token counts are non-zero, the agent sees the tools | A few cents |
| 2. Cost check | Read `/tmp/biomni_smoke/run_log.json -> token_usage.by_caller` | Tokens per turn. Multiply by ~50 turns to estimate a full run's cost before you commit to it | — |
| 3. One full run | `... --condition naive --mode live --output contributions/swechchha_biomni_sonnet_naive` | End to end: real atlas plus scores | ~1 run |
| 4. All conditions | Repeat with `paper_informed` and `pipeline_guided` | The main comparison | 3 runs |
| 5. Iterative | `... --condition naive --iterative --output contributions/swechchha_biomni_sonnet_naive` | Writes `round2/`, `round3/` and `iterative_comparison.json` | 3 runs |
| 6. Repeats (optional) | Re-run with a `_run2` / `_run3` suffix | `config.yaml` asks for `runs_per_condition: 3`. Biomni samples at temperature 0.7 | 3× |

Steps 0, 3, 4 and 5 already worked end to end with a stub agent. See §7.

---

## 4. Experimental design decisions (decide before the real runs)

1. **Backbone LLM = the variable you hold fixed.** Biomni is a *framework*, not a model. Use the same backbone as an existing contribution, e.g. `claude-sonnet-4-6` (as in `claude_sonnet_*` and `claude_runner.MODEL_MAP`). Then any difference in F1 comes from the framework, not the model. The closest comparison is Claude Code (`claude_sonnet_*`, `swechchha_claudecode_*`), since both write code to do the work.
2. **Mode: `live` vs `tools`.**
   - `live` (default): the agent gets `web_search()` plus normal Python HTTP access and downloads PSP, SIGNOR and UniProt itself. This is comparable to most contributions.
   - `tools`: the agent gets the local `DatabaseTools` API. Use it for `pipeline_guided` if you want the tool names in that prompt (`list_databases`, `query_by_kinase`, `query_all_dbs`) to exist. Mistral's `todo.md` notes show what goes wrong when those names are missing. Label these folders `_tools`, as in `swechchha_claudecode_pipeline_guided_tools`.
   - Don't mix modes within one comparison table.
3. **Counting "tool calls".** One Biomni code block can run thousands of DB queries; your Claude Code tools run made 12,751. `run_log.json` therefore records three counts:
   - `tool_calls`: the number of `<execute>` blocks. `evaluation/report.py` displays this one.
   - `agent_turns`: the number of LLM calls.
   - `function_calls`: how many times each injected tool was called.

   State which count you use when comparing.
4. **Token cost is exact here; many other contributions are estimates.** The Max-plan and Claude Code runs used `LiveTokenTracker` or `TokenEstimator`, so compare cost with that caveat.
   - Biomni sends its system prompt on **every** turn, with no prompt caching. That prompt includes all know-how docs (~22 KB) plus the retrieved tools, so expect Biomni to cost more per turn than a plain tool loop. That is a finding worth reporting.
   - Pricing comes from `agents/live_runner.MODEL_PRICING`, for consistency with other runs. Override it with `--price-in/--price-out` if list prices change.
5. **Tool retriever.** Biomni's default spends one extra LLM call choosing tools; it is logged under `by_caller.agent`. Keep it on (the standard config) unless you're running an ablation (`--no-retriever`).
6. **Leakage control (important).** Biomni runs arbitrary code with full filesystem access, unlike the API runners.
   - The runner starts the agent in a fresh temp directory **outside** the repo. That means Biomni's import-time `load_dotenv(".env")` does not pick up the benchmark `.env`, which holds the PhosphoSitePlus credentials.
   - The agent can still reach `gold_standard/` by absolute path. So every run is audited: `leakage_audit` searches the agent's code for gold-standard or repo paths, and, in `live` mode, for reads of local `databases/`. The runner prints `[WARN]` on any hit. Treat a run with a hit as invalid. For airtight runs, use a machine or container where the benchmark repo isn't mounted.
7. **Failure logging (`todo.md` asks for this).** Every HTTP request the agent makes through `requests` or `urllib` is logged with its status or error:
   - `http_requests`, `http_failures` and `http_hosts` cover the network side.
   - `code_execution_errors` and `function_errors` capture tracebacks and errors from tools.

   Together they show which databases were reachable at run time, e.g. SIGNOR timing out.

---

## 5. Logging results in this repo

**Folder naming:** `contributions/swechchha_biomni_<llm>_<condition>[_tools]/`, e.g. `swechchha_biomni_sonnet_naive`, `swechchha_biomni_sonnet_pipeline_guided_tools`.

**What each run writes** (this matches the README "What to upload" contract):

```
contributions/swechchha_biomni_sonnet_naive/
├── agent_runner.py          # copied in automatically (provenance)
├── prompt.txt               # exact prompt sent (condition + harness note [+ feedback])
├── atlas.json
├── run_log.json             # token_usage + Biomni/benchmark commits + audit + failures
├── run.log                  # full Biomni trace (code + observations)
├── scores/{summary,per_kinase,peptide_mismatches}.json
└── round2/, round3/, iterative_comparison.json   # only with --iterative
```

**Key `run_log.json` fields:** `model`, `framework`, `biomni_commit`, `benchmark_commit`, `condition`, `mode`, `stop_reason`, `atlas_source`, `token_usage{total_*, by_caller, per_call, estimated_cost_usd}`, `tool_calls`, `agent_turns`, `function_calls`, `http_failures`, `leakage_audit`, `elapsed_seconds`.

**Report caveat.** `evaluation/report.py` reads model and condition labels from `MODEL_META`. For a folder it doesn't recognise, it uses the folder name as the model and labels the condition **`naive`**. Your `swechchha_claudecode_*` folders already have this problem. The repo README says not to edit files outside your folder, so either ask Hui Lin to add rows like these, or flag the change clearly in your PR:

```python
"swechchha_biomni_sonnet_naive":           ("Biomni A1 (Sonnet 4.6)", "Agent framework", "naive"),
"swechchha_biomni_sonnet_paper_informed":  ("Biomni A1 (Sonnet 4.6)", "Agent framework", "paper_informed"),
"swechchha_biomni_sonnet_pipeline_guided": ("Biomni A1 (Sonnet 4.6)", "Agent framework", "pipeline_guided"),
```

Then regenerate the report with `python -m evaluation.report`. Do not commit `.env`, `databases/*` or the temp workdir.

---

## 6. Interpreting Biomni runs (what to look at)

- **`stop_reason`:**
  - `solution`: Biomni finished on its own.
  - `max_turns` / `timeout`: the budget ran out.
  - `error: ...`: API errors or `GraphRecursionError`.
- **`atlas_source`:**
  - `submit_atlas`: the normal case.
  - `recovered_from_namespace`: the agent never submitted. Report it, but flag it.
  - `none`: nothing was produced. Still commit the folder; failed runs are data.
- **`by_caller.biomni_internal`:** non-zero means Biomni's own tools (e.g. `query_uniprot`) made extra LLM calls.
- **Precision vs. the gold standard:** pulling PSP + UniProt in full gives ~0.8 precision and ~0.88 recall (see `claude_sonnet_pipeline_guided`). Much lower precision usually means extra sources or relaxed filters; much lower recall usually means a source failed (check `http_failures`).

---

## 7. What has been verified and what hasn't

- **Verified in the sandbox (Python 3.10):**
  - The `--dry-run` path works in both modes, and with `--iterative`. Round prompts build up the feedback from earlier rounds correctly.
  - The runner's `scores/summary.json` is identical to running `python -m evaluation.scorer` directly on the same atlas.
  - The leakage audit catches gold-standard paths.
  - Using mock `biomni` and `langchain_core` modules, I also tested the main code path: the runner changes directory so the benchmark `.env` isn't loaded, token counts are split into agent vs. internal calls, the `max_turns` stop works, a missing submission is recovered, and failed HTTP requests and code errors are logged.
- **Not verified:** a real Biomni + LangChain + Anthropic run. The sandbox has no Python 3.11 and no access to install packages. So the smoke test (step 1) is the first real check. If `token_usage` comes back as zeros there, your installed LangChain version stores usage somewhere else; `make_usage_handler` already falls back to `llm_output`.
- The wall-clock limit is only checked between agent steps, so one long code block can overrun it by up to `--exec-timeout` (600 s).
