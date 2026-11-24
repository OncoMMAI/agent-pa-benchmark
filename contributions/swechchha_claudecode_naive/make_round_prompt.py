#!/usr/bin/env python3
"""
Build the system prompt for round N of the iterative condition.

Faithfully mirrors agents/live_runner.py::run_iterative — each round is a
FRESH session whose system prompt is the base prompt plus the accumulated
feedback blocks from every prior round. It is NOT a continued conversation.

Usage:
  # Round 2 — feedback from round 1 only
  python3 contributions/swechchha_claudecode_naive/make_round_prompt.py \
      --round 2 \
      --scores contributions/swechchha_claudecode_naive/scores/summary.json \
      --out    contributions/swechchha_claudecode_naive/round2_prompt.txt

  # Round 3 — feedback from rounds 1 and 2, in order
  python3 contributions/swechchha_claudecode_naive/make_round_prompt.py \
      --round 3 \
      --scores contributions/swechchha_claudecode_naive/scores/summary.json \
               contributions/swechchha_claudecode_naive/round2/scores/summary.json \
      --out    contributions/swechchha_claudecode_naive/round3_prompt.txt
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.live_runner import generate_feedback  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True,
                    help="Round number being prepared (2 or 3)")
    ap.add_argument("--scores", nargs="+", required=True,
                    help="summary.json for each PRIOR round, in round order")
    ap.add_argument("--base-prompt", default="agents/prompts/naive.txt",
                    help="Base condition prompt (default: naive)")
    ap.add_argument("--out", required=True, help="Where to write the prompt")
    args = ap.parse_args()

    if len(args.scores) != args.round - 1:
        sys.exit(f"Round {args.round} needs {args.round - 1} prior "
                 f"score file(s); got {len(args.scores)}.")

    base = (PROJECT_ROOT / args.base_prompt).read_text().strip()

    blocks = []
    for i, path in enumerate(args.scores, start=1):
        blocks.append(generate_feedback(json.load(open(path)), i))

    header = (
        f"You are on Round {args.round} of an iterative curation process.\n"
        f"Below is feedback from your previous round(s). Use it to improve.\n"
    )
    footer = "\nFocus on finding the entries you missed while maintaining accuracy."

    prompt = base + "\n\n" + header + "\n\n".join(blocks) + footer

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(prompt)
    print(f"Wrote {out}  ({len(prompt):,} chars, {len(blocks)} feedback block(s))")


if __name__ == "__main__":
    main()
