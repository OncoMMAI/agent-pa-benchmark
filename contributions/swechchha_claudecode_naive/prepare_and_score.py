#!/usr/bin/env python3
"""
Normalize an agent's raw atlas.json field names and score it.

Shared helper for the swechchha_claudecode_* runs. Does three things:
  1. Unwraps the atlas (bare array, or dict under "atlas"/"entries"/first list)
  2. Renames known field-name variants to the benchmark's schema
  3. Writes it into the contribution folder and runs evaluation.scorer

Values are never changed, added, dropped, or re-fetched — keys only.
Entry count is asserted equal before and after.

Usage:
  python3 contributions/swechchha_claudecode_naive/prepare_and_score.py \
      --raw  ~/pa-pipeline/atlas.json \
      --dest contributions/swechchha_claudecode_pipeline_guided

  # inspect field names without writing anything
  python3 contributions/swechchha_claudecode_naive/prepare_and_score.py \
      --raw ~/pa-pipeline/atlas.json --dest /dev/null --inspect-only
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIELD_MAP = {
    # kinase
    "kinase": "kinase_gene", "enzyme": "kinase_gene",
    "kinase_symbol": "kinase_gene", "kinase_name": "kinase_gene",
    # substrate
    "substrate": "substrate_gene", "target": "substrate_gene",
    "substrate_symbol": "substrate_gene", "substrate_name": "substrate_gene",
    # site
    "site": "phospho_site", "residue": "phospho_site",
    "phosphosite": "phospho_site", "position": "phospho_site",
    # peptide
    "peptide": "heptameric_peptide", "hps": "heptameric_peptide",
    "sequence": "heptameric_peptide",
    # uniprot
    "uniprot": "substrate_uniprot", "sub_acc_id": "substrate_uniprot",
    # provenance
    "sources": "supporting_databases", "databases": "supporting_databases",
    "source": "supporting_databases",
}

SCHEMA = ["kinase_gene", "substrate_gene", "phospho_site",
          "substrate_uniprot", "heptameric_peptide", "supporting_databases"]


def unwrap(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("atlas", "entries", "data", "results"):
            if isinstance(data.get(k), list):
                return data[k]
        for v in data.values():
            if isinstance(v, list):
                return v
    raise SystemExit("Could not find a list of entries in the atlas file.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Agent's raw atlas.json")
    ap.add_argument("--dest", required=True,
                    help="Contribution folder (atlas.json + scores/ go here)")
    ap.add_argument("--gold",
                    default="gold_standard/parsed/phosphoatlas_gold.json")
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    raw_path = Path(args.raw).expanduser()
    entries = unwrap(json.load(open(raw_path)))

    if not entries:
        print("Atlas is EMPTY — 0 entries. Score it as an empty atlas and "
              "record the refusal in run_log.json.")
        return

    found = sorted({k for e in entries for k in e})
    print(f"raw file   : {raw_path}")
    print(f"entries    : {len(entries):,}")
    print(f"fields     : {found}")

    renames = {k: FIELD_MAP[k] for k in found if k in FIELD_MAP}
    unknown = [k for k in found if k not in FIELD_MAP and k not in SCHEMA]
    print(f"renames    : {renames or 'none needed'}")
    if unknown:
        print(f"unmapped   : {unknown}  (passed through unchanged)")

    if args.inspect_only:
        return

    out = [{FIELD_MAP.get(k, k): v for k, v in e.items()} for e in entries]
    assert len(out) == len(entries), "entry count changed — aborting"

    missing = [f for f in SCHEMA[:3] if f not in out[0]]
    if missing:
        print(f"\nWARNING: required field(s) still absent: {missing}")
        print("The scorer will report FP=1 and all-zero metrics.")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    atlas_path = dest / "atlas.json"
    json.dump(out, open(atlas_path, "w"), indent=2)
    print(f"\nwrote      : {atlas_path}  ({len(out):,} entries)")

    subprocess.run([
        sys.executable, "-m", "evaluation.scorer",
        "--atlas", str(atlas_path),
        "--gold", str(PROJECT_ROOT / args.gold),
        "--output", str(dest / "scores"),
    ], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
