#!/usr/bin/env python3
"""
Pipeline-guided PhosphoAtlas curation (agents/prompts/pipeline_guided.txt), executed by
Claude Code through the benchmark's DatabaseTools interface (databases/tools.py).

All phosphorylation records come exclusively from DatabaseTools calls. Two reference
tables are used only for identifier harmonization / sequence context, never as a
source of kinase-substrate relationships:
  - HGNC complete set (Exploration/raw_data/hgnc/hgnc.tsv): symbol backbone
  - UniProt human reviewed canonical sequences: to fill missing 15-mer peptides

Run from repo root:
  python3 contributions/swechchha_claudecode_pipeline_guided_tools/build_dbs.py
  python3 contributions/swechchha_claudecode_pipeline_guided_tools/run_pipeline.py
"""
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from databases.tools import DatabaseTools                      # noqa: E402
from evaluation.normalizer import GENE_ALIASES, normalize_phospho_site  # noqa: E402
from agents.claude_code_runner import TokenEstimator           # noqa: E402

HGNC_PATH = REPO.parent / "raw_data" / "hgnc" / "hgnc.tsv"
FASTA_PATH = HERE / "raw_api_data" / "uniprot_human_reviewed.fasta"
PROMPT = (REPO / "agents" / "prompts" / "pipeline_guided.txt").read_text()
SITE_RE = re.compile(r"^[STYH]\d+$")
PAGE = 100

# UniProt free-text kinase names that HGNC cannot resolve but which name exactly one
# human gene (identifier harmonization only). Family names (PKC, CK1, CaMK2, MAPK,
# PKG, ...) are deliberately NOT mapped and are flagged as ambiguous.
UNIPROT_KINASE_NAMES = {
    "CDC2": "CDK1", "PKB/AKT1": "AKT1", "AKT1": "AKT1", "IKKE": "IKBKE",
    "IKKA": "CHUK", "IKKB": "IKBKB", "DNA-PK": "PRKDC", "DNA-PKCS": "PRKDC",
    "PLK": "PLK1", "MAP3K7": "MAP3K7", "TAK1": "MAP3K7", "TBK1": "TBK1",
    "GSK3-BETA": "GSK3B", "GSK3B": "GSK3B", "GSK3-ALPHA": "GSK3A",
    "ERK1": "MAPK3", "ERK2": "MAPK1", "JNK1": "MAPK8", "JNK2": "MAPK9", "JNK3": "MAPK10",
    "P38": "MAPK14", "P38A": "MAPK14", "PKR": "EIF2AK2", "PERK": "EIF2AK3",
    "CK2": "CSNK2A1", "PKA": "PRKACA", "ROCK": "ROCK1", "MTOR": "MTOR",
    "C-ABL": "ABL1", "ABL": "ABL1", "SRC": "SRC", "C-SRC": "SRC",
    "CHK1": "CHEK1", "CHK2": "CHEK2", "ATM": "ATM", "ATR": "ATR",
}


# ── Reference tables ─────────────────────────────────────────────────────────

class HGNC:
    def __init__(self, path):
        self.approved, self.uniprot_to_sym, self.sym_to_uniprot = set(), {}, {}
        prev, alias = defaultdict(set), defaultdict(set)
        for r in csv.DictReader(open(path), delimiter="\t"):
            if r["status"] != "Approved":
                continue
            sym = r["symbol"]
            self.approved.add(sym)
            ups = [u for u in r["uniprot_ids"].strip('"').split("|") if u]
            if ups:
                self.sym_to_uniprot[sym] = ups[0]
            for u in ups:
                self.uniprot_to_sym.setdefault(u, sym)
            for p in r["prev_symbol"].strip('"').split("|"):
                if p:
                    prev[p.upper()].add(sym)
            for a in r["alias_symbol"].strip('"').split("|"):
                if a:
                    alias[a.upper()].add(sym)
        self.approved_upper = {s.upper(): s for s in self.approved}
        self.prev = {k: next(iter(v)) for k, v in prev.items() if len(v) == 1}
        self.alias = {k: next(iter(v)) for k, v in alias.items() if len(v) == 1}
        self.alias_ambiguous = {k for k, v in alias.items() if len(v) > 1}

    def resolve(self, name, uniprot=""):
        """Return (hgnc_symbol or '', method)."""
        n = (name or "").strip()
        if not n:
            return "", "empty"
        if n in self.approved:
            return n, "approved"
        if uniprot:
            acc = uniprot.split("-")[0]
            if acc in self.uniprot_to_sym:
                return self.uniprot_to_sym[acc], "uniprot"
        u = n.upper()
        if u in self.approved_upper:
            return self.approved_upper[u], "approved_case"
        if n in GENE_ALIASES and GENE_ALIASES[n] in self.approved:
            return GENE_ALIASES[n], "benchmark_alias"
        if u in UNIPROT_KINASE_NAMES:
            return UNIPROT_KINASE_NAMES[u], "uniprot_name_map"
        if u in self.prev:
            return self.prev[u], "prev_symbol"
        if u in self.alias:
            return self.alias[u], "alias_symbol"
        if "/" in n:                         # e.g. "PKB/AKT1"
            for part in n.split("/"):
                s, m = self.resolve(part)
                if s:
                    return s, "slash_part:" + m
        return "", "ambiguous_alias" if u in self.alias_ambiguous else "unmapped"


def load_fasta(path):
    seqs, acc, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if acc:
                seqs[acc] = "".join(buf)
            acc, buf = line.split("|")[1], []
        else:
            buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    return seqs


def window15(seq, site):
    res, pos = site[0], int(site[1:])
    if not seq or pos > len(seq) or seq[pos - 1] != res:
        return ""
    return seq[max(0, pos - 8):pos - 1].rjust(7, "_") + res.lower() + seq[pos:pos + 7].ljust(7, "_")


def peptide_center_ok(pep, site):
    return len(pep) == 15 and pep[7].upper() == site[0]


# ── Pipeline ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    tools = DatabaseTools(str(REPO / "databases"))
    est = TokenEstimator(PROMPT, model="opus")
    log_lines = []

    def call(name, **kw):
        res = getattr(tools, name)(**kw)
        est.record_turn(name, len(json.dumps(kw)), len(json.dumps(res)))
        return res

    def log(phase, msg):
        line = f"[{time.time() - t0:7.1f}s] [{phase}] {msg}"
        log_lines.append(line)
        print(line, flush=True)

    # ── PHASE 1 / Step 1: discover databases ────────────────────────────────
    dbs = [d["id"] for d in call("list_databases")["databases"]]
    stats = {db: call("get_stats", db=db) for db in dbs}
    for db in dbs:
        log("STEP1", f"{db}: {stats[db]}")
    hgnc = HGNC(HGNC_PATH)
    seqs = load_fasta(FASTA_PATH)
    log("STEP1", f"HGNC approved symbols: {len(hgnc.approved)}; UniProt sequences: {len(seqs)}")

    # ── Step 2: paginate kinases and substrates in every database ───────────
    kin_lists, sub_lists = {}, {}
    for db in dbs:
        for kind, store, field in (("list_kinases", kin_lists, "kinases"),
                                   ("list_substrates", sub_lists, "substrates")):
            items, off = [], 0
            while True:
                r = call(kind, db=db, offset=off, limit=PAGE)
                items += r[field]
                off += PAGE
                if off >= r["total_" + field]:
                    break
            store[db] = items
        log("STEP2", f"{db}: {len(kin_lists[db])} kinases, {len(sub_lists[db])} substrates (paginated)")

    # Curation 1: map every listed name to HGNC; flag unmappable names
    name_map, name_flags = {}, []
    role_by_symbol = defaultdict(lambda: defaultdict(set))
    for db in dbs:
        for role, names in (("kinase", kin_lists[db]), ("substrate", sub_lists[db])):
            for n in names:
                sym, how = hgnc.resolve(n)
                name_map[n] = (sym, how)
                if sym:
                    role_by_symbol[sym][role].add(db)
                elif n.lower() != "autocatalysis":
                    name_flags.append((db, role, n, how))
    multi_db_genes = sum(1 for s, r in role_by_symbol.items()
                         if len(r["kinase"] | r["substrate"]) >= 2)
    dual_role = sum(1 for s, r in role_by_symbol.items() if r["kinase"] and r["substrate"])
    log("STEP2", f"symbols seen in 2+ dbs: {multi_db_genes}; kinase+substrate dual-role: {dual_role}; "
                 f"unmappable listed names: {len(name_flags)}")

    # ── PHASE 2 / Step 4: systematic extraction ─────────────────────────────
    raw = {}       # dedupe identical raw records reached via kinase and substrate paths
    reached = Counter()
    for db in dbs:
        for k in kin_lists[db]:
            for e in call("query_by_kinase", db=db, gene=k)["entries"]:
                key = (db, e["kinase_gene"], e["substrate_gene"], e["phospho_site"],
                       e.get("heptameric_peptide", ""), e.get("substrate_uniprot", ""))
                raw.setdefault(key, (db, e)); reached["by_kinase"] += 1
        n_before = len(raw)
        for s in sub_lists[db]:
            for e in call("query_by_substrate", db=db, gene=s)["entries"]:
                key = (db, e["kinase_gene"], e["substrate_gene"], e["phospho_site"],
                       e.get("heptameric_peptide", ""), e.get("substrate_uniprot", ""))
                raw.setdefault(key, (db, e)); reached["by_substrate"] += 1
        log("STEP4", f"{db}: {len(raw)} unique raw records so far "
                     f"(+{len(raw) - n_before} only via substrate path)")
    log("STEP4", f"records returned: {dict(reached)}; unique raw records: {len(raw)}")

    # ── Step 5 + Curation 2/3: harmonize, validate sites, flag ──────────────
    db_label = {"psp": "PhosphoSitePlus", "signor": "SIGNOR", "uniprot": "UniProt"}
    flagged, clean = [], []
    harm = Counter()
    for (db, e) in raw.values():
        kin_raw, sub_raw = e["kinase_gene"], e["substrate_gene"]
        sub_up = e.get("substrate_uniprot", "")
        sub, how_s = hgnc.resolve(sub_raw, sub_up)
        if kin_raw.lower() == "autocatalysis":
            kin, how_k = sub, "autocatalysis->substrate"
        else:
            kin, how_k = hgnc.resolve(kin_raw)
        harm["kinase:" + how_k.split(":")[0]] += 1
        harm["substrate:" + how_s.split(":")[0]] += 1
        site = normalize_phospho_site(e.get("phospho_site", "")).upper()
        reason = None
        if not kin:
            reason = f"kinase symbol not HGNC-mappable ({how_k})"
        elif not sub:
            reason = f"substrate symbol not HGNC-mappable ({how_s})"
        elif not SITE_RE.match(site):
            reason = "invalid/missing phospho-site"
        if reason:
            flagged.append((db_label[db], kin_raw, sub_raw, e.get("phospho_site", ""), reason))
            continue
        pep = (e.get("heptameric_peptide") or "").strip()
        if pep and not peptide_center_ok(pep, site):
            harm["peptide_center_mismatch_dropped"] += 1
            pep = ""
        clean.append({"db": db_label[db], "kinase": kin, "substrate": sub, "site": site,
                      "peptide": pep, "uniprot": sub_up.split("-")[0] if sub_up else "",
                      "kinase_raw": kin_raw, "substrate_raw": sub_raw})
    log("STEP5", f"validated records: {len(clean)}; flagged: {len(flagged)}")

    # ── Step 6/7: assemble atlas, merge identical triplets ──────────────────
    atlas = {}
    pep_conflicts = []
    for r in clean:
        key = (r["kinase"], r["substrate"], r["site"])
        a = atlas.setdefault(key, {"kinase_gene": r["kinase"], "substrate_gene": r["substrate"],
                                   "phospho_site": r["site"], "heptameric_peptide": "",
                                   "substrate_uniprot": "", "supporting_databases": [],
                                   "_peps": {}})
        if r["db"] not in a["supporting_databases"]:
            a["supporting_databases"].append(r["db"])
        if r["peptide"]:
            a["_peps"].setdefault(r["db"], r["peptide"])
        if not a["substrate_uniprot"] and r["uniprot"]:
            a["substrate_uniprot"] = r["uniprot"]

    # Step 7: cross-reference each kinase across all dbs with query_all_dbs
    kinase_symbols = sorted({k for k, _, _ in atlas})
    raw_names_by_symbol = defaultdict(set)
    for r in clean:
        raw_names_by_symbol[r["kinase"]].add(r["kinase_raw"])
    xref_new = 0
    for sym in kinase_symbols:
        for raw_name in sorted(raw_names_by_symbol[sym] | {sym}):
            res = call("query_all_dbs", gene=raw_name)
            for db, d in res["databases"].items():
                for e in d["kinase_entries"]:
                    sub, _ = hgnc.resolve(e["substrate_gene"], e.get("substrate_uniprot", ""))
                    site = normalize_phospho_site(e.get("phospho_site", "")).upper()
                    key = (sym, sub, site)
                    if key in atlas and db_label[db] not in atlas[key]["supporting_databases"]:
                        atlas[key]["supporting_databases"].append(db_label[db])
                        xref_new += 1
    log("STEP7", f"query_all_dbs over {len(kinase_symbols)} kinases; new db supports added: {xref_new}")

    # Peptide resolution + UniProt fill
    pep_source = Counter()
    for a in atlas.values():
        peps = a.pop("_peps")
        distinct = {p.upper() for p in peps.values()}
        if len(distinct) > 1:
            pep_conflicts.append({"triplet": f'{a["kinase_gene"]}|{a["substrate_gene"]}|{a["phospho_site"]}',
                                  "peptides": peps})
        for db in ("PhosphoSitePlus", "SIGNOR"):
            if db in peps:
                a["heptameric_peptide"] = peps[db]
                pep_source[db] += 1
                break
        if not a["substrate_uniprot"]:
            up = hgnc.sym_to_uniprot.get(a["substrate_gene"], "")
            if up:
                a["substrate_uniprot"] = up
                pep_source["uniprot_acc_from_hgnc"] += 1
        if not a["heptameric_peptide"] and a["substrate_uniprot"]:
            p = window15(seqs.get(a["substrate_uniprot"], ""), a["phospho_site"])
            if p:
                a["heptameric_peptide"] = p
                pep_source["derived_from_uniprot_sequence"] += 1
        order = ["PhosphoSitePlus", "SIGNOR", "UniProt"]
        a["supporting_databases"].sort(key=order.index)

    # ── Step 8: final QC ────────────────────────────────────────────────────
    final = [a for a in atlas.values()
             if a["kinase_gene"] in hgnc.approved and a["substrate_gene"] in hgnc.approved
             and SITE_RE.match(a["phospho_site"])]
    final.sort(key=lambda a: (a["kinase_gene"], a["substrate_gene"],
                              a["phospho_site"][0], int(a["phospho_site"][1:])))
    log("STEP8", f"final atlas entries: {len(final)}")

    # ── Outputs ─────────────────────────────────────────────────────────────
    json.dump(final, open(HERE / "atlas.json", "w"), indent=1)

    kin_idx, sub_idx, site_idx, hps_idx = defaultdict(lambda: [set(), 0, set()]), \
        defaultdict(lambda: [set(), set(), 0]), defaultdict(lambda: [set(), "", set()]), defaultdict(set)
    for a in final:
        k, s, p = a["kinase_gene"], a["substrate_gene"], a["phospho_site"]
        kin_idx[k][0].add(s); kin_idx[k][1] += 1; kin_idx[k][2].update(a["supporting_databases"])
        sub_idx[s][0].add(k); sub_idx[s][1].add(a["substrate_uniprot"]); sub_idx[s][2] += 1
        site_idx[(s, p)][0].add(k); site_idx[(s, p)][1] = a["heptameric_peptide"]
        site_idx[(s, p)][2].update(a["supporting_databases"])
        if a["heptameric_peptide"]:
            hps_idx[a["heptameric_peptide"].upper()].add(f"{s}:{p}")

    def write_tsv(name, header, rows):
        with open(HERE / name, "w") as f:
            f.write("\t".join(header) + "\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")

    write_tsv("kinase_index.tsv", ["kinase_gene", "n_substrates", "n_entries", "databases"],
              [(k, len(v[0]), v[1], ",".join(sorted(v[2]))) for k, v in sorted(kin_idx.items())])
    write_tsv("substrate_index.tsv", ["substrate_gene", "uniprot", "n_kinases", "n_entries"],
              [(s, ",".join(sorted(x for x in v[1] if x)), len(v[0]), v[2]) for s, v in sorted(sub_idx.items())])
    write_tsv("site_index.tsv", ["substrate_gene", "phospho_site", "heptameric_peptide", "kinases", "databases"],
              [(s, p, v[1], ",".join(sorted(v[0])), ",".join(sorted(v[2]))) for (s, p), v in sorted(site_idx.items())])
    write_tsv("hps_index.tsv", ["heptameric_peptide", "substrate_sites"],
              [(h, ",".join(sorted(v))) for h, v in sorted(hps_idx.items())])
    write_tsv("flagged_records.tsv", ["source", "kinase", "substrate", "site", "reason"],
              flagged + [(db, role, n, "", f"listed name not HGNC-mappable ({how})")
                         for db, role, n, how in name_flags])
    json.dump(pep_conflicts, open(HERE / "peptide_conflicts.json", "w"), indent=1)

    n_multi = sum(1 for a in final if len(a["supporting_databases"]) >= 2)
    per_db = Counter(d for a in final for d in a["supporting_databases"])
    qc = {
        "data_provenance": {
            "PhosphoSitePlus": "OmniPath enz_sub records with PhosphoSite* source tokens "
                               "(PSP bulk download not authorized for our account); peptides "
                               "from UniProt canonical sequence",
            "SIGNOR": "Exploration/raw_data/signor/human.tsv (getHumanData), mechanism == phosphorylation, protein->protein",
            "UniProt": "Exploration/raw_data/uniprot/mod.tsv, '; by <kinase>' phospho MOD_RES, non-experimental ECO excluded",
        },
        "db_stats": stats,
        "atlas_entries": len(final),
        "kinases": len(kin_idx),
        "substrates": len(sub_idx),
        "unique_substrate_sites": len(site_idx),
        "unique_peptides": len(hps_idx),
        "entries_with_peptide": sum(1 for a in final if a["heptameric_peptide"]),
        "entries_with_uniprot": sum(1 for a in final if a["substrate_uniprot"]),
        "entries_2plus_dbs": n_multi,
        "entries_3_dbs": sum(1 for a in final if len(a["supporting_databases"]) == 3),
        "entries_per_db": dict(per_db),
        "peptide_source": dict(pep_source),
        "peptide_conflicts_across_dbs": len(pep_conflicts),
        "raw_records": len(raw),
        "validated_records": len(clean),
        "flagged_records": len(flagged),
        "flag_reasons": dict(Counter(f[4].split(" (")[0] for f in flagged)),
        "unmappable_listed_names": len(name_flags),
        "harmonization_methods": dict(harm),
        "tool_calls": dict(Counter(c["tool"] for c in tools.call_log)),
    }
    json.dump(qc, open(HERE / "qc_report.json", "w"), indent=2)

    tok = est.summary()
    run_log = {
        "agent": "Claude Code (claude-opus-5-5) executing pipeline_guided via DatabaseTools",
        "condition": "pipeline_guided",
        "elapsed_seconds": round(time.time() - t0, 1),
        "total_tool_calls": tools.call_count,
        "tool_call_breakdown": qc["tool_calls"],
        "atlas_entries": len(final),
        "token_usage": {
            "total_input_tokens": tok["total_input_tokens"],
            "total_output_tokens": tok["total_output_tokens"],
            "total_tokens": tok["total_tokens"],
            "estimated_cost_usd": tok["estimated_cost_usd"],
            "api_calls": tok["estimated_api_turns"],
            "note": "Estimated with agents/claude_code_runner.TokenEstimator; the Claude Code "
                    "session's exact token counts are not exposed to the script.",
        },
        "log": log_lines,
    }
    json.dump(run_log, open(HERE / "run_log.json", "w"), indent=2)
    log("DONE", f"wrote atlas.json ({len(final)} entries), indexes, qc_report.json, run_log.json")


if __name__ == "__main__":
    main()
