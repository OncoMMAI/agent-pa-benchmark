#!/usr/bin/env python3
"""
Build the three local database files that databases/tools.py (DatabaseTools) reads.

PhosphoSitePlus bulk download is not available to our account (login works, but
the account is PSP_WEB_USER only and /downloads/ loops back to login). PSP records
are therefore taken from OmniPath's enz_sub table, keeping ONLY records whose
`sources` include a PhosphoSite token (PhosphoSite, PhosphoSite_MIMP,
PhosphoSite_ProtMapper). OmniPath records from other resources are ignored.

Inputs (raw_api_data/ is gitignored; Exploration/raw_data is outside the repo):
  raw_api_data/omnipath_enz_sub_human_phospho.tsv   omnipathdb.org/enz_sub (9606, phosphorylation)
  raw_api_data/uniprot_human_reviewed.fasta         rest.uniprot.org (9606, reviewed)
  ../../../raw_data/signor/human.tsv                SIGNOR getHumanData (no header, 29 cols)
  ../../../raw_data/uniprot/mod.tsv                 UniProt Entry / Gene / Modified residue / Sequence

Outputs (databases/* is gitignored):
  databases/psp/Kinase_Substrate_Dataset
  databases/signor/signor_phospho_human.json
  databases/uniprot/uniprot_phospho_parsed.json
  build_log.json, build_flagged_records.tsv   (in this folder)

Names are written as the source gives them (no HGNC harmonization here); that is
the pipeline's job (Curation 1/2 in run_pipeline.py).
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RAW = HERE / "raw_api_data"
EXT = REPO.parent / "raw_data"
DB = REPO / "databases"

# UniProt evidence codes that are NOT experimental (similarity / prediction).
NON_EXPERIMENTAL_ECO = {"ECO:0000250", "ECO:0000255", "ECO:0000256", "ECO:0000259",
                        "ECO:0000312", "ECO:0000313", "ECO:0000322", "ECO:0007829"}
PHOSPHO_NOTE = {"Phosphoserine": "S", "Phosphothreonine": "T", "Phosphotyrosine": "Y",
                "Phosphohistidine": "H"}
THREE = {"Ser": "S", "Thr": "T", "Tyr": "Y", "His": "H"}

flagged = []
log = {"inputs": {}, "psp": Counter(), "signor": Counter(), "uniprot": Counter()}


def load_fasta(path):
    seqs, acc, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if acc:
                seqs[acc] = "".join(buf)
            acc = line.split("|")[1]
            buf = []
        else:
            buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    return seqs


def window15(seq, residue, pos):
    """PSP-style SITE_+/-7_AA: 15-mer centred on the site, central residue lowercase,
    '_' padding at termini. Returns '' if the residue does not match the sequence."""
    if not seq or pos < 1 or pos > len(seq) or seq[pos - 1] != residue:
        return ""
    left = seq[max(0, pos - 8):pos - 1].rjust(7, "_")
    right = seq[pos:pos + 7].ljust(7, "_")
    return left + residue.lower() + right


def build_psp(seqs):
    out_rows = []
    path = RAW / "omnipath_enz_sub_human_phospho.tsv"
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            log["psp"]["omnipath_rows"] += 1
            tokens = row["sources"].split(";")
            psp_tokens = [t for t in tokens if t.startswith("PhosphoSite")]
            if not psp_tokens:
                continue
            log["psp"]["phosphosite_rows"] += 1
            if row["enzyme"].startswith("COMPLEX:") or row["substrate"].startswith("COMPLEX:"):
                log["psp"]["dropped_complex"] += 1
                flagged.append(("PhosphoSitePlus(OmniPath)", row["enzyme_genesymbol"],
                                row["substrate_genesymbol"], row["residue_type"] + row["residue_offset"],
                                "complex enzyme/substrate"))
                continue
            res, pos = row["residue_type"], int(row["residue_offset"])
            pep = window15(seqs.get(row["substrate"], ""), res, pos)
            log["psp"]["with_peptide" if pep else "no_peptide"] += 1
            out_rows.append({
                "GENE": row["enzyme_genesymbol"], "KINASE": row["enzyme_genesymbol"],
                "KIN_ACC_ID": row["enzyme"], "KIN_ORGANISM": "human",
                "SUBSTRATE": row["substrate_genesymbol"], "SUB_GENE_ID": "",
                "SUB_ACC_ID": row["substrate"], "SUB_GENE": row["substrate_genesymbol"],
                "SUB_ORGANISM": "human", "SUB_MOD_RSD": f"{res}{pos}", "SITE_GRP_ID": "",
                "SITE_+/-7_AA": pep, "DOMAIN": "", "IN_VIVO_RXN": "", "IN_VITRO_RXN": "",
                "CST_CAT#": "",
            })
    cols = list(out_rows[0].keys())
    (DB / "psp").mkdir(parents=True, exist_ok=True)
    with open(DB / "psp" / "Kinase_Substrate_Dataset", "w") as f:
        f.write("# Reconstructed PhosphoSitePlus Kinase_Substrate_Dataset (human)\n")
        f.write("# Source: OmniPath enz_sub records with PhosphoSite* source tokens\n")
        f.write("# SITE_+/-7_AA derived from UniProt canonical sequence at residue_offset\n")
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    log["psp"]["written"] = len(out_rows)


def build_signor():
    out = []
    with open(EXT / "signor" / "human.tsv", errors="replace") as f:
        for line in f:
            c = line.rstrip("\n").replace("\x00", "").split("\t")
            if len(c) < 22:
                log["signor"]["malformed"] += 1
                continue
            mech = c[9].strip().lower()
            if mech != "phosphorylation":       # excludes dephosphorylation etc.
                continue
            log["signor"]["phosphorylation_rows"] += 1
            # All SIGNOR getHumanData entities are human proteins; TAX_ID is the
            # experimental system (-1 = unspecified/in vitro), so it is not filtered.
            log["signor"]["experimental_taxid_non9606"] += c[12].strip() != "9606"
            site = c[10].strip()
            if not site:
                log["signor"]["no_residue"] += 1
                flagged.append(("SIGNOR", c[0], c[4], "", "no site"))
                continue
            if c[1] != "protein" or c[5] != "protein":
                log["signor"]["non_protein_entity"] += 1
                flagged.append(("SIGNOR", c[0], c[4], site, f"non-protein entity ({c[1]}->{c[5]})"))
                continue
            out.append({
                "kinase_gene": c[0].strip(), "substrate_gene": c[4].strip(),
                "phospho_site": site, "heptameric_peptide": c[11].strip(),
                "mechanism": c[9].strip(), "pubmed_id": c[21].strip(),
                "kinase_uniprot": c[2].strip(), "substrate_uniprot": c[6].strip(),
            })
    (DB / "signor").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(DB / "signor" / "signor_phospho_human.json", "w"))
    log["signor"]["written"] = len(out)


MOD_RE = re.compile(r'MOD_RES (\d+); /note="([^"]*)"(?:; /evidence="([^"]*)")?')


def split_kinases(text):
    text = text.split(";")[0].strip().rstrip(".")
    parts = re.split(r",\s*|\s+and\s+|\s+or\s+", text)
    return [p.strip() for p in parts if p.strip()]


def build_uniprot():
    out = []
    with open(EXT / "uniprot" / "mod.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            acc = row["Entry"]
            gene = (row.get("Gene Names (primary)") or "").split(";")[0].strip()
            for m in MOD_RE.finditer(row.get("Modified residue") or ""):
                pos, note, ev = int(m.group(1)), m.group(2), m.group(3) or ""
                kind = note.split(";")[0].strip()
                if kind not in PHOSPHO_NOTE or "; by " not in note:
                    continue
                log["uniprot"]["kinase_attributed_sites"] += 1
                codes = {e.split("|")[0].strip() for e in ev.split(",") if e.strip()}
                if codes and codes <= NON_EXPERIMENTAL_ECO:
                    log["uniprot"]["excluded_non_experimental"] += 1
                    flagged.append(("UniProt", note.split("; by ")[1], gene,
                                    f"{PHOSPHO_NOTE[kind]}{pos}", "non-experimental evidence (" + ",".join(sorted(codes)) + ")"))
                    continue
                if not gene:
                    log["uniprot"]["no_gene"] += 1
                    continue
                for kin in split_kinases(note.split("; by ", 1)[1]):
                    out.append({"kinase_gene": kin, "substrate_gene": gene,
                                "substrate_uniprot": acc, "phospho_site": f"{PHOSPHO_NOTE[kind]}{pos}"})
    (DB / "uniprot").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(DB / "uniprot" / "uniprot_phospho_parsed.json", "w"))
    log["uniprot"]["written"] = len(out)


def main():
    seqs = load_fasta(RAW / "uniprot_human_reviewed.fasta")
    log["inputs"]["uniprot_sequences"] = len(seqs)
    build_psp(seqs)
    build_signor()
    build_uniprot()
    with open(HERE / "build_flagged_records.tsv", "w") as f:
        f.write("source\tkinase\tsubstrate\tsite\treason\n")
        for r in flagged:
            f.write("\t".join(map(str, r)) + "\n")
    out = {k: dict(v) if isinstance(v, Counter) else v for k, v in log.items()}
    out["flagged_total"] = len(flagged)
    json.dump(out, open(HERE / "build_log.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
