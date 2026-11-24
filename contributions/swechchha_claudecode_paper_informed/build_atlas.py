import sys, re, csv, json, collections
D = sys.argv[1]; OUT = sys.argv[2]
csv.field_size_limit(10**9)

# ---- HGNC ----
sym, up2sym, alias = set(), {}, collections.defaultdict(set)
with open(f"{D}/hgnc/hgnc.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["status"] != "Approved": continue
        s = r["symbol"]; sym.add(s)
        for u in filter(None, r["uniprot_ids"].split("|")): up2sym.setdefault(u, s)
        for a in filter(None, r["alias_symbol"].split("|") + r["prev_symbol"].split("|")):
            alias[a.upper()].add(s)
symU = {s.upper(): s for s in sym}

# family / ambiguous names that must not be collapsed onto one gene via aliases
FAMILY = {"AMPK","CK1","CK2","CAMK2","PKC","PKA","PDGFR","PHK","PI3K","PKD","PKG","PKB","CDK11","MAPK","GSK3",
          "ABL","AKT","ERK","ERK1/2","JNK","P38","RSK","S6K","PIM","PAK","SGK","IKK","CAMK","SRC-TYPE TYR-KINASES"}
def hgnc(name):
    """Return HGNC symbol for a name, or None if not uniquely resolvable."""
    n = name.strip()
    cands = [n, re.sub(r"-alpha$", "A", n, flags=re.I), re.sub(r"-beta$", "B", n, flags=re.I),
             re.sub(r"-gamma$", "G", n, flags=re.I), re.sub(r"-delta$", "D", n, flags=re.I),
             n.replace("-", "")]
    for c in cands:
        if c.upper() in symU: return symU[c.upper()]
    if n.upper() in FAMILY: return None
    for c in cands:
        a = alias.get(c.upper())
        if a and len(a) == 1: return next(iter(a))
    return None

def resolve_kinase(name):
    if " or " in name: return None  # kinase identity uncertain
    if name.upper() in FAMILY: return None
    if name.upper() in symU: return symU[name.upper()]
    parts = name.split("/")
    for p in parts:  # exact HGNC symbol first, left to right: STK4/MST1 -> STK4, PKB/AKT1 -> AKT1
        if p.strip().upper() in symU and p.strip().upper() not in FAMILY: return symU[p.strip().upper()]
    for p in [name] + parts:
        s = hgnc(p)
        if s: return s
    return None

def base_acc(a):
    return re.split(r"[-_]", a.strip())[0]

def pep(seq, pos):
    i = pos - 1
    left = seq[max(0, i-7):i].rjust(7, "_"); right = seq[i+1:i+8].ljust(7, "_")
    return left + seq[i].lower() + right

# ---- UniProt ----
RES = {"Phosphoserine": "S", "Phosphothreonine": "T", "Phosphotyrosine": "Y", "Phosphohistidine": "H",
       "Pros-phosphohistidine": "H", "Tele-phosphohistidine": "H", "Phosphoarginine": "R",
       "Phosphoaspartate": "D", "Phospholysine": "K", "Phosphocysteine": "C"}
seqs, acc_gene = {}, {}
atlas = {}  # (kin, sub, site) -> dict
unresolved = collections.Counter(); notes = collections.Counter()
def add(kin, sub, acc, site, peptide, src, raw_kin):
    k = (kin, sub, site)
    e = atlas.setdefault(k, {"kinase": kin, "substrate": sub, "site": site, "peptide": "", "substrate_uniprot": acc,
                             "sources": set(), "raw_kinase_names": set()})
    e["sources"].add(src); e["raw_kinase_names"].add(raw_kin)
    if peptide and not e["peptide"]: e["peptide"] = peptide
    if acc and not e["substrate_uniprot"]: e["substrate_uniprot"] = acc

with open(f"{D}/uniprot/mod.tsv") as f:
    rd = csv.reader(f, delimiter="\t"); next(rd)
    rows = list(rd)
for acc, gene, ft, seq in rows:
    seqs[acc] = seq
    sub = up2sym.get(acc) or hgnc(gene.split(";")[0]) if gene else up2sym.get(acc)
    sub = sub or gene
    acc_gene[acc] = sub
    for pos, note in re.findall(r'MOD_RES (\d+); /note="([^"]*)"', ft):
        parts = [p.strip() for p in note.split(";")]
        if parts[0] not in RES: continue
        by = [p[3:] for p in parts[1:] if p.startswith("by ")]
        if not by: continue
        pos = int(pos); aa = RES[parts[0]]
        if pos > len(seq) or seq[pos-1] != aa: notes["uniprot_residue_mismatch"] += 1; continue
        site = f"{aa}{pos}"
        for name in re.split(r",\s*|\s+and\s+", by[0]):
            name = name.strip()
            if not name: continue
            if name.lower() == "autocatalysis": kin = sub
            elif name.lower() in ("host", "host kinase"): continue
            else: kin = resolve_kinase(name)
            if not kin: unresolved[("UniProt", name)] += 1; continue
            add(kin, sub, acc, site, pep(seq, pos), "UniProt", name)

# ---- SIGNOR ----
AA3 = {"Ser": "S", "Thr": "T", "Tyr": "Y", "His": "H", "Arg": "R", "Asp": "D", "Lys": "K", "Cys": "C"}
with open(f"{D}/signor/human.tsv") as f:
    for r in csv.reader(f, delimiter="\t"):
        if len(r) < 13 or r[9] != "phosphorylation" or r[12] != "9606": continue
        if r[1] != "protein": unresolved[("SIGNOR-non-protein-kinase:" + r[1], r[0])] += 1; continue
        if r[5] != "protein": notes["signor_nonprotein_substrate"] += 1; continue
        if not r[10].strip(): notes["signor_no_site"] += 1; continue
        ka, sa = base_acc(r[2]), base_acc(r[6])
        kin = up2sym.get(ka) or hgnc(r[0]); sub = up2sym.get(sa) or hgnc(r[4]) or r[4]
        if not kin: unresolved[("SIGNOR", r[0])] += 1; continue
        seqs_s = [s.strip() for s in r[11].split(";")]
        for i, res in enumerate(x.strip() for x in r[10].split(";")):
            m = re.match(r"([A-Z][a-z]{2})\s*(\d+)$", res)
            if not m or m.group(1) not in AA3: notes["signor_unparsed_site"] += 1; continue
            aa, pos = AA3[m.group(1)], int(m.group(2))
            site = f"{aa}{pos}"
            p = ""
            if sa in seqs and pos <= len(seqs[sa]) and seqs[sa][pos-1] == aa: p = pep(seqs[sa], pos)
            elif i < len(seqs_s) and len(seqs_s[i]) == 15: p = seqs_s[i][:7].upper() + seqs_s[i][7].lower() + seqs_s[i][8:].upper()
            add(kin, sub, sa, site, p, "SIGNOR", r[0])

# ---- output ----
rows = sorted(atlas.values(), key=lambda e: (e["kinase"], e["substrate"], int(e["site"][1:]), e["site"]))
with open(f"{OUT}/phospho_atlas.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["kinase", "substrate", "site", "peptide_15mer", "substrate_uniprot", "sources", "n_sources", "raw_kinase_names"])
    for e in rows:
        w.writerow([e["kinase"], e["substrate"], e["site"], e["peptide"], e["substrate_uniprot"],
                    ";".join(sorted(e["sources"])), len(e["sources"]), ";".join(sorted(e["raw_kinase_names"]))])
with open(f"{OUT}/unresolved_kinase_names.tsv", "w") as f:
    f.write("source\tname\tcount\n")
    for (s, n), c in unresolved.most_common(): f.write(f"{s}\t{n}\t{c}\n")
src = collections.Counter(";".join(sorted(e["sources"])) for e in rows)
print("triplets", len(rows), "kinases", len({e["kinase"] for e in rows}), "substrates", len({e["substrate"] for e in rows}))
print("by source", dict(src)); print("notes", dict(notes))
print("unresolved names", len(unresolved), "rows", sum(unresolved.values()))
print("with peptide", sum(1 for e in rows if e["peptide"]))
