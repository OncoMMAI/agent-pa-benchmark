#!/usr/bin/env python3
"""Build round-3 phosphorylation atlas.

Sources:
  OmniPath enz_sub (human, phosphorylation) restricted to rows supported by a
    manually curated resource: PhosphoSite(Plus), SIGNOR, HPRD, Phospho.ELM, dbPTM.
    Rows supported only by KEA, PhosphoNetworks, Li2012 or text mining
    (ProtMapper/REACH/Sparser/RLIMS-P/BEL) are excluded (main source of
    round-1 false positives).
  SIGNOR native download (human) — protein-level kinases + BCR-ABL fusion.
  UniProt Swiss-Prot MOD_RES "Phospho...; by KINASE" annotations, where the
    kinase resolves to a human gene symbol (or autocatalysis).
Peptides: 15-mer (site +/-7, site lowercase) from UniProt canonical sequence,
  falling back to SIGNOR's peptide.
"""
import csv, json, re, sys, collections
from pathlib import Path

dl = Path(sys.argv[1]); out = Path(sys.argv[2])
csv.field_size_limit(10**9)
CURATED = {'PhosphoSite', 'SIGNOR', 'HPRD', 'phosphoELM', 'dbPTM'}
AA3 = {'Ser': 'S', 'Thr': 'T', 'Tyr': 'Y', 'His': 'H'}

# ---- UniProt ----
seq_by_acc, acc_by_gene, gene_by_acc, uni_feats = {}, {}, {}, []
with open(dl / 'uniprot_human.tsv') as f:
    rd = csv.reader(f, delimiter='\t'); next(rd)
    for acc, gene, seq, modres in rd:
        gene = gene.split(';')[0].strip()
        seq_by_acc[acc] = seq; gene_by_acc[acc] = gene
        if gene and gene not in acc_by_gene: acc_by_gene[gene] = acc
        for m in re.finditer(r'MOD_RES (\d+); /note="(Phospho\w+)(?:; ([^"]*))?"', modres):
            pos, note, extra = int(m.group(1)), m.group(2), m.group(3) or ''
            bym = re.search(r'\bby ([^;]+)', extra)
            if bym: uni_feats.append((acc, gene, pos, bym.group(1)))

def peptide(acc, pos, res):
    s = seq_by_acc.get(acc)
    if not s or pos < 1 or pos > len(s) or s[pos - 1] != res: return ''
    left = s[max(0, pos - 8):pos - 1].rjust(7, '_')
    right = s[pos:pos + 7].ljust(7, '_')
    return left + res.lower() + right

atlas = {}
def add(kin, sub, site, src, sub_acc='', pep='', kname='', sname=''):
    key = (kin, sub, site)
    e = atlas.setdefault(key, dict(kinase_gene=kin, substrate_gene=sub, phospho_site=site,
                                   heptameric_peptide='', substrate_uniprot='',
                                   supporting_databases=set(), reported_kinase_names=set()))
    e['supporting_databases'].add(src)
    if kname: e['reported_kinase_names'].add(f'{src}:{kname}')
    if not e['substrate_uniprot'] and sub_acc: e['substrate_uniprot'] = sub_acc
    if not e['heptameric_peptide']:
        p = peptide(e['substrate_uniprot'], int(site[1:]), site[0]) if e['substrate_uniprot'] else ''
        e['heptameric_peptide'] = p or pep

# ---- OmniPath ----
srcname = {'PhosphoSite': 'PhosphoSitePlus', 'phosphoELM': 'Phospho.ELM'}
with open(dl / 'omni_enzsub.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        base = {x.split('_')[0] for x in r['sources'].split(';')}
        cur = base & CURATED
        if not cur or not r['enzyme_genesymbol'] or not r['substrate_genesymbol']: continue
        site = r['residue_type'] + r['residue_offset']
        for s in cur:
            add(r['enzyme_genesymbol'], r['substrate_genesymbol'], site, srcname.get(s, s),
                sub_acc=r['substrate'], kname=r['enzyme_genesymbol'])

# ---- SIGNOR native ----
with open(dl / 'signor.tsv') as f:
    for r in csv.reader(f, delimiter='\t'):
        if len(r) < 12 or r[9] != 'phosphorylation' or r[7] != 'UNIPROT' or r[5] != 'protein': continue
        m = re.fullmatch(r'([A-Z][a-z]{2})(\d+)', r[10].strip())
        if not m or m.group(1) not in AA3: continue
        site = AA3[m.group(1)] + m.group(2)
        sub_acc = r[6].split('-')[0]
        sub = gene_by_acc.get(sub_acc) or r[4]
        if r[1] == 'protein' and r[3] == 'UNIPROT':
            kin = gene_by_acc.get(r[2].split('-')[0]) or r[0]
        elif r[1] == 'fusion protein' and r[0] == 'BCR-ABL':
            kin = 'BCR/ABL'
        else:
            continue  # families/complexes: OmniPath already expands these to members
        add(kin, sub, site, 'SIGNOR', sub_acc=sub_acc, pep=r[11].strip(), kname=r[0])

# ---- UniProt ----
uni_stats = collections.Counter()
for acc, gene, pos, by in uni_feats:
    res = seq_by_acc[acc][pos - 1] if pos <= len(seq_by_acc[acc]) else ''
    if res not in 'STYH' or not gene: continue
    for tok in re.split(r',| and ', by):
        tok = tok.strip().rstrip('.').strip()
        tok = re.sub(r'\s*\(.*?\)\s*', '', tok).strip()
        if tok.lower() in ('autocatalysis', 'autophosphorylation'):
            kin = gene; uni_stats['auto'] += 1
        elif tok in acc_by_gene:
            kin = tok; uni_stats['resolved'] += 1
        else:
            uni_stats['unresolved'] += 1; continue
        add(kin, gene, res + str(pos), 'UniProt', sub_acc=acc, kname=tok)

rows = []
for e in sorted(atlas.values(), key=lambda e: (e['kinase_gene'], e['substrate_gene'], e['phospho_site'][0], int(e['phospho_site'][1:]))):
    e['supporting_databases'] = sorted(e['supporting_databases'])
    e['reported_kinase_names'] = '; '.join(sorted(e['reported_kinase_names']))
    rows.append(e)

out.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / 'phospho_atlas.json', 'w'), indent=1)
cols = ['kinase_gene', 'substrate_gene', 'phospho_site', 'heptameric_peptide', 'substrate_uniprot', 'supporting_databases', 'reported_kinase_names']
with open(out / 'phospho_atlas.tsv', 'w') as f:
    f.write('\t'.join(cols) + '\n')
    for e in rows:
        f.write('\t'.join(';'.join(e[c]) if isinstance(e[c], list) else e[c] for c in cols) + '\n')
summ = dict(total_triplets=len(rows),
            by_source=dict(collections.Counter(';'.join(e['supporting_databases']) for e in rows).most_common()),
            per_database=dict(collections.Counter(d for e in rows for d in e['supporting_databases'])),
            multi_source=sum(len(e['supporting_databases']) > 1 for e in rows),
            unique_kinases=len({e['kinase_gene'] for e in rows}),
            unique_substrates=len({e['substrate_gene'] for e in rows}),
            with_peptide=sum(bool(e['heptameric_peptide']) for e in rows),
            with_uniprot=sum(bool(e['substrate_uniprot']) for e in rows),
            uniprot_parse=dict(uni_stats))
json.dump(summ, open(out / 'atlas_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1)[:3000])
