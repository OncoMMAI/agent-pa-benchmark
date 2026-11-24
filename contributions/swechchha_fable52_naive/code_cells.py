# ── cell 1 ──
import os
print(os.getcwd())
print(os.listdir('.'))
print([k for k in globals().keys() if not k.startswith('_')])

# ── cell 2 ──
for q in ["human kinase substrate phosphosite database download", "PhosphoSitePlus Kinase_Substrate_Dataset download", "OmniPath enzsub kinase substrate API", "PhosphoELM download", "SIGNOR phosphorylation download"]:
    r = web_search(query=q)
    print(q, '->', str(r)[:1500])
    print()

# ── cell 3 ──
for q in ["phosphorylation", "kinase", "database", "PhosphoSitePlus", "OmniPath"]:
    r = web_search(query=q)
    print(q, '->', str(r)[:1500])
urls = ["https://omnipathdb.org/enz_sub?organisms=9606&types=phosphorylation&fields=sources,references&format=tsv&limit=5",
        "https://omnipathdb.org/enz_sub?genesymbols=1&types=phosphorylation&limit=5",
        "https://signor.uniroma2.it/getData.php?organism=9606",
        "https://www.phosphosite.org/downloads/Kinase_Substrate_Dataset.gz",
        "https://rest.uniprot.org/uniprotkb/search?query=P06493&format=tsv&size=1",
        "https://example.com"]
for u in urls:
    try:
        r = web_fetch(url=u)
        print(u, '->', str(r)[:1200])
    except Exception as e:
        print(u, 'ERR', e)
    print()

# ── cell 4 ──
r = fetch_and_parse_db(url="https://omnipathdb.org/enz_sub?organisms=9606&types=phosphorylation&genesymbols=1&fields=sources,references&format=tsv", source_name="OmniPath")
print(str(r)[:3000])

# ── cell 5 ──
print(os.listdir('.'))
r = fetch_and_parse_db(url="https://signor.uniroma2.it/getData.php?organism=9606", source_name="SIGNOR")
print(str(r)[:3000])
print(os.listdir('.'))
import inspect
for f in (fetch_and_parse_db, web_fetch, ingest_entries_file, submit_atlas):
    try:
        print(inspect.signature(f), f.__doc__)
    except Exception as e:
        print('ERR', e)

# ── cell 6 ──
r = web_fetch(url="https://omnipathdb.org/queries/enz_sub")
print(r['content'][:6000])
r = fetch_and_parse_db(url="https://omnipathdb.org/enz_sub?organisms=9606&types=phosphorylation&genesymbols=1&enzymes=P06239&format=json", source_name="OmniPath")
print(r)

# ── cell 7 ──
r = fetch_and_parse_db(url="https://omnipathdb.org/enz_sub?organisms=9606&types=phosphorylation&genesymbols=1&fields=sources&format=json", source_name="OmniPath")
print(r)
r2 = web_fetch(url="https://omnipathdb.org/enz_sub?organisms=9606&types=phosphorylation&genesymbols=1&fields=sources&format=json&limit=3")
print(r2['content'])

# ── cell 8 ──
import time
base="https://omnipathdb.org/enz_sub?organisms=9606&types=phosphorylation&genesymbols=1&fields=sources&format=json"
for extra in ["&enzymes=P06239","&enzymes=P06493","&residues=Y","&databases=SIGNOR","&limit=5000", "&limit=20000"]:
    t=time.time()
    r = fetch_and_parse_db(url=base+extra, source_name="OmniPath")
    print(extra, {k:v for k,v in r.items() if k!='url'}, round(time.time()-t,1))

# ── cell 9 ──
import json
test=[{"kinase_gene":"LCK","substrate_gene":"SOCS3","phospho_site":"Y204","substrate_uniprot":"O14543","supporting_databases":["KEA", "MIMP", "PhosphoNetworks", "PhosphoSite", "PhosphoSite_MIMP", "PhosphoSite_ProtMapper", "ProtMapper", "phosphoELM", "phosphoELM_MIMP"]}]
json.dump(test,open('t.json','w'))
print(ingest_entries_file(path='t.json', source_name='OmniPath'))
print(ingest_entries_file(path='t.json', source_name='OmniPath'))
r = fetch_and_parse_db(url=base+"&enzymes=P06239", source_name="OmniPath")
print(r)

# ── cell 10 ──
b2="https://omnipathdb.org/enz_sub?organisms=9606&types=phosphorylation&format=json"
def count(extra):
    r = fetch_and_parse_db(url=b2+extra, source_name="probe")
    return r['entries_parsed']
for extra in ["", "&limit=10000","&limit=15000","&limit=30000","&residues=S","&residues=T","&residues=Y","&residues=H", "&residues=S10"]:
    print(extra, count(extra))

# ── cell 11 ──
import urllib.parse
q=urllib.parse.quote("(keyword:KW-0418) AND (organism_id:9606) AND (reviewed:true)")
r=web_fetch(url=f"https://rest.uniprot.org/uniprotkb/stream?query={q}&fields=accession&format=tsv")
print(r['length'], r['truncated'])
kin=[x for x in r['content'].split('\n')[1:] if x.strip()]
print(len(kin), kin[:5])

# ── cell 12 ──
OB="https://omnipathdb.org/enz_sub?organisms=9606&types=phosphorylation&genesymbols=1&fields=sources&format=tsv"
rows={}   # key (enz,sub,restype,offset) -> dict
log=[]
def parse(content, truncated):
    lines=content.split('\n')
    if truncated: lines=lines[:-1]
    out=[]
    for ln in lines[1:]:
        p=ln.split('\t')
        if len(p)<8: continue
        out.append(p)
    return out
def fetch(extra, retries=3):
    for i in range(retries):
        try:
            r=web_fetch(url=OB+extra)
            if 'content' in r:
                return r
            err=r
        except Exception as e:
            err=e
        time.sleep(2)
    raise RuntimeError(str(err)[:300])
def add(ps):
    for p in ps:
        if p[6]!='phosphorylation': continue
        k=(p[0],p[2],p[4],p[5])
        rows[k]={'enzyme':p[0],'kinase_gene':p[1],'substrate':p[2],'substrate_gene':p[3],'res':p[4],'off':p[5],'sources':p[7].split(';')}
r=fetch("&enzymes=P06239")
print(r['length'], r['truncated'], r['content'][:300])
ps=parse(r['content'], r['truncated']); print(len(ps))

# ── cell 13 ──
MB="https://omnipathdb.org/enz_sub?organisms=9606&types=phosphorylation&format=tsv"
DBS="BEL-Large-Corpus_ProtMapper;DEPOD;HPRD;HPRD_MIMP;KEA;Li2012;MIMP;NCI-PID_ProtMapper;PhosphoNetworks;PhosphoSite;PhosphoSite_MIMP;PhosphoSite_ProtMapper;ProtMapper;REACH_ProtMapper;RLIMS-P_ProtMapper;Reactome_ProtMapper;SIGNOR;SIGNOR_ProtMapper;Sparser_ProtMapper;dbPTM;phosphoELM;phosphoELM_MIMP".split(';')
ncalls=0
def wf(url, retries=4):
    global ncalls
    err=None
    for i in range(retries):
        try:
            ncalls+=1
            r=web_fetch(url=url)
            if isinstance(r,dict) and 'content' in r and 'error' not in r:
                return r
            err=r
        except Exception as e:
            err=e
        time.sleep(2+2*i)
    raise RuntimeError(str(err)[:300])
def min_rows(extra):
    """returns (set of keys, complete?, full_length)"""
    r=wf(MB+extra)
    lines=r['content'].split('\n')
    if r['truncated']: lines=lines[:-1]
    ks=set()
    for ln in lines[1:]:
        p=ln.split('\t')
        if len(p)>=5 and p[4]=='phosphorylation': ks.add((p[0],p[1],p[2],p[3]))
    return ks, not r['truncated'], r['length']
def min_all(extra):
    """complete set of minimal keys for a filter, splitting as needed. returns (keys, ok)"""
    ks,ok,L=min_rows(extra)
    if ok: return ks,True
    HL=len("enzyme\tsubstrate\tresidue_type\tresidue_offset\tmodification\n")
    target=L-HL
    allk=set(ks); good=True
    for res in "STY":
        k2,ok2,L2=min_rows(extra+"&residues="+res)
        allk|=k2
        if not ok2:
            for db in DBS:
                k3,ok3,L3=min_rows(extra+"&residues="+res+"&databases="+db)
                allk|=k3
                if not ok3: good=False
    got=sum(len('\t'.join(k))+len('\tphosphorylation\n') for k in allk)
    # non-phospho rows shouldn't exist due to types filter
    if got!=target: good=False
    return allk, good
def full_fetch(extra):
    r=wf(OB+extra)
    ps=parse(r['content'], r['truncated'])
    add(ps)
    return not r['truncated'], len(ps)
problems=[]
done_enz={}
def do_enzyme(E):
    ok,n=full_fetch("&enzymes="+E)
    if ok:
        done_enz[E]=n; return
    ks,good=min_all("&enzymes="+E)
    if not good: problems.append(('min_incomplete',E))
    from collections import Counter
    c=Counter(k[1] for k in ks if k[0]==E)
    subs=sorted(c)
    # batch substrates
    batches=[];cur=[];cn=0
    for s in subs:
        if cur and cn+c[s]>90:
            batches.append(cur);cur=[];cn=0
        cur.append(s);cn+=c[s]
    if cur: batches.append(cur)
    stack=batches[::-1]
    while stack:
        b=stack.pop()
        ok,n=full_fetch("&enzymes="+E+"&substrates="+",".join(b))
        if not ok:
            if len(b)==1:
                # split single substrate by residue
                for res in "STY":
                    ok2,n2=full_fetch("&enzymes="+E+"&substrates="+b[0]+"&residues="+res)
                    if not ok2: problems.append(('single_trunc',E,b[0],res))
            else:
                h=len(b)//2
                stack.append(b[h:]);stack.append(b[:h])
    missing=[k for k in ks if k not in rows]
    if missing: problems.append(('missing_after',E,len(missing)))
    done_enz[E]=len(ks)
t=time.time()
for E in kin[:40]:
    if E in done_enz: continue
    do_enzyme(E)
print(time.time()-t, ncalls, len(rows), problems)

# ── cell 14 ──
for extra in ["&enzymes=O14757&substrates=O00716","&enzymes=O14757&substrates=O00716&enzyme_substrate=AND","&enzymes=O14757&substrates=O00716&residues=S&enzyme_substrate=AND"]:
    r=wf(OB+extra)
    print(extra, r['length'], r['truncated'])
    print(r['content'][:600])

# ── cell 15 ──
from collections import Counter
problems=[]
def do_enzyme(E):
    ok,n=full_fetch("&enzymes="+E)
    if ok:
        done_enz[E]=n; return
    ks,good=min_all("&enzymes="+E)
    if not good: problems.append(('min_incomplete',E))
    c=Counter(k[1] for k in ks if k[0]==E)
    subs=sorted(c)
    batches=[];cur=[];cn=0
    for s in subs:
        if cur and cn+c[s]>100:
            batches.append(cur);cur=[];cn=0
        cur.append(s);cn+=c[s]
    if cur: batches.append(cur)
    stack=batches[::-1]
    while stack:
        b=stack.pop()
        ok,n=full_fetch("&enzyme_substrate=AND&enzymes="+E+"&substrates="+",".join(b))
        if not ok:
            if len(b)==1:
                for res in "STY":
                    ok2,n2=full_fetch("&enzyme_substrate=AND&enzymes="+E+"&substrates="+b[0]+"&residues="+res)
                    if not ok2: problems.append(('single_trunc',E,b[0],res))
            else:
                h=len(b)//2
                stack.append(b[h:]);stack.append(b[:h])
    missing=[k for k in ks if k not in rows]
    if missing: problems.append(('missing_after',E,len(missing)))
    done_enz[E]=len(ks)
done_enz={}
def run(lst, budget=800):
    t=time.time()
    for E in lst:
        if E in done_enz: continue
        if time.time()-t>budget: break
        try:
            do_enzyme(E)
        except Exception as e:
            problems.append(('exc',E,str(e)[:100]))
    print(round(time.time()-t), ncalls, len(rows), len(done_enz), problems[-10:])
run(kin)

# ── cell 16 ──
print(len(problems)); print(Counter(p[0] for p in problems)); print(problems)

# ── cell 17 ──
E='P12931'
ks,ok,L=min_rows("&enzymes="+E)
print(ok,L,len(ks))
HL=len("enzyme\tsubstrate\tresidue_type\tresidue_offset\tmodification\n")
tot=set()
for res in "STY":
    k2,ok2,L2=min_rows("&enzymes="+E+"&residues="+res)
    print(res, ok2, L2, len(k2))
    tot|=k2
    if not ok2:
        for db in DBS:
            k3,ok3,L3=min_rows("&enzymes="+E+"&residues="+res+"&databases="+db)
            print('  ',db,ok3,L3,len(k3)); tot|=k3
got=sum(len('\t'.join(k))+len('\tphosphorylation\n') for k in tot)
print(got, L-HL, len(tot), count("&enzymes="+E))
r=wf(MB+"&enzymes="+E); print(repr(r['content'][:200]))

# ── cell 18 ──
big=[p[1] for p in problems]
have=Counter(k[0] for k in rows)
orc={}
for E in big:
    orc[E]=count("&enzymes="+E)
    print(E, orc[E], have[E])
U=sorted(set(k[1] for k in rows)); print(len(U), len(set(k[0] for k in rows)))

# ── cell 19 ──
def min_by_universe(E, U, bs=150):
    ks=set()
    stack=[U[i:i+bs] for i in range(0,len(U),bs)][::-1]
    while stack:
        b=stack.pop()
        k,ok,L=min_rows("&enzyme_substrate=AND&enzymes="+E+"&substrates="+",".join(b))
        if ok or len(b)==1:
            ks|=k
            if not ok: problems.append(('min_single_trunc',E,b[0]))
        else:
            h=len(b)//2; stack.append(b[h:]); stack.append(b[:h])
    return ks
def fetch_pairs(E, subs_count):
    subs=sorted(subs_count)
    batches=[];cur=[];cn=0
    for s in subs:
        if cur and cn+subs_count[s]>100:
            batches.append(cur);cur=[];cn=0
        cur.append(s);cn+=subs_count[s]
    if cur: batches.append(cur)
    stack=batches[::-1]
    while stack:
        b=stack.pop()
        ok,n=full_fetch("&enzyme_substrate=AND&enzymes="+E+"&substrates="+",".join(b))
        if not ok:
            if len(b)==1:
                for res in "STY":
                    ok2,n2=full_fetch("&enzyme_substrate=AND&enzymes="+E+"&substrates="+b[0]+"&residues="+res)
                    if not ok2: problems.append(('single_trunc',E,b[0],res))
            else:
                h=len(b)//2; stack.append(b[h:]); stack.append(b[:h])
problems=[]
t=time.time()
for E in big:
    if orc[E]==have[E]: continue
    ks=min_by_universe(E,U)
    miss=[k for k in ks if k not in rows]
    print(E, len(ks), len(miss))
    if miss:
        fetch_pairs(E, Counter(k[1] for k in miss))
have=Counter(k[0] for k in rows)
for E in big: print(E, orc[E], have[E])
print(time.time()-t, problems, len(rows))

# ── cell 20 ──
def up(a,b):
    q=urllib.parse.quote(f"(organism_id:9606) AND (reviewed:true) AND (length:[{a} TO {b}])")
    r=wf(f"https://rest.uniprot.org/uniprotkb/stream?query={q}&fields=accession&format=tsv")
    return r
human=set()
def rec(a,b):
    r=up(a,b)
    if r['truncated']:
        if a==b:
            print('cannot split',a); return
        m=(a+b)//2
        rec(a,m); rec(m+1,b)
    else:
        acc=[x.strip() for x in r['content'].split('\n')[1:] if x.strip()]
        human.update(acc)
for a,b in [(1,150),(151,250),(251,350),(351,450),(451,550),(551,700),(701,900),(901,1300),(1301,40000)]:
    rec(a,b)
print(len(human), ncalls)

# ── cell 21 ──
M=set()
min_done=set()
def min_subs(batch_list):
    stack=batch_list[::-1]
    while stack:
        b=stack.pop()
        k,ok,L=min_rows("&substrates="+",".join(b))
        if ok:
            M.update(k); min_done.update(b)
        elif len(b)==1:
            # split by residue, then db
            allk=set(k); good=True
            for res in "STY":
                k2,ok2,L2=min_rows("&substrates="+b[0]+"&residues="+res)
                allk|=k2
                if not ok2: good=False
            HL=len("enzyme\tsubstrate\tresidue_type\tresidue_offset\tmodification\n")
            got=sum(len('\t'.join(x))+len('\tphosphorylation\n') for x in allk)
            if not good or got!=L-HL: problems.append(('sub_incomplete',b[0],got,L-HL))
            M.update(allk); min_done.add(b[0])
        else:
            h=len(b)//2; stack.append(b[h:]); stack.append(b[:h])
t=time.time()
cnt=Counter(k[1] for k in rows)
# known substrates: batch by known row counts (~300 rows per batch)
batches=[];cur=[];cn=0
for s in U:
    if cur and cn+cnt[s]>250:
        batches.append(cur);cur=[];cn=0
    cur.append(s);cn+=cnt[s]
if cur: batches.append(cur)
print(len(batches))
min_subs(batches)
print(time.time()-t, len(M), problems, ncalls)

# ── cell 22 ──
t=time.time()
rest=sorted(human-set(U))
min_subs([rest[i:i+200] for i in range(0,len(rest),200)])
print(time.time()-t, len(M), problems, ncalls)
ENZ=sorted(set(k[0] for k in M)); print(len(ENZ), len(set(k[1] for k in M)))

# ── cell 23 ──
problems=[]
for S_ in ['P10636','P24928']:
    before=len([k for k in M if k[1]==S_])
    stack=[ENZ[i:i+150] for i in range(0,len(ENZ),150)][::-1]
    while stack:
        b=stack.pop()
        k,ok,L=min_rows("&enzyme_substrate=AND&substrates="+S_+"&enzymes="+",".join(b))
        if ok: M.update(k)
        elif len(b)==1:
            problems.append(('pair_trunc',S_,b[0])); M.update(k)
        else:
            h=len(b)//2; stack.append(b[h:]); stack.append(b[:h])
    after=len([k for k in M if k[1]==S_])
    print(S_, before, after, count("&substrates="+S_))
print(len(M), problems)

# ── cell 24 ──
for S_ in ['P10636','P24928']:
    for res in "STY":
        k2,ok2,L2=min_rows("&substrates="+S_+"&residues="+res)
        M.update(k2)
        if not ok2:
            for db in DBS:
                k3,ok3,L3=min_rows("&substrates="+S_+"&residues="+res+"&databases="+db)
                M.update(k3)
                if not ok3: print('trunc',S_,res,db,L3)
    print(S_, len([k for k in M if k[1]==S_]), count("&substrates="+S_))
print(len(M))

# ── cell 25 ──
t=time.time()
problems=[]
missk=[k for k in M if k not in rows]
print(len(missk))
menz=Counter(k[0] for k in missk)
tot_enz=Counter(k[0] for k in M)
elist=sorted(menz)
batches=[];cur=[];cn=0
for e in elist:
    if cur and cn+tot_enz[e]>100:
        batches.append(cur);cur=[];cn=0
    cur.append(e);cn+=tot_enz[e]
if cur: batches.append(cur)
print(len(elist), len(batches))
stack=batches[::-1]
while stack:
    b=stack.pop()
    ok,n=full_fetch("&enzymes="+",".join(b))
    if not ok:
        if len(b)==1:
            E=b[0]
            fetch_pairs(E, Counter(k[1] for k in M if k[0]==E and k not in rows))
        else:
            h=len(b)//2; stack.append(b[h:]); stack.append(b[:h])
missk=[k for k in M if k not in rows]
print(time.time()-t, len(rows), len(missk), problems, len([k for k in rows if k not in M]))

# ── cell 26 ──
print(Counter(v['res'] for v in rows.values()))
print(sum(1 for v in rows.values() if not v['kinase_gene']), sum(1 for v in rows.values() if not v['substrate_gene']))
print(Counter(s for v in rows.values() for s in v['sources']).most_common())
import random
print(random.sample(list(rows.values()),3))
subs=sorted(set(k[1] for k in rows)); print(len(subs), [s for s in subs if '-' in s][:5], len([e for e in set(k[0] for k in rows) if e not in human]))
# save checkpoint
json.dump(list(rows.values()), open('omnipath_rows.json','w'))

# ── cell 27 ──
for u in ["https://signor.uniroma2.it/getData.php?organism=9606&id=P06239",
          "https://signor.uniroma2.it/getData.php?type=phosphorylation&organism=9606&id=P06239",
          "https://signor.uniroma2.it/API/",
          "https://signor.uniroma2.it/APIs.php"]:
    try:
        r=web_fetch(url=u)
        print(u, r.get('length'), r.get('truncated'), repr(r.get('content','')[:700]))
    except Exception as e:
        print(u,'ERR',e)

# ── cell 28 ──
t=time.time()
lens={}
def up_fields(accs, fields):
    q=urllib.parse.quote("("+" OR ".join("accession:"+a for a in accs)+")")
    return wf(f"https://rest.uniprot.org/uniprotkb/stream?query={q}&fields={fields}&format=tsv")
stack=[subs[i:i+100] for i in range(0,len(subs),100)][::-1]
while stack:
    b=stack.pop()
    try:
        r=up_fields(b,"accession,length")
    except Exception as e:
        if len(b)>1:
            h=len(b)//2; stack.append(b[h:]); stack.append(b[:h]); continue
        print('fail',b,e); continue
    if r['truncated']:
        h=len(b)//2; stack.append(b[h:]); stack.append(b[:h]); continue
    for ln in r['content'].split('\n')[1:]:
        p=ln.split('\t')
        if len(p)==2 and p[1].isdigit(): lens[p[0]]=int(p[1])
print(time.time()-t, len(lens), len(subs), [s for s in subs if s not in lens][:10])

# ── cell 29 ──
t=time.time()
seqs={}
order=sorted(subs, key=lambda s: lens[s])
batches=[];cur=[];cn=0
for s in order:
    cost=lens[s]+8
    if cur and (cn+cost>14500 or len(cur)>=60):
        batches.append(cur);cur=[];cn=0
    cur.append(s);cn+=cost
if cur: batches.append(cur)
print(len(batches), sum(lens.values()))
partial={}
stack=batches[::-1]
while stack:
    b=stack.pop()
    r=up_fields(b,"accession,sequence")
    if r['truncated'] and len(b)>1:
        h=len(b)//2; stack.append(b[h:]); stack.append(b[:h]); continue
    lines=r['content'].split('\n')[1:]
    for ln in lines:
        p=ln.split('\t')
        if len(p)==2:
            if len(p[1])==lens.get(p[0]): seqs[p[0]]=p[1]
            elif r['truncated']: partial[p[0]]=p[1]
print(time.time()-t, len(seqs), len(partial), {k:(len(v),lens[k]) for k,v in partial.items()})

# ── cell 30 ──
atlas={}
mism=0; nopep=0
for v in rows.values():
    site=v['res']+v['off']; acc=v['substrate']; off=int(v['off'])
    seq=seqs.get(acc) or partial.get(acc,'')
    pep=''
    full = acc in seqs
    if 1<=off<=len(seq) and seq[off-1]==v['res'] and (full or off+3<=len(seq)):
        pep=''.join(seq[i] if 0<=i<len(seq) else '_' for i in range(off-4, off+3))
    else:
        if full: mism+=1
        nopep+=1
    key=(v['kinase_gene'],v['substrate_gene'],site,acc)
    e=atlas.get(key)
    if e is None:
        atlas[key]={'kinase_gene':v['kinase_gene'],'substrate_gene':v['substrate_gene'],'phospho_site':site,
                    'heptameric_peptide':pep,'substrate_uniprot':acc,'supporting_databases':list(v['sources'])}
    else:
        for s in v['sources']:
            if s not in e['supporting_databases']: e['supporting_databases'].append(s)
entries=list(atlas.values())
for e in entries:
    e['supporting_databases']=sorted(e['supporting_databases'])+['OmniPath']
    if not e['heptameric_peptide']: del e['heptameric_peptide']
print(len(entries), mism, nopep, len(set((e['kinase_gene'],e['substrate_gene'],e['phospho_site']) for e in entries)))
print(entries[:2])
kinset=set(kin)
nonk=sum(1 for v in rows.values() if v['enzyme'] not in kinset); print('rows with enzyme not UniProt-kinase-keyword:', nonk, len(set(v['enzyme'] for v in rows.values() if v['enzyme'] not in kinset)))
tm_only=sum(1 for v in rows.values() if v['enzyme'] not in kinset and all('ProtMapper' in s for s in v['sources'])); print(tm_only)

# ── cell 31 ──
json.dump(entries, open('atlas_omnipath.json','w'))
r=ingest_entries_file(path='atlas_omnipath.json', source_name='OmniPath')
print(r)

# ── cell 32 ──
summary=("Source: OmniPath enz_sub web service (organisms=9606, types=phosphorylation, academic licence), which aggregates PhosphoSite, SIGNOR, phosphoELM, HPRD, dbPTM, KEA, MIMP, PhosphoNetworks, Li2012 and ProtMapper-derived text-mining sets. "
"Retrieved via web_fetch in <15k-char chunks (per-enzyme, then per-substrate over all 20,429 reviewed human UniProt accessions, with residue/database/pair splits for oversized chunks); completeness verified: 39,037 unique enzyme-substrate-residue rows collected = total row count reported for the unfiltered query. "
"Per-row source lists kept as given by OmniPath (plus 'OmniPath'). Heptamers (+/-3) derived from UniProt canonical sequences only where the residue at the stated position matches; left blank for 3,478 rows where it does not or sequence unavailable. "
"Not included: PhosphoSitePlus direct download (login-gated), SIGNOR direct dump (too large for chunked fetch; covered only via OmniPath's SIGNOR snapshot). "
"Caveat: 6,419 rows have an 'enzyme' not carrying the UniProt kinase keyword (mostly text-mining-only ProtMapper records); kept for exhaustiveness. fetch_and_parse_db reported entries but did not grow the atlas, so everything was ingested via ingest_entries_file.")
r=submit_atlas(additional_entries=[], strategy_summary=summary)
print(r)

