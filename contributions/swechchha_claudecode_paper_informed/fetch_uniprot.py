import sys, re, urllib.request, time
out = sys.argv[1]
url = ("https://rest.uniprot.org/uniprotkb/search?query=organism_id:9606+AND+reviewed:true+AND+ft_mod_res:phospho*"
       "&fields=accession,gene_primary,ft_mod_res,sequence&format=tsv&size=500")
first = True
n = 0
with open(out, "w") as f:
    while url:
        for attempt in range(6):
            try:
                r = urllib.request.urlopen(url, timeout=120); break
            except Exception as e:
                print("retry", e, file=sys.stderr); time.sleep(10)
        else:
            sys.exit("failed")
        body = r.read().decode()
        lines = body.splitlines(True)
        f.writelines(lines if first else lines[1:])
        n += len(lines) - 1; first = False
        m = re.search(r'<([^>]+)>; rel="next"', r.headers.get("Link", ""))
        url = m.group(1) if m else None
        print(n, file=sys.stderr)
