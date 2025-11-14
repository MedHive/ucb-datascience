import sys, os
src = sys.argv[1]
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)
with open(src, 'r', encoding='utf-8', errors='ignore') as f:
    txt = f.read()
pages = [p.strip() for p in txt.split('\x0c') if p.strip()]
for i, p in enumerate(pages, start=1):
    fn = os.path.join(outdir, f"K101995_p{str(i).zfill(2)}.txt")
    with open(fn, 'w', encoding='utf-8') as out:
        out.write(p + "\n")
print(len(pages))
