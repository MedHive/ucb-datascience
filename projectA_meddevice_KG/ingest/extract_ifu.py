import sys, os, re
page_path = sys.argv[1]
out_path = sys.argv[2]
with open(page_path, encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
idx = next((i for i,l in enumerate(lines) if re.search(r'indications\s+for\s+use', l, re.I)), None)
if idx is None:
    print("ERROR: IFU header not found", file=sys.stderr)
    sys.exit(1)
body = ''.join(lines[idx+1:]).strip()
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as out:
    out.write(body + "\n")
print(out_path)
