import os, re, glob, json
pagedir = "projectA_meddevice_KG/data/texts/pages"
out_md = "projectA_meddevice_KG/data/extracts/K101995_fields.md"

files = sorted(glob.glob(os.path.join(pagedir, "K101995_p*.txt")))
def readp(f):
    pg = int(re.search(r'_p(\d+)\.txt$', f).group(1))
    with open(f, encoding="utf-8", errors="ignore") as fh:
        return pg, fh.readlines()

def find_first(patterns):
    rx = re.compile(patterns, re.I)
    for f in files:
        pnum, lines = readp(f)
        for i,l in enumerate(lines,1):
            m = rx.search(l)
            if m:
                return pnum, i, l.strip(), m
    return None

def find_all(patterns):
    hits = []
    rx = re.compile(patterns, re.I)
    for f in files:
        pnum, lines = readp(f)
        for i,l in enumerate(lines,1):
            if rx.search(l):
                hits.append((pnum, i, l.strip()))
    return hits

def snag_after(header_regex, take_lines=5):
    rx = re.compile(header_regex, re.I)
    for f in files:
        pnum, lines = readp(f)
        for i,l in enumerate(lines):
            if rx.search(l):
                block = []
                for j in range(i+1, min(i+1+take_lines, len(lines))):
                    s = lines[j].strip()
                    if s:
                        block.append(s)
                if block:
                    return pnum, i+2, block
    return None

# Trade / device name
trade = find_first(r"(device\s*name|trade\s*name|proprietary\s*name)\s*:\s*(.+)")
if not trade:
    # fallback: any line with Capnostream
    trade = find_first(r"\bcapnostream ?20p\b.*")

# Common name
common = find_first(r"common\s*name\s*:\s*(.+)")

# Applicant / Manufacturer
submitter_block = snag_after(r"SUBMITTER['’]S\s+NAME\s+AND\s+ESTABLISHMENT\s+ADDRESS", take_lines=6)
manufacturer_block = snag_after(r"^\s*Manufacturer\s*$", take_lines=4)

# Regulations and class
reg_lines = find_all(r"\b21\s*CFR\b.*")
class_lines = find_all(r"\bclass\s*(i{1,3}|iv|v|vi|ii|iii)\b")

# Product codes (look for Product Code or three-letter codes in parentheses following classification)
prod_lines = find_all(r"(product\s*code\s*:?\s*[A-Z0-9]{3})|\(Classification\s*[A-Z0-9]{3}\)")

# Predicates and K-numbers
pred_context = find_all(r"predicate\s+device|substantial\s+equivalence")
knums = []
rxk = re.compile(r"\bK\d{6}\b", re.I)
for f in files:
    pnum, lines = readp(f)
    for i,l in enumerate(lines,1):
        for m in rxk.findall(l):
            knums.append((pnum, i, m))
knums = sorted(set(knums))

def fmt_hits(hits, label):
    if not hits: return f"## {label}\n(none found)\n"
    out = [f"## {label}"]
    for p,i,t in hits:
        out.append(f"- p.{p}:{i}: {t}")
    return "\n".join(out) + "\n"

with open(out_md, "w", encoding="utf-8") as w:
    w.write("# K101995 field harvest\n\n")
    # Trade
    if trade:
        p,i,line,m = trade
        val = m.group(2).strip() if m and m.groups() else line
        w.write(f"## trade_or_device_name\n- p.{p}:{i}: {val}\n\n")
    else:
        w.write("## trade_or_device_name\n(none found)\n\n")
    # Common name
    if common:
        p,i,line,m = common
        val = m.group(1).strip()
        w.write(f"## common_name\n- p.{p}:{i}: {val}\n\n")
    else:
        w.write("## common_name\n(none found)\n\n")
    # Applicant / Manufacturer
    if submitter_block:
        p,i,blk = submitter_block
        w.write(f"## applicant_block (starts p.{p}:{i})\n")
        for row in blk: w.write(f"- {row}\n")
        w.write("\n")
    if manufacturer_block:
        p,i,blk = manufacturer_block
        w.write(f"## manufacturer_block (starts p.{p}:{i})\n")
        for row in blk: w.write(f"- {row}\n")
        w.write("\n")
    # Regulations
    w.write(fmt_hits(reg_lines, "regulation_lines"))
    # Class
    w.write(fmt_hits(class_lines, "device_class_lines"))
    # Product codes
    w.write(fmt_hits(prod_lines, "product_code_lines"))
    # Predicate context
    w.write(fmt_hits(pred_context, "predicate_mentions"))
    # K numbers
    if knums:
        w.write("## k_numbers_found\n")
        for p,i,k in knums:
            w.write(f"- p.{p}:{i}: {k}\n")
        w.write("\n")
    else:
        w.write("## k_numbers_found\n(none found)\n\n")
print(out_md)
