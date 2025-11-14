import sys, os, re, glob
pagedir = sys.argv[1]
out = sys.argv[2]
os.makedirs(os.path.dirname(out), exist_ok=True)
patterns = {
  "k_number": r"\bK10\d{3}\b|\bK101995\b",
  "trade_or_device_name": r"(trade name|device name|proprietary name|capnostream)",
  "applicant_or_mfr": r"(applicant|manufacturer|submitted by|company|address)",
  "common_name": r"(common name)",
  "ifu": r"(indications for use)",
  "predicate": r"(predicate device|substantial equivalence|SE comparison)",
  "product_code": r"(product code|procode)",
  "regulation": r"(regulation number|21\s*CFR|regulation no\.)",
  "device_class": r"(class\s*[iIvVxX]+|device class)",
}
files = sorted(glob.glob(os.path.join(pagedir, "*.txt")))
with open(out, "w", encoding="utf-8") as w:
    for key, pat in patterns.items():
        w.write(f"## {key}\n")
        rx = re.compile(pat, re.IGNORECASE)
        for f in files:
            pg = re.search(r'_p(\d+)\.txt$', f).group(1)
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, start=1):
                    if rx.search(line):
                        w.write(f"p{pg}:{i}: {line.strip()}\n")
        w.write("\n")
print(out)
