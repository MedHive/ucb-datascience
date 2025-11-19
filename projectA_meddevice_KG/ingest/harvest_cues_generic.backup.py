import argparse, re
from pathlib import Path

FIELD_PATTERNS = {
    "trade_name": [
        r"Trade\s*/?\s*Device\s*Name\s*:\s*(.+)",
        r"Trade\s*Name\s*:\s*(.+)",
        r"Proprietary\s*Name\s*:\s*(.+)",
        r"Device\s*Name\s*:\s*(.+)"
    ],
    "common_name": [
        r"Common\s*Name\s*:\s*(.+)",
        r"Classification\s*Name\s*:\s*(.+)"
    ],
    "applicant": [
        r"Applicant\s*:\s*(.+)",
        r"Submitter\s*:\s*(.+)",
        r"Company\s*Name\s*:\s*(.+)",
        r"Manufacturer\s*:\s*(.+)",
        r"Sponsor\s*:\s*(.+)"
    ],
    "product_code": [
        r"Product\s*Code(?:s)?\s*:\s*([A-Z0-9,\-\s]+)",
        r"\(Classification\s+([A-Z0-9]{3,4})\)",
        r"Classification\s*[:\-]?\s*([A-Z0-9]{3,4})"
    ],
    "regulation_number": [
        r"Regulation\s*Number\s*:\s*(?:21\s*CFR\s*)?([\d]+\.[\d]+)",
        r"(?:21\s*CFR\s*)([\d]+\.[\d]+)"
    ],
    "device_class": [
        r"(?:Regulatory\s*)?Class(?:ification)?\s*[:\-]?\s*(Class\s*[I|II|III]+|I{1,3}|[1-3]|IL)"
    ],
    "predicate_knums": [
        r"Predicate\s*Device(?:s)?\s*:\s*(.+)",
        r"\b(K\d{6})\b"
    ]
}

HEADINGS_STOP = [
    r"^\s*Device\s+Description",
    r"^\s*Substantial\s+Equivalence",
    r"^\s*Performance\s+Data",
    r"^\s*Summary",
    r"^\s*Truthful\s+and\s+Accuracy",
    r"^\s*Basis\s+for",
]

def read_pages(pages_dir: Path, knum: str):
    pages = []
    for p in sorted(pages_dir.glob(f"{knum}_p*.txt")):
        pages.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
    return pages

def _append_hit(hits, key, file, value):
    value = re.sub(r"\s{2,}", " ", value.strip())
    value = value.rstrip(" .;:,")
    if value and (len(hits[key]) < 20):
        hits[key].append({"file": file, "value": value})

def harvest_fields(pages):
    hits = {k: [] for k in FIELD_PATTERNS}

    for fname, text in pages:
        for field, pats in FIELD_PATTERNS.items():
            for pat in pats:
                for m in re.finditer(pat, text, flags=re.I|re.M):
                    _append_hit(hits, field, fname, m.group(1))

    pcs = []
    for h in hits["product_code"]:
        for tok in re.split(r"[,\s/]+", h["value"]):
            tok = tok.strip().upper()
            if tok in {"", "NAME"}:
                continue
            if re.fullmatch(r"[A-Z0-9]{3,4}", tok):
                pcs.append({"file": h["file"], "value": tok})
    if pcs:
        hits["product_code"] = pcs

    if not hits["applicant"]:
        for fname, text in pages:
            m = re.search(r"SUBMITTER['’]S\s+NAME.*?:\s*$", text, flags=re.I|re.M)
            if m:
                tail = text[m.end():]
                for line in tail.splitlines():
                    line = line.strip()
                    if line:
                        _append_hit(hits, "applicant", fname, line)
                        break
                if hits["applicant"]:
                    break

    mapped = []
    roman_map = {"1":"I","2":"II","3":"III","I":"I","II":"II","III":"III","11":"II","IL":"II"}
    for h in hits["device_class"]:
        raw = h["value"].upper()
        m = re.search(r"(CLASS)?\s*(I{1,3}|1|2|3|11|IL)", raw)
        if m:
            key = m.group(2)
            norm = roman_map.get(key, h["value"])
            mapped.append({"file": h["file"], "value": f"Class {norm}" if not norm.startswith("Class") else norm})
        else:
            mapped.append(h)
    if mapped:
        hits["device_class"] = mapped

    return hits

def extract_ifu(pages):
    cap = []
    capturing = False
    for fname, text in pages:
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if not capturing and re.search(r"Indications\s+for\s+Use", line, flags=re.I):
                capturing = True
                i += 1
                continue
            if capturing:
                if any(re.search(h, line, flags=re.I) for h in HEADINGS_STOP):
                    return "\n".join(cap).strip(), fname
                cap.append(line)
            i += 1
    return ("\n".join(cap).strip() if cap else ""), (pages[0][0] if pages else None)

def write_markdown(out_md: Path, knum: str, hits: dict, ifu_text: str, ifu_file: str|None):
    md = []
    md.append(f"# Cues for {knum}")
    for key, title in [
        ("trade_name","Trade/Device Name"),
        ("common_name","Common / Classification Name"),
        ("applicant","Applicant / Submitter"),
        ("product_code","Product Code(s)"),
        ("regulation_number","Regulation Number(s)"),
        ("device_class","Device Class"),
        ("predicate_knums","Predicate devices / K-numbers"),
    ]:
        md.append(f"\n## {title}\n")
        if hits.get(key):
            for h in hits[key]:
                md.append(f"- {h['value']}  _(from {h['file']})_")
        else:
            md.append("- None found")
    md.append("\n## Indications for Use (preview)\n")
    if ifu_text:
        preview = ifu_text.strip().replace("\r","")
        preview = re.sub(r"\n{3,}", "\n\n", preview)
        md.append(f"Source: {ifu_file or 'unknown'}\n")
        md.append("```\n" + preview[:1200] + ("\n... (truncated)" if len(preview)>1200 else "") + "\n```")
    else:
        md.append("- None found")
    out_md.write_text("\n".join(md), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages_dir", required=True)
    ap.add_argument("--knum", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_ifu", required=True)
    args = ap.parse_args()

    pages = read_pages(Path(args.pages_dir), args.knum)
    hits = harvest_fields(pages)
    ifu_text, ifu_file = extract_ifu(pages)
    Path(args.out_ifu).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_ifu).write_text(ifu_text, encoding="utf-8")
    write_markdown(Path(args.out_md), args.knum, hits, ifu_text, ifu_file)

    print(f"pages={len(pages)} wrote_md={args.out_md} wrote_ifu={args.out_ifu}")
if __name__ == "__main__":
    main()
