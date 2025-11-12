import argparse, re, glob, os, datetime
import pandas as pd

def read_text_blocks(pages_dir, knum):
    paths = sorted(glob.glob(os.path.join(pages_dir, f"{knum}_p*.txt")))
    blocks = []
    for p in paths:
        try:
            t = open(p, "r", encoding="utf-8", errors="ignore").read()
        except:
            t = open(p, "r", errors="ignore").read()
        blocks.append((p, t))
    return blocks

def load_triples(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    need = ["subject","predicate","object","source_documents","intext_evidence","timestamp"]
    for c in need:
        if c not in df.columns:
            df[c] = ""
    return df[need]

def now_iso():
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()

def add_row(rows, s,p,o,src,ev):
    rows.append({"subject":s,"predicate":p,"object":o,"source_documents":src,"intext_evidence":ev,"timestamp":now_iso()})

def first_applicant(blocks):
    lab = re.compile(r"\b(Applicant|Submitter|Submitter's Name|Manufacturer|Company|Owner)\b", re.I)
    for src, txt in blocks:
        lines = txt.splitlines()
        for i, line in enumerate(lines):
            if lab.search(line):
                window = " ".join(lines[i:i+4])
                m = re.search(r"(Applicant|Submitter|Submitter's Name|Manufacturer|Company|Owner)[^:]{0,20}[:\-]?\s*([A-Za-z0-9&().,\- ]{3,})", window, re.I)
                if m:
                    val = m.group(2).strip()
                    val = re.sub(r"\s{2,}.*$", "", val)
                    val = re.sub(r"[,;.]$", "", val)
                    if len(val) >= 3:
                        return src, window.strip(), val
    comp = re.compile(r"\b([A-Z][A-Za-z0-9&.\- ]{2,}?(?:Inc\.?|Incorporated|Ltd\.?|LLC|Corp\.?))\b")
    for src, txt in blocks:
        m = comp.search(txt)
        if m:
            return src, m.group(0), m.group(0)
    return None, None, None

def product_codes(blocks):
    hits = []
    key = re.compile(r"Product\s*Codes?", re.I)
    codepat = re.compile(r"\b[A-Z]{3}\b")
    for src, txt in blocks:
        lines = txt.splitlines()
        for i, line in enumerate(lines):
            if key.search(line):
                cand = " ".join(lines[i:i+2])
                for c in codepat.findall(cand):
                    hits.append((src, cand.strip(), c))
    seen = set()
    out = []
    for src, ev, c in hits:
        if c not in seen:
            seen.add(c)
            out.append((src, ev, c))
    return out

def predicate_ks(blocks):
    ks = []
    key = re.compile(r"Predicate|Substantial\s+Equivalence|SE\s+to|equivalent", re.I)
    kpat = re.compile(r"\bK\d{6}\b")
    for src, txt in blocks:
        for line in txt.splitlines():
            if key.search(line) and kpat.search(line):
                for k in kpat.findall(line):
                    ks.append((src, line.strip(), k))
    seen = set()
    out = []
    for src, ev, k in ks:
        if k not in seen:
            seen.add(k)
            out.append((src, ev, k))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knum", required=True)
    ap.add_argument("--pages_dir", required=True)
    ap.add_argument("--triples", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    blocks = read_text_blocks(args.pages_dir, args.knum)
    df = load_triples(args.triples)
    rows = df.to_dict("records")
    s_md = f"MedicalDevice:{args.knum}"
    s_sub = f"Submission:{args.knum}"
    src, ev, val = first_applicant(blocks)
    if val:
        add_row(rows, s_md, "hasApplicant", val, src or "", ev or "")
    for src, ev, code in product_codes(blocks):
        add_row(rows, s_md, "hasProductCode", code, src or "", ev or "")
    for src, ev, kpred in predicate_ks(blocks):
        add_row(rows, s_sub, "HASPREDICATEDEVICE", f"Submission:{kpred}", src or "", ev or "")
    out = pd.DataFrame(rows).drop_duplicates(subset=["subject","predicate","object"])
    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
