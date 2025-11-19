import argparse, csv, re
from pathlib import Path
from datetime import datetime, timezone
import harvest_cues_generic as hc
from collections import Counter

REG_TO_CLASS = {"868.1400": "Class II", "870.2700": "Class II"}
PCODE_TO_CLASS = {"CCK": "Class II", "DQA": "Class II", "COK": "Class II"}

MONTH_NAME_PATTERN = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
    re.I,
)

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+[@~][A-Za-z0-9.-]+\.[A-Za-z]{2,}")

SOFTWARE_VERSION_PATTERN = re.compile(
    r"(software\s+version|Version)\s*[:=]?\s*([A-Za-z0-9._-]+)",
    re.I,
)

TESTING_HEADERS = [
    "Performance Testing",
    "Bench Testing",
    "Nonclinical Testing",
    "Non clinical Testing",
    "Clinical Testing",
    "Verification and Validation",
    "Software Verification and Validation",
]

def uniq(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def roman_normalize(v):
    if not v:
        return v
    v = v.upper().strip().replace("CLASS", "").strip()
    mapping = {"1": "I", "I": "I", "2": "II", "II": "II", "11": "II", "IL": "II", "3": "III", "III": "III"}
    m = re.search(r"(III|II|I|11|IL|3|2|1)", v)
    return "Class " + mapping.get(m.group(1), "II") if m else None

def class_tokens_from_pages(pages):
    tokens = set()
    for _, txt in pages:
        for m in re.finditer(r"\(Classification\s+([A-Z0-9]{3,4})\)", txt or "", flags=re.I):
            tokens.add(m.group(1).upper())
    return tokens

def class_mentions_from_pages(pages):
    vals = []
    for _, txt in pages:
        for m in re.finditer(r"(Regulatory\s*Class\s*[:\-]?\s*(\d+)|Class\s+(III|II|I))", txt or "", flags=re.I):
            num = m.group(2)
            roman = m.group(3)
            if num:
                vals.append(roman_normalize(num))
            elif roman:
                vals.append(roman_normalize(roman))
    return [v for v in vals if v]

def pick_device_class(regs, pcodes, values, mentions):
    for rn in regs:
        if rn in REG_TO_CLASS:
            return REG_TO_CLASS[rn]
    mapped = [PCODE_TO_CLASS[p] for p in pcodes if p in PCODE_TO_CLASS]
    if mapped:
        return Counter(mapped).most_common(1)[0][0]
    pool = [roman_normalize(v) for v in values if v] + [roman_normalize(v) for v in mentions if v]
    pool = [p for p in pool if p and p.startswith("Class ")]
    if not pool:
        return "Class II"
    counts = Counter(pool)
    pref = {"Class II": 3, "Class III": 2, "Class I": 1}
    return sorted(counts.items(), key=lambda kv: (kv[1], pref.get(kv[0], 0)), reverse=True)[0][0]

def first_or_none(lst):
    return lst[0] if lst else None

def extract_decision_date(pages):
    for idx, (fname, text) in enumerate(pages):
        if idx >= 3:
            break
        lower_text = text.lower()
        pos = lower_text.find("decision date")
        if pos != -1:
            snippet = text[pos : pos + 160]
            m = MONTH_NAME_PATTERN.search(snippet)
            if m:
                return fname, m.group(0).strip()
    return None, None

def extract_applicant_email(pages):
    for idx, (fname, text) in enumerate(pages):
        if idx >= 3:
            break
        for line in text.splitlines():
            m = EMAIL_PATTERN.search(line)
            if m:
                email = m.group(0).replace("~", "@")
                return fname, email, line.strip()
    return None, None, None

def extract_software_version(pages):
    for fname, text in pages:
        m = SOFTWARE_VERSION_PATTERN.search(text)
        if m:
            full = m.group(0).strip()
            version = m.group(2).strip()
            return fname, version, full
    return None, None, None

def extract_testing_sections(pages):
    results = []
    for fname, text in pages:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            for header in TESTING_HEADERS:
                if header.lower() in line.lower():
                    window = " ".join(lines[i : i + 8]).strip()
                    if window:
                        results.append((fname, header, window))
                    break
    return results

def extract_predicate_details(pages, kpred):
    pattern = re.compile(r".{0,80}\b" + re.escape(kpred) + r"\b.{0,80}", re.I | re.S)
    org_pattern = re.compile(
        r"([A-Z][A-Za-z0-9&\-\(\) ]+(?:Corporation|Corp\.?|Inc\.?|Ltd\.?|LLC|GmbH|S\.A\.?|S\.p\.A\.?))"
    )
    for fname, text in pages:
        for m in pattern.finditer(text):
            snippet = re.sub(r"\s+", " ", m.group(0)).strip()
            snippet_no_k = snippet.replace(kpred, "").strip(" ,;:.()")
            name = snippet_no_k
            org_match = org_pattern.search(snippet)
            org = org_match.group(1).strip() if org_match else None
            return fname, name, org, snippet
    return None, None, None, None

def make_rows(knum, pages_dir):
    pages = hc.read_pages(Path(pages_dir), knum)
    hits = hc.harvest_fields(pages)
    ifu_text, ifu_file = hc.extract_ifu(pages)
    now = datetime.now(timezone.utc).isoformat()

    subj = f"Submission:{knum}"
    dev = f"MedicalDevice:{knum}"

    rows = []
    rows.append(
        {
            "subject": subj,
            "predicate": "hasKNumber",
            "object": knum,
            "source_documents": f"{knum}_p01.txt",
            "intext_evidence": knum,
            "creation_timestamp": now,
            "last_updated_timestamp": now,
        }
    )
    rows.append(
        {
            "subject": subj,
            "predicate": "CONCERNSDEVICE",
            "object": dev,
            "source_documents": f"{knum}_p01.txt",
            "intext_evidence": "see DEVICE INFORMATION block",
            "creation_timestamp": now,
            "last_updated_timestamp": now,
        }
    )

    trades = uniq([h["value"] for h in hits.get("trade_name", [])])
    common = uniq([h["value"] for h in hits.get("common_name", [])])
    regs = uniq([h["value"] for h in hits.get("regulation_number", [])])
    klass_candidates = uniq([h["value"] for h in hits.get("device_class", [])])
    klass_mentions = class_mentions_from_pages(pages)
    apps = uniq([h["value"] for h in hits.get("applicant", [])])

    cls_tokens = class_tokens_from_pages(pages)
    harvested_pcs = [h["value"].upper() for h in hits.get("product_code", [])]
    pcodes = uniq([pc for pc in harvested_pcs if pc in cls_tokens])

    klass = pick_device_class(regs, pcodes, klass_candidates, klass_mentions)

    if trades:
        rows.append(
            {
                "subject": dev,
                "predicate": "hasTradeName",
                "object": trades[0],
                "source_documents": hits["trade_name"][0]["file"],
                "intext_evidence": trades[0],
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )
    if common:
        rows.append(
            {
                "subject": dev,
                "predicate": "hasCommonName",
                "object": common[0],
                "source_documents": hits["common_name"][0]["file"],
                "intext_evidence": common[0],
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )
    for pc in pcodes:
        src = first_or_none([h["file"] for h in hits["product_code"] if h["value"].upper() == pc]) or f"{knum}_p01.txt"
        rows.append(
            {
                "subject": dev,
                "predicate": "hasProductCode",
                "object": pc,
                "source_documents": src,
                "intext_evidence": pc,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )
    for rn in regs:
        src = first_or_none([h["file"] for h in hits["regulation_number"] if h["value"] == rn]) or f"{knum}_p01.txt"
        rows.append(
            {
                "subject": dev,
                "predicate": "hasRegulationNumber",
                "object": rn,
                "source_documents": src,
                "intext_evidence": rn,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )
    if klass:
        src = first_or_none([h["file"] for h in hits.get("device_class", [])]) or f"{knum}_p01.txt"
        rows.append(
            {
                "subject": dev,
                "predicate": "hasDeviceClass",
                "object": klass,
                "source_documents": src,
                "intext_evidence": klass,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )
    for a in apps:
        rows.append(
            {
                "subject": subj,
                "predicate": "HASAPPLICANT",
                "object": f"Applicant:{a}",
                "source_documents": hits["applicant"][0]["file"] if hits.get("applicant") else f"{knum}_p01.txt",
                "intext_evidence": a,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )

    preds = uniq(
        [
            h["value"]
            for h in hits.get("predicate_knums", [])
            if re.fullmatch(r"K\d{6}", h["value"]) and h["value"] != knum
        ]
    )
    for pk in preds:
        src = first_or_none([h["file"] for h in hits["predicate_knums"] if h["value"] == pk]) or f"{knum}_p02.txt"
        rows.append(
            {
                "subject": subj,
                "predicate": "HASPREDICATEDEVICE",
                "object": f"PredicateDevice:{pk}",
                "source_documents": src,
                "intext_evidence": pk,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )
        pd_src, pd_name, pd_org, pd_snip = extract_predicate_details(pages, pk)
        if pd_name:
            rows.append(
                {
                    "subject": f"PredicateDevice:{pk}",
                    "predicate": "hasPredicateName",
                    "object": pd_name,
                    "source_documents": pd_src or src,
                    "intext_evidence": pd_snip or pd_name,
                    "creation_timestamp": now,
                    "last_updated_timestamp": now,
                }
            )
        if pd_org:
            rows.append(
                {
                    "subject": f"PredicateDevice:{pk}",
                    "predicate": "hasPredicateManufacturer",
                    "object": pd_org,
                    "source_documents": pd_src or src,
                    "intext_evidence": pd_snip or pd_org,
                    "creation_timestamp": now,
                    "last_updated_timestamp": now,
                }
            )

    if ifu_text:
        excerpt = ifu_text[:950]
        rows.append(
            {
                "subject": dev,
                "predicate": "hasIndicationsForUseText",
                "object": excerpt,
                "source_documents": ifu_file or f"{knum}_p01.txt",
                "intext_evidence": excerpt[:200],
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )

    src_dd, decision = extract_decision_date(pages)
    if decision:
        rows.append(
            {
                "subject": subj,
                "predicate": "hasDecisionDate",
                "object": decision,
                "source_documents": src_dd or f"{knum}_p01.txt",
                "intext_evidence": decision,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )

    src_email, email, email_line = extract_applicant_email(pages)
    if email and apps:
        rows.append(
            {
                "subject": f"Applicant:{apps[0]}",
                "predicate": "hasContactEmail",
                "object": email,
                "source_documents": src_email or f"{knum}_p01.txt",
                "intext_evidence": email_line or email,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )

    src_sw, sw_version, sw_full = extract_software_version(pages)
    if sw_version:
        sw_uri = f"SoftwareInformation:{knum}"
        rows.append(
            {
                "subject": dev,
                "predicate": "hasSoftwareComponent",
                "object": sw_uri,
                "source_documents": src_sw or f"{knum}_p01.txt",
                "intext_evidence": sw_full or sw_version,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )
        rows.append(
            {
                "subject": sw_uri,
                "predicate": "hasSoftwareVersion",
                "object": sw_version,
                "source_documents": src_sw or f"{knum}_p01.txt",
                "intext_evidence": sw_full or sw_version,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )

    tests = extract_testing_sections(pages)
    for idx, (src_t, header_t, snippet_t) in enumerate(tests, start=1):
        t_uri = f"TestingData:{knum}:{idx}"
        rows.append(
            {
                "subject": dev,
                "predicate": "supportedByTestingData",
                "object": t_uri,
                "source_documents": src_t or f"{knum}_p01.txt",
                "intext_evidence": snippet_t,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )
        rows.append(
            {
                "subject": t_uri,
                "predicate": "hasTestType",
                "object": header_t,
                "source_documents": src_t or f"{knum}_p01.txt",
                "intext_evidence": snippet_t,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )
        rows.append(
            {
                "subject": t_uri,
                "predicate": "hasTestSummaryText",
                "object": snippet_t,
                "source_documents": src_t or f"{knum}_p01.txt",
                "intext_evidence": snippet_t,
                "creation_timestamp": now,
                "last_updated_timestamp": now,
            }
        )

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knum", required=True)
    ap.add_argument("--pages_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    a = ap.parse_args()

    rows = make_rows(a.knum, a.pages_dir)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "predicate",
                "object",
                "source_documents",
                "intext_evidence",
                "creation_timestamp",
                "last_updated_timestamp",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(a.out_csv, len(rows))

if __name__ == "__main__":
    main()

def extract_decision_date(pages):
    months = [
        "january","february","march","april","may","june",
        "july","august","september","october","november","december"
    ]
    for idx, (fname, text) in enumerate(pages):
        if idx >= 3:
            break
        for line in text.splitlines():
            lower = line.lower()
            if any(m in lower for m in months) and re.search(r"\b(19|20)\d{2}\b", line):
                return fname, line.strip()
    return None, None

def extract_testing_sections(pages):
    results = []
    for fname, text in pages:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            for header in TESTING_HEADERS:
                if header.lower() in line.lower():
                    window = " ".join(lines[i : i + 8]).strip()
                    if window:
                        results.append((fname, header, window))
                    break
    if results:
        return results

    for fname, text in pages:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "testing" in line.lower():
                window = " ".join(lines[max(0, i - 2) : i + 4]).strip()
                if window:
                    results.append((fname, "Testing", window))
    return results

