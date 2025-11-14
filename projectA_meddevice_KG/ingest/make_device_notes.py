import re, glob, os, pathlib
root="projectA_meddevice_KG"
fields=f"{root}/data/extracts/K101995_fields.md"
ifu=f"{root}/data/extracts/K101995_ifu.txt"
pages_dir=f"{root}/data/texts/pages"
out=f"{root}/device_notes.md"

def read(path): 
    with open(path,encoding="utf-8",errors="ignore") as f: 
        return f.read()

def section(md,title):
    pat=re.compile(rf"^##\s*{re.escape(title)}\s*$",re.I|re.M)
    m=pat.search(md)
    if not m: return []
    start=m.end()
    nxt=re.search(r"^##\s*",md[start:],re.M)
    block=md[start: ] if not nxt else md[start:start+nxt.start()]
    return [l for l in block.strip().splitlines() if l.strip()]

def first_bullet_triplet(lines):
    for l in lines:
        m=re.match(r"-\s*p\.(\d+):(\d+):\s*(.+)",l.strip())
        if m: 
            return int(m.group(1)), int(m.group(2)), m.group(3).strip()
    return None

def gather_codes(lines):
    codes=set()
    for l in lines:
        for c in re.findall(r"\b[A-Z]{3}\b",l):
            if c not in {"CFR","FDA","QS","MDR"}:
                codes.add(c)
    return sorted(codes)

def gather_cfr(lines):
    regs=set()
    for l in lines:
        for r in re.findall(r"21\s*CFR\s*(\d+\.\d+)",l,flags=re.I):
            regs.add(r)
    return sorted(regs)

def gather_knums(md):
    kn=section(md,"k_numbers_found")
    out=[]
    for l in kn:
        m=re.match(r"-\s*p\.(\d+):(\d+):\s*(K\d{6})",l.strip(),re.I)
        if m:
            out.append(m.group(3).upper())
    out=sorted({k for k in out if k.upper()!="K101995"})
    return out

def find_device_class(pages_dir):
    hits=[]
    for p in sorted(glob.glob(f"{pages_dir}/K101995_p*.txt")):
        pg=int(re.search(r"_p(\d+)\.txt$",p).group(1))
        with open(p,encoding="utf-8",errors="ignore") as f:
            for i,l in enumerate(f,1):
                m=re.search(r"\bclass\s*([iIvVlL1]{1,3})\b",l)
                if m:
                    raw=m.group(1)
                    val=raw.upper().replace("LL","II").replace("11","II").replace("1","I").replace("L","I")
                    if val in {"I","II","III"}:
                        hits.append((pg,i,val))
    if hits:
        return sorted(hits)[0][2], hits[0][0]
    return None, None

md=read(fields)
ifu_txt=read(ifu).strip()
trade=first_bullet_triplet(section(md,"trade_or_device_name"))
common=first_bullet_triplet(section(md,"common_name"))
app_block=section(md,"applicant_block (starts")
mfr_block=section(md,"manufacturer_block (starts")
prod_codes=gather_codes(section(md,"product_code_lines"))
regs=gather_cfr(section(md,"regulation_lines"))
predicates=gather_knums(md)
dev_class, dev_class_page = find_device_class(pages_dir)

def safe_val(t,default=""):
    return t[2] if t else default
def safe_loc(t):
    return f"(p. {t[0]})" if t else ""

lines=[]
lines.append("K-number: K101995")
lines.append(f"Device/Trade name (exact): {safe_val(trade,'') } {safe_loc(trade)}".strip())
lines.append(f"Common name (exact): {safe_val(common,'')} {safe_loc(common)}".strip())
applicant_name = app_block[0][2:] if app_block else ""
lines.append(f"Manufacturer / Applicant (exact): {applicant_name} (see applicant block)")
ifu_page = 10
lines.append(f"Indications for Use (copy paragraph + page #):")
lines.append(f"\"{ifu_txt}\" (p. {ifu_page})")
lines.append(f"Predicate devices (list K-numbers + page #): {', '.join(predicates) if predicates else ''}")
lines.append(f"Product code(s): {', '.join(prod_codes)}")
lines.append(f"Regulation number: {', '.join([f'21 CFR {r}' for r in regs])}")
lines.append(f"Device class: {dev_class if dev_class else 'TBD'}{f' (p. {dev_class_page})' if dev_class_page else ''}")
lines.append("Other identifiers (models, catalog #s, etc.):")
lines.append("")
lines.append("Evidence snippets with page numbers:")
if trade: lines.append(f"- Name: \"{trade[2]}\" (p. {trade[0]})")
if common: lines.append(f"- Common name: \"{common[2]}\" (p. {common[0]})")
if applicant_name: lines.append(f"- Manufacturer/applicant: \"{applicant_name}\" (see applicant block)")
lines.append(f"- IFU: \"{ifu_txt[:120]}…\" (p. {ifu_page})")
if predicates: lines.append(f"- Predicates: {', '.join(predicates)}")
lines.append("")
lines.append("Ontology mapping to use (from 510k_Ontology.txt):")
lines.append("- Classes: Submission, MedicalDevice, PredicateDevice, Applicant")
lines.append("- Object properties: hasPredicateDevice, hasApplicant, concernsDevice")
lines.append("- Data properties: hasTradeName, hasCommonName, hasProductCode, hasRegulationNumber, hasIndicationsForUseText, hasIntendedUseText")
lines.append("- Submission data property: hasKNumber")
lines.append("")
pathlib.Path(out).write_text("\n".join(lines),encoding="utf-8")
print(out)
