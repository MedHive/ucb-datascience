import re, csv, time, pathlib
root="projectA_meddevice_KG"
fields=f"{root}/data/extracts/K101995_fields.md"
ifu=f"{root}/data/extracts/K101995_ifu.txt"
out=pathlib.Path(f"{root}/data/triples"); out.mkdir(parents=True,exist_ok=True)
csv_path=out/"K101995_triples.csv"
md=open(fields,encoding="utf-8",errors="ignore").read()
ifu_txt=open(ifu,encoding="utf-8",errors="ignore").read().strip()
def bullets_after(header_pat):
    m=re.search(header_pat,md,re.I)
    if not m: return []
    s=m.end()
    n=re.search(r"^##\s",md[s:],re.M)
    block=md[s:] if not n else md[s:s+n.start()]
    return [l.strip()[2:].strip() for l in block.strip().splitlines() if l.strip().startswith("- ")]
def clean_bullet(txt):
    return re.sub(r"^p\.\d+:\d+:\s*","",txt).strip()
trade_lines=bullets_after(r"##\s*trade_or_device_name")
trade=clean_bullet(trade_lines[0]) if trade_lines else "Capnostream20p"
common_lines=bullets_after(r"##\s*common_name")
common=clean_bullet(common_lines[0]) if common_lines else ""
app_lines=bullets_after(r"##\s*applicant_block")
applicant=app_lines[0].strip() if app_lines else ""
prod_lines=bullets_after(r"##\s*product_code_lines")
prod_codes=sorted({c for l in prod_lines for c in re.findall(r"\b[A-Z]{3}\b",l) if c not in {"CFR","FDA","QS","MDR"}})
reg_lines=bullets_after(r"##\s*regulation_lines")
reg_nums=sorted({f"21 CFR {r}" for l in reg_lines for r in re.findall(r"21\s*CFR\s*(\d+\.\d+)",l,re.I)})
knums_lines=bullets_after(r"##\s*k_numbers_found")
predicates=sorted({k for l in knums_lines for k in re.findall(r"\bK\d{6}\b",l) if k!="K101995"})
ts=time.strftime("%Y-%m-%dT%H:%M:%S")
rows=[]
def add(s,p,o,src,evid):
    rows.append([s,p,o,src,evid,ts,ts])
submission="Submission:K101995"
device=f"Device:{trade}"
src="K101995.pdf"
add(submission,"hasKNumber","K101995",src,"cover page")
add(submission,"concernsDevice",device,src,"p.1 device name line")
if applicant:
    add(submission,"hasApplicant",f"Applicant:{applicant}",src,"applicant block p.1")
add(device,"hasTradeName",trade,src,"p.1 name")
if common:
    add(device,"hasCommonName",common,src,"p.1 common name")
for c in prod_codes:
    add(device,"hasProductCode",c,src,"product code lines")
for r in reg_nums:
    add(device,"hasRegulationNumber",r,src,"regulation lines")
if ifu_txt:
    add(device,"hasIndicationsForUseText",ifu_txt,src,"IFU p.10")
for k in predicates:
    add(submission,"hasPredicateDevice",f"Predicate:{k}",src,"predicate section")
with open(csv_path,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["subject","predicate","object","source_documents","intext_evidence","creation_timestamp","last_updated_timestamp"])
    w.writerows(rows)
print(csv_path)
