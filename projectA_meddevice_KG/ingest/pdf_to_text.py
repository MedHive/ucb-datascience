import sys, os
from pdfminer.high_level import extract_text
src = sys.argv[1] if len(sys.argv)>1 else "projectA_meddevice_KG/data/raw_pdfs/K101995.pdf"
dst = sys.argv[2] if len(sys.argv)>2 else "projectA_meddevice_KG/data/texts/K101995.txt"
os.makedirs(os.path.dirname(dst), exist_ok=True)
text = extract_text(src)
with open(dst, "w", encoding="utf-8") as f:
    f.write(text)
