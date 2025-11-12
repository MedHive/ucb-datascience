import sys, re
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("usage: python3 split_text_by_page_generic.py <input_txt> <out_dir> [prefix]")
        sys.exit(2)

    in_txt = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) > 3 else in_txt.stem

    text = in_txt.read_text(encoding="utf-8", errors="ignore")

    parts = re.split(r'\f|\x0c', text)
    if len(parts) == 1:
        parts = re.split(r'\n\s*Page\s+\d+(?:\s+of\s+\d+)?\s*\n', text)

    if len(parts) == 1:
        parts = [text]

    out_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(parts))))
    for i, content in enumerate(parts, start=1):
        p = out_dir / f"{prefix}_p{str(i).zfill(width)}.txt"
        p.write_text(content, encoding="utf-8")

    print(len(parts))

if __name__ == "__main__":
    main()
