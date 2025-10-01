import PyPDF2

def convert_pdf(name_of_file):
    with open("data/" + name_of_file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    with open(name_of_file + ".txt", "w", encoding="utf-8") as out:
        out.write(text)


if __name__ == "__main__":
    convert_pdf("K060065.pdf")