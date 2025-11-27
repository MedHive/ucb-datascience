import spacy
import spacy_component
from datetime import datetime
import os
import PyPDF2
import re
import csv

def convert_pdf(name_of_file):
    with open("data/" + name_of_file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    with open("data/" + name_of_file + ".txt", "w", encoding="utf-8") as out:
        out.write(text)


def NER_RE():
    # Load spaCy and REBEL
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("rebel", after="senter", config={
        'device': 0,  # GPU, -1 for CPU
        'model_name': 'Babelscape/rebel-large'
    })

    # Folder containing documehts
    folder_path = "data"
    all_str = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # only process text files
            stri = ""
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Process the document
            doc = nlp(text)
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            print(f"\nProcessing file: {filename}\n")
            
            # Extract triples
            for key, rel_dict in doc._.rel.items():
                start_token, end_token = key
                
                # Ensure start_token <= end_token
                token_start = min(start_token, end_token)
                token_end = max(start_token, end_token)
                
                # Character indices
                char_start = doc[token_start].idx
                char_end = doc[token_end-1].idx + len(doc[token_end-1])
                
                # Get sentence as context
                span = doc.char_span(char_start, char_end)
                if span is not None:
                    context_text = span.sent.text
                else:
                    context_text = doc[token_start:token_end].text
                
                # Add new keys
                rel_dict['source_document'] = filename
                rel_dict['context'] = context_text
                rel_dict['creation_timestamp'] = timestamp
                
                # Print result
                print(f"{key}: {rel_dict}")
                stri = stri + f"{key}: {rel_dict}"
            all_str.append(stri)
    return all_str


def process_big_str(text):
    pattern = re.compile(
        r"\(\d+,\s*\d+\):\s*\{"
        r"\s*'relation':\s*'(?P<relation>[^']+)'[, ]*"
        r"'head_span':\s*(?P<head>[^,]+?),\s*"
        r"'tail_span':\s*(?P<tail>[^,]+?),\s*"
        r"'source_document':\s*'(?P<source_document>[^']+)'[, ]*"
        r"'context':\s*'(?P<context>.*?)'[, ]*"
        r"'creation_timestamp':\s*'(?P<timestamp>[^']+)'"
        r"\s*\}",
        re.DOTALL
    )

    edges = []
    for match in pattern.finditer(text):
        edges.append({
            'head_span': match.group('head').strip(),
            'tail_span': match.group('tail').strip(),
            'relation': match.group('relation').strip(),
            'context': match.group('context').strip(),
            'source_document': match.group('source_document').strip(),
            'creation_timestamp': match.group('timestamp').strip()
        })

    nodes = sorted(set([e['head_span'] for e in edges] + [e['tail_span'] for e in edges]))

    with open('nodes.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name'])
        for i, node in enumerate(nodes):
            writer.writerow([i, node])

    node_id_map = {name: i for i, name in enumerate(nodes)}

    with open('edges.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'relation', 'context', 'source_document', 'creation_timestamp'])
        for e in edges:
            writer.writerow([
                node_id_map[e['head_span']],
                node_id_map[e['tail_span']],
                e['relation'],
                e['context'],
                e['source_document'],
                e['creation_timestamp']
            ])

if __name__ == "__main__":
    folder_path="data"
    for filename in os.listdir("data"):
        if filename.endswith(".pdf"):
            convert_pdf(filename)
            break
    big_str_list = NER_RE()
    big_str = "\n".join(big_str_list)
    process_big_str(big_str)




