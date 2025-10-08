from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
from pathlib import Path

EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
LLM_MODEL = 'google/flan-t5-base'
VECTOR_DB = []

print("loading embedding model")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("loading language model")
llm = pipeline('text2text-generation', model=LLM_MODEL)

def load_dataset(data_path='ucb-datascience/RE/K060065.pdf.txt'):
    """Load dataset from text file"""
    dataset = []

    if os.path.isfile(data_path):
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            # chunking
            chunks = [line.strip() for line in content.split('\n') if line.strip()]
            dataset.extend(chunks)
    else:
        print(f"Warning: {data_path} not found")

    return dataset

def add_chunk_to_database(chunk):
    """Add a text chunk and its embedding to the vector database."""
    embedding = embedder.encode(chunk).tolist()
    VECTOR_DB.append((chunk, embedding))

def build_vector_database(dataset):
    """Build the vector database from the dataset."""
    print(f"Building vector database with {len(dataset)} chunks...")
    for i, chunk in enumerate(dataset):
        if i % 50 == 0:
            print(f"Processing chunk {i}/{len(dataset)}")
        add_chunk_to_database(chunk)
    print("Vector database built successfully!")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    """Retrieve the most relevant chunks for a query."""
    query_embedding = embedder.encode(query).tolist()
    similarities = []

    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in similarities[:top_n]] #returns top n

def generate_response(query, context):
    """Generate a response using the LLM with retrieved context."""
    prompt = f"""Answer the question based on the context below.
                 Context:
                 {context}

                 Question: {query}

                 Answer:"""

    response = llm(prompt, max_length=200, do_sample=False)
    return response[0]['generated_text']

def rag_query(query, top_n=3):
    """Main RAG function"""
    print(f"\nQuery: {query}")
    print("Retrieving relevant context...")

    # Retrieve relevant chunks
    relevant_chunks = retrieve(query, top_n=top_n)
    context = "\n".join(relevant_chunks)

    print(f"Retrieved {len(relevant_chunks)} relevant chunks")
    print("\nGenerating response...")

    # Generate response
    response = generate_response(query, context)

    return response, relevant_chunks

def main():
    """Main function to run the RAG pipeline."""
    print("Step 1: Loading dataset")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} chunks\n")

    # vector database
    print("Step 2: Building vector database")
    build_vector_database(dataset)
    print()

    # interactive loop
    print("RAG pipeline ready")
    print("You can now ask questions about the document!")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        query = input("Enter your question: ").strip()

        if query.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            response, chunks = rag_query(query)
            print(f"\nAnswer: {response}\n")
            print("-" * 80)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
