import os
from typing import List
from sentence_transformers import SentenceTransformer
import pinecone

# Initialize the Hugging Face embedding model
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
VECTOR_DIM = 384  # all-MiniLM-L6-v2 outputs 384-dimensional vectors

# Pinecone initialization (expects PINECONE_API_KEY and PINECONE_ENV in .env)
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')
index_name = os.getenv('PINECONE_INDEX')

pc = pinecone.Pinecone(
        api_key="pcsk_2aomp7_R5nHqNt1bXZrwHf1jB5cFt4pMfTxnxx9EJu3DL3bTmYwsA1x2mZfKgVhPuUoGNL"
)
# Ensure index exists
index = pc.Index("ent-kb-adk-index")

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of chunk_size with overlap.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_kb_from_file(file_path: str = 'ent_kb.txt'):
    """
    Reads lines from ENT_KB.txt in 'question|||answer' format, embeds each (question + answer), and upserts to Pinecone.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        print("ENT_KB.txt is empty.")
        return
    questions, answers, texts = [], [], []
    for line in lines:
        if '|||' not in line:
            continue
        q, a = line.split('|||', 1)
        q, a = q.strip(), a.strip()
        questions.append(q)
        answers.append(a)
        texts.append(f"Q: {q}\nA: {a}")
    embeddings = model.encode(texts, show_progress_bar=True)
    vectors = [
        (f"kb-{i}", emb.tolist(), {'question': q, 'answer': a})
        for i, (q, a, emb) in enumerate(zip(questions, answers, embeddings))
    ]
    index.upsert(vectors)
    print(f"Upserted {len(vectors)} Q&A pairs to Pinecone.")

def search_kb(query: str, top_k: int = 5) -> List[str]:
    """
    Embeds the query and searches Pinecone for similar chunks.
    Returns the top_k most similar texts.
    """
    query_emb = model.encode([query])[0]
    result = index.query(vector=[query_emb.tolist()], top_k=top_k, include_metadata=True)
    matches = result['matches']
    print(matches)
    return [match['metadata'] for match in matches]

if __name__ == "__main__":
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Vector dimension: {VECTOR_DIM}")
    # Example usage:
    # ingest_kb_from_file()
    print(search_kb("hoarse voice for over two weeks"))
