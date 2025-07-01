from sentence_transformers import SentenceTransformer
import psycopg2
from typing import List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
        self.vector_dimensions = 384

    def get_embedding(self, text: str) -> List[float]:
        """Convert text to embedding vector"""
        return self.model.encode(text).tolist()

    def store_qa_pair(self, question: str, answer: str) -> None:
        """Store question-answer pair in PostgreSQL vector database"""
        try:
            conn = psycopg2.connect(
                dbname=os.getenv('PG_DBNAME'),
                user=os.getenv('PG_USER'),
                password=os.getenv('PG_PASSWORD'),
                host=os.getenv('PG_HOST'),
                port=os.getenv('PG_PORT')
            )
            cur = conn.cursor()
            
            question_embedding = self.get_embedding(question)
            answer_embedding = self.get_embedding(answer)
            
            cur.execute("""
                INSERT INTO qa_embeddings (question, question_embedding, answer, answer_embedding)
                VALUES (%s, %s, %s, %s)
            """, (question, question_embedding, answer, answer_embedding))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            print(f"Error storing QA pair: {e}")
            raise

    def search_similar_questions(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for similar questions in the database"""
        try:
            conn = psycopg2.connect(
                dbname=os.getenv('PG_DBNAME'),
                user=os.getenv('PG_USER'),
                password=os.getenv('PG_PASSWORD'),
                host=os.getenv('PG_HOST'),
                port=os.getenv('PG_PORT')
            )
            cur = conn.cursor()
            
            query_embedding = self.get_embedding(query)
            
            cur.execute("""
                SELECT question, answer, 
                (question_embedding <=> %s) as similarity
                FROM qa_embeddings
                ORDER BY similarity ASC
                LIMIT %s
            """, (query_embedding, top_k))
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            return results
            
        except Exception as e:
            print(f"Error searching similar questions: {e}")
            raise
