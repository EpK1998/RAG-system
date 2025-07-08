import os
import numpy as np
from pathlib import Path
import psycopg2
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')
DB_URL = os.getenv('POSTGRES_URL')

if not API_KEY or not DB_URL:
    raise RuntimeError('Missing GEMINI_API_KEY or POSTGRES_URL in .env')

genai.configure(api_key=API_KEY)

def get_conn():
    """Establish database connection"""
    return psycopg2.connect(DB_URL)

def embed_query(query: str) -> list[float]:
    """Generate embedding for search query using Gemini API"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        raise

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(vec1)
    b = np.array(vec2)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def search_documents(query: str, limit: int = 5) -> list[dict]:
    """
    Search for documents similar to query using cosine similarity.
    
    Process:
    1. Generate embedding for user query using Gemini API
    2. Compare query vector to all document vectors in database
    3. Calculate cosine similarity for each comparison
    4. Return top-k most similar document chunks
    """
    try:
        print(f"Searching for: '{query}'")
        query_embedding = embed_query(query)
        print(f"Generated query embedding (size: {len(query_embedding)})")
        
        conn = get_conn()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, chunk_text, embedding, filename, split_strategy, created_at 
            FROM embeddings
        """)
        
        all_chunks = cur.fetchall()
        print(f"Found {len(all_chunks)} chunks in database")
        
        if not all_chunks:
            print("No chunks found in database")
            return []
        
        similarities = []
        for chunk_id, chunk_text, embedding_array, filename, split_strategy, created_at in all_chunks:
            # Convert PostgreSQL array to list
            chunk_embedding = list(embedding_array)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            
            similarities.append({
                'id': chunk_id,
                'chunk_text': chunk_text,
                'filename': filename,
                'split_strategy': split_strategy,
                'created_at': created_at,
                'similarity': similarity
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top results
        top_results = similarities[:limit]
        
        print(f"\nTop {len(top_results)} results:")
        print("-" * 60)
        
        for i, result in enumerate(top_results, 1):
            print(f"\n{i}. Similarity: {result['similarity']:.4f}")
            print(f"   File: {result['filename']}")
            print(f"   Strategy: {result['split_strategy']}")
            print(f"   Text: {result['chunk_text'][:150]}...")
            if len(result['chunk_text']) > 150:
                print(f"   Full length: {len(result['chunk_text'])} characters")
        
        cur.close()
        conn.close()
        
        return top_results
        
    except Exception as e:
        print(f"Search error: {e}")
        raise

def search_documents_jupyter(query: str, limit: int = 5):
    """Convenience function for Jupyter Notebook usage"""
    return search_documents(query, limit)

def test_search_system():
    """Test search system with sample queries"""
    test_queries = [
        "artificial intelligence",
        "machine learning", 
        "data processing",
        "algorithm",
        "technology"
    ]
    
    print("Testing search system")
    print("=" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        try:
            results = search_documents(query, limit=3)
            if results:
                print(f"Found {len(results)} results")
                print(f"Best similarity: {results[0]['similarity']:.4f}")
            else:
                print("No results found")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 25)

print("Search documents module loaded successfully")
print("Usage: search_documents_jupyter('your_query', 5)")
print("Test system: test_search_system()")
