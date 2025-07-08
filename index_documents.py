import os
import re
import uuid
from pathlib import Path
import psycopg2
from dotenv import load_dotenv
from datetime import datetime



# PDF extraction with fallback
try:
    from pdfminer.high_level import extract_text as extract_pdf
except ImportError:
    from pypdf import PdfReader
    def extract_pdf(path: str) -> str:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
        return "\n".join(texts)

from docx import Document
import google.generativeai as genai



# Load environment variables
load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')
DB_URL = os.getenv('POSTGRES_URL')

if not API_KEY or not DB_URL:
    raise RuntimeError('Missing GEMINI_API_KEY or POSTGRES_URL in .env')

genai.configure(api_key=API_KEY)

BASE_DIR = Path.cwd()


def get_conn():
    """Establish database connection"""
    return psycopg2.connect(DB_URL)


def embed_text(text: str) -> list[float]:
    """Generate text embedding using Gemini API"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

        
def clean_text(text: str) -> str:
    """Clean and normalize text whitespace"""
    return re.sub(r"\s+", " ", text).strip()


def fixed_chunks(text: str, size: int = 300, overlap: int = 50):
    """Fixed-size chunking with overlap"""
    words = text.split()
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i+size])
        if chunk.strip():
            yield chunk

            
def sentence_chunks(text: str, limit: int = 300):
    """Sentence-based splitting"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    buf, count = [], 0
    for s in sentences:
        if not s.strip():
            continue
        tok = len(s.split())
        if count + tok > limit and buf:
            yield ' '.join(buf)
            buf, count = [], 0
        buf.append(s)
        count += tok
    if buf:
        yield ' '.join(buf)

        
def paragraph_chunks(text: str, max_paragraphs: int = 3):
    """Paragraph-based splitting"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    for i in range(0, len(paragraphs), max_paragraphs):
        chunk = '\n\n'.join(paragraphs[i:i+max_paragraphs])
        if chunk.strip():
            yield chunk

METHODS = {
    'fixed': fixed_chunks, 
    'sentence': sentence_chunks,
    'paragraph': paragraph_chunks
}

def extract_file(path: Path) -> str:
    """Extract text from PDF or DOCX file"""
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        return extract_pdf(str(path))
    elif suffix == '.docx':
        doc = Document(path)
        return '\n'.join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f'Unsupported file type: {suffix}')


def create_table():
    """Create embeddings table if not exists"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id VARCHAR(36) PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            embedding FLOAT[] NOT NULL,
            filename VARCHAR(255) NOT NULL,
            split_strategy VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()


def index_folder(folder: Path, method: str):
    """
    Index documents from folder using specified chunking method.
    
    Process:
    1. Extract text from PDF/DOCX files
    2. Split text into chunks using selected strategy
    3. Generate embeddings for each chunk
    4. Store chunks and embeddings in PostgreSQL
    """
    folder_path = BASE_DIR / folder
    if not folder_path.exists():
        raise ValueError(f"Folder {folder_path} does not exist")
    
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}', choose from {list(METHODS)}")
    
    # Create table if not exists
    create_table()
    
    splitter = METHODS[method]
    conn = get_conn()
    cur = conn.cursor()
    
    processed_files = 0
    total_chunks = 0
    
    try:
        print(f"Processing files in: {folder_path}")
        
        for file in folder_path.rglob('*'):
            if file.suffix.lower() not in ('.pdf', '.docx'):
                continue
            
            print(f"Processing: {file.name}")
            
            raw_text = clean_text(extract_file(file))
            print(f"   Text length: {len(raw_text)} characters")
            
            file_chunks = 0
            for chunk in splitter(raw_text):
                if not chunk.strip():
                    continue
                
                embedding_vector = embed_text(chunk)
                
                cur.execute(
                    """INSERT INTO embeddings(id, chunk_text, embedding, filename, split_strategy, created_at) 
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        str(uuid.uuid4()),
                        chunk,
                        embedding_vector,
                        str(file.relative_to(BASE_DIR)),
                        method,
                        datetime.now()
                    )
                )
                file_chunks += 1
                total_chunks += 1
            
            processed_files += 1
            print(f"   Created {file_chunks} chunks")
            
        conn.commit()
        print(f"\nIndexing completed successfully")
        print(f"Files processed: {processed_files}")
        print(f"Total chunks: {total_chunks}")
        print(f"Split strategy: {method}")
        
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def index_documents_jupyter(folder_name: str, method: str = 'sentence'):
    """Convenience function for Jupyter Notebook usage"""
    folder_path = Path(folder_name)
    index_folder(folder_path, method)
    

def check_database_status():
    """Check current database status"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM embeddings")
        total = cur.fetchone()[0]
        print(f"Total chunks in database: {total}")
        
        cur.execute("""
            SELECT filename, split_strategy, COUNT(*) 
            FROM embeddings 
            GROUP BY filename, split_strategy 
            ORDER BY filename, split_strategy
        """)
        results = cur.fetchall()
        
        if results:
            print("\nBreakdown by file:")
            for filename, strategy, count in results:
                print(f"   {filename} ({strategy}): {count} chunks")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error checking database: {e}")

print("Index documents module loaded successfully")
print("Usage: index_documents_jupyter('folder_name', 'method')")
print("Available methods: 'fixed', 'sentence', 'paragraph'")
print("Check status: check_database_status()")

