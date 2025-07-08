# Document RAG System

A document retrieval system that uses embeddings for semantic search across PDF and DOCX files.

## Overview

This system implements a simple RAG (Retrieval-Augmented Generation) pipeline that indexes documents and enables semantic search using Google Gemini embeddings stored in PostgreSQL.

## Requirements

```
psycopg2-binary
google-generativeai
python-docx
python-dotenv
pypdf
numpy
```

## Setup

1. Install dependencies:
```bash
pip install psycopg2-binary google-generativeai python-docx python-dotenv pypdf numpy
```

2. Create `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
POSTGRES_URL=postgresql://user:password@host:port/database
```

3. Add `.env` to `.gitignore`:
```
.env
```

## Quick Start

1. Place your PDF/DOCX documents in the project folder or create a documents subfolder
2. Run the indexing script  
3. Use the search functionality

## Usage

### Document Indexing

Open terminal and run:

```bash
python -i index_documents.py
```

Then use the functions:

```python
index_documents_jupyter('.', 'sentence')
check_database_status()
```

### Document Search

Open terminal and run:

```bash
python -i search_documents.py
```

Then search:

```python
search_documents_jupyter('machine learning', 5)
```

## Running the Scripts

### Interactive Mode (Recommended)
```bash
# For indexing
python -i index_documents.py
>>> index_documents_jupyter('.', 'sentence')
>>> check_database_status()

# For searching  
python -i search_documents.py
>>> search_documents_jupyter('your query', 5)
>>> test_search_system()
```

### Alternative: Python Console
```bash
python
>>> from index_documents import index_documents_jupyter
>>> index_documents_jupyter('.', 'sentence')
```

## Text Splitting Strategies

- **Fixed**: Fixed-size chunks with overlap
- **Sentence**: Groups sentences until word limit  
- **Paragraph**: Combines multiple paragraphs per chunk

## Database Schema

```sql
CREATE TABLE embeddings (
    id VARCHAR(36) PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding FLOAT[] NOT NULL,
    filename VARCHAR(255) NOT NULL,
    split_strategy VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Architecture

1. **Text Extraction**: Supports PDF and DOCX files
2. **Text Chunking**: Three different splitting strategies
3. **Embedding Generation**: Uses Google Gemini API
4. **Vector Storage**: PostgreSQL with FLOAT[] arrays
5. **Similarity Search**: Cosine similarity for retrieval

## File Structure

```
├── index_documents.py    # Document indexing module
├── search_documents.py   # Document search module
├── README.md            # This file
├── .env                 # Environment variables (create this)
└── .gitignore           # Git ignore file
```

## Notes

- Database table is created automatically
- Uses Google Gemini for embeddings  
- Returns top-5 similar chunks by default
