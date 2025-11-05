import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict

class RAGEngine:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        # Load embedding model (lightweight, fast)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Embedding model loaded.")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        print(f"[RAG] Extracting text from {pdf_path}...")
        text = ""
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            print(f"[RAG] PDF has {page_count} pages")
            
            for page_num, page in enumerate(doc):
                text += page.get_text()
                if (page_num + 1) % 10 == 0:
                    print(f"[RAG] Processed {page_num + 1}/{page_count} pages")
            
            doc.close()
            print(f"[RAG] Text extraction complete. Total characters: {len(text)}")
        except Exception as e:
            print(f"[RAG ERROR] Error extracting text from {pdf_path}: {e}")
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        print(f"[RAG] Chunking text with size={self.chunk_size}, overlap={self.chunk_overlap}...")
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        print(f"[RAG] Created {len(chunks)} chunks from {len(words)} words")
        return chunks
    
    def process_pdfs(self, pdf_paths: List[str]):
        """Process multiple PDFs: extract, chunk, embed, index."""
        all_text = ""
        
        # Extract text from all PDFs
        for pdf_path in pdf_paths:
            print(f"[RAG] Processing {pdf_path}...")
            text = self.extract_text_from_pdf(pdf_path)
            all_text += "\n\n" + text
        
        # Chunk the combined text
        self.chunks = self.chunk_text(all_text)
        print(f"[RAG] Created {len(self.chunks)} chunks total")
        
        if not self.chunks:
            raise ValueError("No text extracted from PDFs.")
        
        # Generate embeddings
        print(f"[RAG] Generating embeddings for {len(self.chunks)} chunks...")
        self.embeddings = self.embedding_model.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        print(f"[RAG] Embeddings generated. Shape: {self.embeddings.shape}")
        
        # Normalize embeddings for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(self.embeddings)
        print(f"[RAG] Embeddings normalized")
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        self.index.add(self.embeddings)
        
        print(f"[RAG] FAISS index built with {self.index.ntotal} vectors (dimension: {dimension})")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k most relevant chunks for a query."""
        print(f"\n[RAG] Retrieving chunks for query: {query[:50]}...")
        
        if self.index is None or not self.chunks:
            print(f"[RAG ERROR] Index or chunks not available!")
            return []
        
        # Embed query
        print(f"[RAG] Encoding query...")
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        print(f"[RAG] Query embedding generated, shape: {query_embedding.shape}")
        
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        print(f"[RAG] Searching FAISS index for top-{top_k} results...")
        scores, indices = self.index.search(query_embedding, top_k)
        print(f"[RAG] FAISS search completed")
        print(f"[RAG] Relevance scores: {scores[0]}")
        print(f"[RAG] Chunk indices: {indices[0]}")
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'text': self.chunks[idx],
                    'score': float(score)
                })
        
        print(f"[RAG] Returning {len(results)} results")
        return results
    
    def reset(self):
        """Clear all processed data."""
        print(f"[RAG] Resetting RAG engine...")
        self.chunks = []
        self.embeddings = None
        self.index = None
        print(f"[RAG] Reset complete")
