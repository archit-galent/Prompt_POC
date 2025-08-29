"""
Code indexer that chunks code files and stores embeddings in FAISS.
Handles function-level chunking and creates vector representations using OpenAI embeddings.
"""

import os
import ast
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from openai import OpenAI


class CodeChunk:
    """Represents a chunk of code with metadata."""
    
    def __init__(self, content: str, file_path: str, start_line: int, end_line: int, 
                 chunk_type: str = "function", name: str = ""):
        self.content = content
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_type = chunk_type  # "function", "class", "module", "block"
        self.name = name


class CodeIndexer:
    """Indexes code files by creating embeddings and storing them in FAISS."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the code indexer.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chunks: List[CodeChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        
    def _extract_functions_and_classes(self, content: str, file_path: str) -> List[CodeChunk]:
        """Extract functions and classes from Python code using AST."""
        chunks = []
        lines = content.split('\n')
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start_line + 10
                    
                    # Extract the actual code content
                    chunk_content = '\n'.join(lines[start_line-1:end_line])
                    
                    chunk_type = "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class"
                    
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type=chunk_type,
                        name=node.name
                    ))
                    
        except SyntaxError:
            # If AST parsing fails, fall back to simple line-based chunking
            return self._chunk_by_lines(content, file_path)
            
        return chunks
    
    def _chunk_by_lines(self, content: str, file_path: str, chunk_size: int = 100) -> List[CodeChunk]:
        """Fall back to line-based chunking when AST parsing fails."""
        lines = content.split('\n')
        chunks = []
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            chunks.append(CodeChunk(
                content=chunk_content,
                file_path=file_path,
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines)),
                chunk_type="block",
                name=f"block_{i+1}_{min(i + chunk_size, len(lines))}"
            ))
            
        return chunks
    
    def chunk_repository(self, repo_path: str) -> List[CodeChunk]:
        """
        Walk through repository and create code chunks.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            List of CodeChunk objects
        """
        repo_path = Path(repo_path)
        all_chunks = []
        
        # Supported file extensions
        extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.rb', '.go'}
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if file_path.suffix == '.py':
                        # Use AST for Python files
                        chunks = self._extract_functions_and_classes(content, str(file_path))
                    else:
                        # Use line-based chunking for other languages
                        chunks = self._chunk_by_lines(content, str(file_path))
                    
                    all_chunks.extend(chunks)
                    
                except (UnicodeDecodeError, PermissionError) as e:
                    print(f"Warning: Could not read file {file_path}: {e}")
                    continue
                    
        self.chunks = all_chunks
        return all_chunks
    
    def generate_embeddings(self, chunks: Optional[List[CodeChunk]] = None) -> np.ndarray:
        """
        Generate embeddings for code chunks using OpenAI API.
        
        Args:
            chunks: List of chunks to embed, defaults to self.chunks
            
        Returns:
            numpy array of embeddings
        """
        if chunks is None:
            chunks = self.chunks
            
        if not chunks:
            raise ValueError("No chunks to embed")
        
        # Prepare text for embedding
        texts = []
        for chunk in chunks:
            # Create descriptive text including metadata
            chunk_text = f"File: {chunk.file_path}\n"
            chunk_text += f"Type: {chunk.chunk_type}\n"
            if chunk.name:
                chunk_text += f"Name: {chunk.name}\n"
            chunk_text += f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            chunk_text += f"Content:\n{chunk.content}"
            texts.append(chunk_text)
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings in batches to avoid API limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                raise
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        self.embeddings = embeddings
        
        return embeddings
    
    def build_faiss_index(self, embeddings: Optional[np.ndarray] = None) -> faiss.IndexFlatIP:
        """
        Build FAISS index for similarity search.
        
        Args:
            embeddings: Embeddings array, defaults to self.embeddings
            
        Returns:
            FAISS index
        """
        if embeddings is None:
            embeddings = self.embeddings
            
        if embeddings is None:
            raise ValueError("No embeddings available to index")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        self.index = index
        print(f"Built FAISS index with {index.ntotal} vectors")
        
        return index
    
    def save_index(self, index_path: str, metadata_path: str):
        """
        Save FAISS index and chunk metadata to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save chunk metadata
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save chunk metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks, f)
            
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """
        Load FAISS index and chunk metadata from disk.
        
        Args:
            index_path: Path to the FAISS index
            metadata_path: Path to the chunk metadata
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load chunk metadata
        with open(metadata_path, 'rb') as f:
            self.chunks = pickle.load(f)
            
        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.chunks)} chunk metadata entries")
    
    def index_repository(self, repo_path: str, save_path: str = "./faiss_index"):
        """
        Complete indexing pipeline: chunk -> embed -> index -> save.
        
        Args:
            repo_path: Path to repository to index
            save_path: Base path for saving index files
        """
        print(f"Indexing repository: {repo_path}")
        
        # Step 1: Chunk the repository
        chunks = self.chunk_repository(repo_path)
        print(f"Created {len(chunks)} code chunks")
        
        # Step 2: Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Step 3: Build FAISS index
        index = self.build_faiss_index(embeddings)
        
        # Step 4: Save to disk
        index_path = f"{save_path}.index"
        metadata_path = f"{save_path}.metadata"
        self.save_index(index_path, metadata_path)
        
        return index, chunks


if __name__ == "__main__":
    # Example usage
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    indexer = CodeIndexer(api_key)
    
    # Index a repository
    repo_path = "./test_repo"  # Replace with actual repo path
    indexer.index_repository(repo_path)
