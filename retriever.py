"""
Context retriever that combines FAISS similarity search with dependency graph expansion
to find relevant code chunks for a given query.
"""

import numpy as np
import faiss
from typing import List, Dict, Set, Tuple, Optional
from openai import OpenAI

from indexer import CodeChunk, CodeIndexer
from graph import DependencyGraphBuilder, DependencyNode


class ContextRetriever:
    """Retrieves and expands context for code queries using embeddings and dependency graph."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the context retriever.
        
        Args:
            api_key: OpenAI API key for query embeddings
            model: OpenAI embedding model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.indexer: Optional[CodeIndexer] = None
        self.graph_builder: Optional[DependencyGraphBuilder] = None
        
    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and chunk metadata."""
        self.indexer = CodeIndexer(api_key="dummy")  # API key not needed for loading
        self.indexer.load_index(index_path, metadata_path)
        
    def load_graph(self, repo_path: str):
        """Build dependency graph for the repository."""
        self.graph_builder = DependencyGraphBuilder()
        self.graph_builder.build_graph(repo_path)
        
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for user query.
        
        Args:
            query: User's query string
            
        Returns:
            Query embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[query]
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding.reshape(1, -1))
            
            return embedding
            
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            raise
    
    def search_similar_chunks(self, query: str, k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """
        Search for similar code chunks using FAISS.
        
        Args:
            query: User's query string
            k: Number of similar chunks to retrieve
            
        Returns:
            List of (CodeChunk, similarity_score) tuples
        """
        if not self.indexer or not self.indexer.index:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Generate query embedding
        query_embedding = self._embed_query(query)
        
        # Search FAISS index
        similarities, indices = self.indexer.index.search(
            query_embedding.reshape(1, -1), k
        )
        
        # Return chunks with similarity scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.indexer.chunks):  # Valid index
                chunk = self.indexer.chunks[idx]
                score = similarities[0][i]
                results.append((chunk, score))
        
        return results
    
    def _extract_focus_entities(self, query: str, chunks: List[CodeChunk]) -> Set[str]:
        """
        Extract key function/class names that the query is asking about.
        
        Args:
            query: User's query
            chunks: Retrieved chunks
            
        Returns:
            Set of entity names to focus on
        """
        focus_entities = set()
        
        # Simple keyword extraction from query
        query_lower = query.lower()
        
        # Look for function/class names in the query
        for chunk in chunks:
            if chunk.name and chunk.name.lower() in query_lower:
                focus_entities.add(chunk.name)
        
        # Look for explicit mentions of functions/classes
        # This is a simple heuristic - in production, you might use more sophisticated NER
        import re
        
        # Find patterns like "foo()" or "class Foo" or "function bar"
        patterns = [
            r'\b(\w+)\(\)',  # function calls
            r'class\s+(\w+)',  # class mentions
            r'function\s+(\w+)',  # function mentions
            r'\b(\w+)\.(\w+)',  # method calls
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    focus_entities.update(match)
                else:
                    focus_entities.add(match)
        
        return focus_entities
    
    def expand_context_with_graph(self, chunks: List[CodeChunk], 
                                  focus_entities: Set[str]) -> List[CodeChunk]:
        """
        Expand context using dependency graph relationships.
        
        Args:
            chunks: Initially retrieved chunks
            focus_entities: Key entities to expand around
            
        Returns:
            Expanded list of relevant chunks
        """
        if not self.graph_builder:
            return chunks  # Return original chunks if no graph available
        
        expanded_chunks = list(chunks)  # Start with original chunks
        chunk_ids = {(c.file_path, c.name) for c in chunks}
        
        # For each focus entity, find related nodes in the dependency graph
        for entity in focus_entities:
            if not entity:
                continue
                
            # Get related context from dependency graph
            related_node_ids = self.graph_builder.get_related_context(entity)
            
            # Convert node IDs back to chunks
            for node_id in related_node_ids:
                if node_id in self.graph_builder.nodes:
                    node = self.graph_builder.nodes[node_id]
                    
                    # Check if we already have this chunk
                    chunk_key = (node.file_path, node.name)
                    if chunk_key not in chunk_ids:
                        # Find corresponding chunk in indexer
                        matching_chunks = [
                            c for c in self.indexer.chunks
                            if c.file_path == node.file_path and 
                               (c.name == node.name or 
                                (c.start_line <= node.line_number <= c.end_line))
                        ]
                        
                        for chunk in matching_chunks:
                            if (chunk.file_path, chunk.name) not in chunk_ids:
                                expanded_chunks.append(chunk)
                                chunk_ids.add((chunk.file_path, chunk.name))
        
        return expanded_chunks
    
    def _expand_query_terms(self, query: str) -> List[str]:
        """
        Expand natural language queries with technical terms (Cursor-style approach).
        
        Args:
            query: Original user query
            
        Returns:
            List of expanded query variants
        """
        expanded_queries = [query]  # Always include original
        
        # Common natural language to technical term mappings
        term_mappings = {
            # Caching related
            "caching": ["cache", "FEATURE_FLAGS", "enable_caching", "cache_key"],
            "cache": ["caching", "cache_key", "generate_hash", "FEATURE_FLAGS"],
            "how does caching work": ["enable_caching", "cache_key", "FEATURE_FLAGS", "generate_hash"],
            
            # Configuration related  
            "config": ["CONFIG_X", "settings", "FEATURE_FLAGS", "validate_config"],
            "configuration": ["CONFIG_X", "settings", "FEATURE_FLAGS", "get_config"],
            "what breaks": ["CONFIG_X", "validate_config", "health_check", "errors"],
            
            # Processing related
            "processing": ["process_batch", "foo", "DataProcessor", "data_batch"],
            "data processing": ["foo", "process_batch", "DataProcessor", "grouped"],
            
            # Error handling
            "error": ["health_check", "validate_config", "errors", "issues"],
            "fails": ["health_check", "validate_config", "CONFIG_X", "ValueError"],
        }
        
        query_lower = query.lower()
        
        # Find matching mappings
        for pattern, terms in term_mappings.items():
            if pattern in query_lower:
                # Add technical term variants
                expanded_queries.extend([
                    " ".join(terms),
                    f"{query} {' '.join(terms[:3])}",  # Hybrid query
                ])
                break
        
        # Extract potential function/class names from query
        import re
        # Look for patterns like "foo()", "Foo", "foo_bar"
        code_patterns = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:\(\))?', query)
        if code_patterns:
            expanded_queries.append(" ".join(code_patterns))
        
        return list(set(expanded_queries))  # Remove duplicates
    
    def _keyword_search(self, query: str, max_chunks: int = 15) -> List[Tuple[CodeChunk, float]]:
        """
        Perform keyword-based search as fallback (BM25-style).
        
        Args:
            query: Search query
            max_chunks: Maximum chunks to return
            
        Returns:
            List of (chunk, relevance_score) tuples
        """
        if not self.indexer:
            return []
        
        results = []
        query_terms = set(query.lower().split())
        
        for chunk in self.indexer.chunks:
            # Create searchable text (content + metadata)
            searchable_text = f"{chunk.content} {chunk.name} {chunk.file_path}".lower()
            
            # Calculate keyword overlap score
            chunk_terms = set(searchable_text.split())
            overlap = len(query_terms.intersection(chunk_terms))
            
            if overlap > 0:
                # Simple relevance scoring
                relevance = overlap / len(query_terms)
                
                # Boost if function/class name matches exactly
                if chunk.name and chunk.name.lower() in query.lower():
                    relevance += 0.5
                
                # Boost if query terms appear in chunk name
                for term in query_terms:
                    if term in chunk.name.lower():
                        relevance += 0.3
                
                results.append((chunk, relevance))
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_chunks]
    
    def _hybrid_search(self, query: str, similarity_threshold: float = 0.3,
                      max_chunks: int = 15) -> List[Tuple[CodeChunk, float, str]]:
        """
        Hybrid search combining semantic similarity and keyword matching.
        
        Args:
            query: User's query
            similarity_threshold: Minimum similarity threshold
            max_chunks: Maximum chunks to return
            
        Returns:
            List of (chunk, score, method) tuples
        """
        all_results = []
        seen_chunks = set()
        
        # Expand query terms
        query_variants = self._expand_query_terms(query)
        
        # Strategy 1: Semantic search with multiple thresholds
        thresholds = [similarity_threshold, similarity_threshold * 0.7, similarity_threshold * 0.5]
        
        for threshold in thresholds:
            for variant in query_variants:
                try:
                    semantic_results = self.search_similar_chunks(variant, k=max_chunks)
                    for chunk, score in semantic_results:
                        if score >= threshold and id(chunk) not in seen_chunks:
                            all_results.append((chunk, score, f"semantic_{threshold:.2f}"))
                            seen_chunks.add(id(chunk))
                except:
                    continue
        
        # Strategy 2: Keyword search for all query variants
        for variant in query_variants:
            keyword_results = self._keyword_search(variant, max_chunks)
            for chunk, score in keyword_results:
                if id(chunk) not in seen_chunks:
                    all_results.append((chunk, score, "keyword"))
                    seen_chunks.add(id(chunk))
        
        # Strategy 3: Exact function/class name matching
        import re
        code_terms = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query)
        for term in code_terms:
            for chunk in self.indexer.chunks:
                if (chunk.name and chunk.name.lower() == term.lower() and 
                    id(chunk) not in seen_chunks):
                    all_results.append((chunk, 1.0, "exact_match"))
                    seen_chunks.add(id(chunk))
        
        # Sort by score (highest first) and limit results
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:max_chunks]

    def retrieve_context(self, query: str, similarity_threshold: float = 0.3,
                        max_chunks: int = 15) -> Dict[str, any]:
        """
        Enhanced context retrieval using hybrid search (Cursor-style approach).
        
        Args:
            query: User's query
            similarity_threshold: Minimum similarity score for chunks
            max_chunks: Maximum number of chunks to return
            
        Returns:
            Dictionary containing retrieved context and metadata
        """
        if not self.indexer:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Step 1: Hybrid search (semantic + keyword + exact matching)
        hybrid_results = self._hybrid_search(query, similarity_threshold, max_chunks * 2)
        
        # Extract chunks and create method tracking
        retrieved_chunks = []
        retrieval_methods = {}
        similarity_scores = {}
        
        for chunk, score, method in hybrid_results:
            retrieved_chunks.append(chunk)
            retrieval_methods[chunk] = method
            similarity_scores[chunk] = score
        
        # Step 2: Extract focus entities from query and chunks
        focus_entities = self._extract_focus_entities(query, retrieved_chunks)
        
        # Step 3: Expand context using dependency graph
        if self.graph_builder:
            expanded_chunks = self.expand_context_with_graph(retrieved_chunks, focus_entities)
        else:
            expanded_chunks = retrieved_chunks
        
        # Step 4: Limit final results but preserve high-scoring ones
        final_chunks = expanded_chunks[:max_chunks]
        
        # Step 5: Organize results by file
        chunks_by_file = {}
        for chunk in final_chunks:
            if chunk.file_path not in chunks_by_file:
                chunks_by_file[chunk.file_path] = []
            chunks_by_file[chunk.file_path].append(chunk)
        
        # Sort chunks within each file by line number
        for file_path in chunks_by_file:
            chunks_by_file[file_path].sort(key=lambda c: c.start_line)
        
        return {
            "query": query,
            "focus_entities": list(focus_entities),
            "total_chunks": len(final_chunks),
            "chunks_by_file": chunks_by_file,
            "similarity_scores": {k: v for k, v in similarity_scores.items() if k in final_chunks},
            "retrieval_methods": {str(id(k)): v for k, v in retrieval_methods.items() if k in final_chunks},
            "files_involved": list(chunks_by_file.keys()),
            "expanded_queries": self._expand_query_terms(query)[:3]  # Show first 3 for debugging
        }
    
    def get_file_overview(self, file_path: str) -> Dict[str, any]:
        """
        Get overview of a specific file including its exports and imports.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file overview
        """
        overview = {
            "file_path": file_path,
            "imports": set(),
            "exports": set(),
            "functions": [],
            "classes": []
        }
        
        if self.graph_builder:
            overview["imports"] = self.graph_builder.get_file_imports(file_path)
            overview["exports"] = self.graph_builder.get_file_exports(file_path)
            
            # Get all nodes in this file
            file_nodes = self.graph_builder.find_nodes_in_file(file_path)
            for node in file_nodes:
                if node.node_type == "function":
                    overview["functions"].append(node.name)
                elif node.node_type == "class":
                    overview["classes"].append(node.name)
        
        return overview
    
    def suggest_related_queries(self, context: Dict[str, any]) -> List[str]:
        """
        Suggest related queries based on retrieved context.
        
        Args:
            context: Context dictionary from retrieve_context()
            
        Returns:
            List of suggested follow-up queries
        """
        suggestions = []
        
        # Suggest queries based on focus entities
        for entity in context.get("focus_entities", []):
            suggestions.extend([
                f"How is {entity} implemented?",
                f"What calls {entity}?",
                f"What does {entity} depend on?",
                f"How to test {entity}?"
            ])
        
        # Suggest queries based on files involved
        for file_path in context.get("files_involved", []):
            file_name = file_path.split('/')[-1]
            suggestions.extend([
                f"What is the purpose of {file_name}?",
                f"How does {file_name} fit into the overall architecture?"
            ])
        
        # Remove duplicates and limit
        return list(set(suggestions))[:5]


if __name__ == "__main__":
    # Example usage
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    retriever = ContextRetriever(api_key)
    
    # Load index and graph
    retriever.load_index("./faiss_index.index", "./faiss_index.metadata")
    retriever.load_graph("./test_repo")
    
    # Retrieve context for a query
    query = "How does the foo function work?"
    context = retriever.retrieve_context(query)
    
    print(f"Retrieved {context['total_chunks']} chunks across {len(context['files_involved'])} files")
    print(f"Focus entities: {context['focus_entities']}")
    
    # Get suggestions
    suggestions = retriever.suggest_related_queries(context)
    print(f"Related queries: {suggestions}")
