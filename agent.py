"""
CLI agent that ties together all components to create contextual prompts for code queries.
Main entry point for the prompt engineering agent POC.
"""

import os
import argparse
import sys
from pathlib import Path
from typing import Optional

from indexer import CodeIndexer
from graph import DependencyGraphBuilder
from retriever import ContextRetriever
from prompt_builder import PromptBuilder


class PromptAgent:
    """Main agent that orchestrates the prompt creation pipeline."""
    
    def __init__(self, api_key: str, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the prompt agent.
        
        Args:
            api_key: OpenAI API key
            embedding_model: OpenAI embedding model to use
        """
        self.api_key = api_key
        self.embedding_model = embedding_model
        
        # Initialize components
        self.indexer = CodeIndexer(api_key, embedding_model)
        self.graph_builder = DependencyGraphBuilder()
        self.retriever = ContextRetriever(api_key, embedding_model)
        self.prompt_builder = PromptBuilder()
        
        self.repo_path: Optional[str] = None
        self.index_built = False
        
    def index_repository(self, repo_path: str, force_rebuild: bool = False):
        """
        Index a repository for the first time or rebuild if needed.
        
        Args:
            repo_path: Path to the repository to index
            force_rebuild: Whether to force rebuild even if index exists
        """
        self.repo_path = repo_path
        
        # Check if index already exists
        index_path = f"{repo_path}/.prompt_agent_index.index"
        metadata_path = f"{repo_path}/.prompt_agent_index.metadata"
        
        if not force_rebuild and Path(index_path).exists() and Path(metadata_path).exists():
            print(f"Loading existing index for {repo_path}...")
            self.retriever.load_index(index_path, metadata_path)
            self.retriever.load_graph(repo_path)
            self.index_built = True
            print("Index loaded successfully!")
            return
        
        print(f"Indexing repository: {repo_path}")
        print("This may take a moment depending on repository size...")
        
        # Step 1: Build dependency graph
        print("1. Building dependency graph...")
        self.graph_builder.build_graph(repo_path)
        
        # Step 2: Create embeddings index
        print("2. Creating embeddings index...")
        self.indexer.index_repository(repo_path, f"{repo_path}/.prompt_agent_index")
        
        # Step 3: Setup retriever
        print("3. Setting up retriever...")
        self.retriever.load_index(index_path, metadata_path)
        self.retriever.load_graph(repo_path)
        
        self.index_built = True
        print("Repository indexed successfully!")
    
    def query(self, query: str, task_type: str = "analysis", 
              similarity_threshold: float = 0.3, max_chunks: int = 15,
              max_tokens: int = 16000) -> str:
        """
        Process a query and return the constructed prompt.
        
        Args:
            query: User's query about the code
            task_type: Type of analysis (analysis, debugging, refactoring, explanation)
            similarity_threshold: Minimum similarity for chunk retrieval
            max_chunks: Maximum number of chunks to include
            max_tokens: Maximum tokens in the prompt
            
        Returns:
            Constructed prompt string
        """
        if not self.index_built:
            raise ValueError("Repository not indexed. Call index_repository() first.")
        
        print(f"Processing query: {query}")
        
        # Step 1: Retrieve relevant context
        print("Retrieving relevant code context...")
        context = self.retriever.retrieve_context(
            query, 
            similarity_threshold=similarity_threshold,
            max_chunks=max_chunks
        )
        
        print(f"Found {context['total_chunks']} relevant chunks across {len(context['files_involved'])} files")
        
        if context['focus_entities']:
            print(f"Focus entities: {', '.join(context['focus_entities'])}")
        
        # Step 2: Build structured prompt
        print("Building structured prompt...")
        prompt = self.prompt_builder.build_prompt(context, query, task_type)
        
        # Step 3: Check token count and truncate if needed
        estimated_tokens = self.prompt_builder.estimate_token_count(prompt)
        print(f"Estimated token count: {estimated_tokens}")
        
        if estimated_tokens > max_tokens:
            print(f"Prompt too long ({estimated_tokens} tokens), truncating to {max_tokens} tokens...")
            prompt = self.prompt_builder.truncate_prompt(prompt, max_tokens)
        
        return prompt
    
    def get_repository_stats(self) -> dict:
        """Get statistics about the indexed repository."""
        if not self.index_built:
            return {"error": "Repository not indexed"}
        
        stats = {
            "repository_path": self.repo_path,
            "total_chunks": len(self.retriever.indexer.chunks),
            "total_nodes": len(self.retriever.graph_builder.nodes),
            "files_indexed": len(set(chunk.file_path for chunk in self.retriever.indexer.chunks)),
            "embedding_model": self.embedding_model
        }
        
        # File type breakdown
        file_types = {}
        for chunk in self.retriever.indexer.chunks:
            ext = Path(chunk.file_path).suffix or "no_extension"
            file_types[ext] = file_types.get(ext, 0) + 1
        
        stats["file_types"] = file_types
        
        return stats


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prompt Engineering Agent - Generate contextual prompts for code queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --repo ./myproject --query "How does the authentication work?"
  %(prog)s --repo ./utils --query "Explain the foo() function" --task debugging
  %(prog)s --repo . --query "What would break if CONFIG_X is missing?" --rebuild
        """
    )
    
    parser.add_argument(
        "--repo", 
        required=True,
        help="Path to the repository to analyze"
    )
    
    parser.add_argument(
        "--query",
        required=True,
        help="Your question about the code"
    )
    
    parser.add_argument(
        "--task",
        choices=["analysis", "debugging", "refactoring", "explanation"],
        default="analysis",
        help="Type of analysis to perform (default: analysis)"
    )
    
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the index even if it exists"
    )
    
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.3,
        help="Minimum similarity threshold for chunk retrieval (default: 0.3)"
    )
    
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=15,
        help="Maximum number of code chunks to include (default: 15)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="Maximum tokens in the generated prompt (default: 16000)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show repository statistics instead of processing query"
    )
    
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model to use (default: text-embedding-3-small)"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Validate repository path
    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    if not repo_path.is_dir():
        print(f"Error: Repository path is not a directory: {repo_path}")
        sys.exit(1)
    
    try:
        # Initialize agent
        print("Initializing Prompt Agent...")
        agent = PromptAgent(api_key, args.model)
        
        # Index repository
        agent.index_repository(str(repo_path), force_rebuild=args.rebuild)
        
        # Show stats if requested
        if args.stats:
            stats = agent.get_repository_stats()
            print("\n" + "="*50)
            print("REPOSITORY STATISTICS")
            print("="*50)
            for key, value in stats.items():
                print(f"{key}: {value}")
            return
        
        # Process query and generate prompt
        prompt = agent.query(
            args.query,
            task_type=args.task,
            similarity_threshold=args.similarity_threshold,
            max_chunks=args.max_chunks,
            max_tokens=args.max_tokens
        )
        
        # Output the final prompt
        print("\n" + "="*80)
        print("GENERATED PROMPT")
        print("="*80)
        print(prompt)
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
