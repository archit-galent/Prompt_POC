"""
Prompt builder that constructs structured prompts for LLMs using retrieved context.
Creates well-formatted prompts with system instructions, code context, and user queries.
"""

from typing import Dict, List, Optional
from datetime import datetime
from indexer import CodeChunk


class PromptBuilder:
    """Builds structured prompts for code analysis tasks."""
    
    def __init__(self):
        self.default_system_prompt = """You are an expert code analyst and software engineer. Your job is to analyze code and provide clear, accurate explanations based on the provided context.

Key guidelines:
- Focus on the specific code provided in the context
- Explain functionality, dependencies, and relationships
- Identify potential issues or areas of concern
- Provide practical insights and recommendations
- Be concise but thorough in your explanations
- Reference specific functions, classes, and files when relevant"""

    def _format_code_chunk(self, chunk: CodeChunk, include_metadata: bool = True) -> str:
        """
        Format a code chunk for inclusion in the prompt.
        
        Args:
            chunk: Code chunk to format
            include_metadata: Whether to include file path and line numbers
            
        Returns:
            Formatted code string
        """
        formatted = ""
        
        if include_metadata:
            formatted += f"\n=== {chunk.file_path} ===\n"
            if chunk.name:
                formatted += f"# {chunk.chunk_type.title()}: {chunk.name}\n"
            formatted += f"# Lines: {chunk.start_line}-{chunk.end_line}\n\n"
        
        formatted += chunk.content
        formatted += "\n"
        
        return formatted
    
    def _create_system_section(self, custom_instructions: Optional[str] = None) -> str:
        """Create the system instruction section."""
        if custom_instructions:
            return f"## SYSTEM INSTRUCTIONS\n\n{custom_instructions}\n"
        else:
            return f"## SYSTEM INSTRUCTIONS\n\n{self.default_system_prompt}\n"
    
    def _create_context_section(self, context: Dict[str, any]) -> str:
        """
        Create the repository context section.
        
        Args:
            context: Context dictionary from retriever
            
        Returns:
            Formatted context section
        """
        section = "## REPOSITORY CONTEXT\n\n"
        
        # Overview
        section += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        section += f"**Total Files Analyzed:** {len(context.get('files_involved', []))}\n"
        section += f"**Total Code Chunks:** {context.get('total_chunks', 0)}\n"
        
        if context.get('focus_entities'):
            section += f"**Key Entities:** {', '.join(context['focus_entities'])}\n"
        
        section += "\n"
        
        # File overview
        if context.get('files_involved'):
            section += "**Files Involved:**\n"
            for file_path in context['files_involved']:
                section += f"- {file_path}\n"
            section += "\n"
        
        return section
    
    def _create_code_section(self, context: Dict[str, any], 
                           max_lines_per_file: int = 200) -> str:
        """
        Create the code snippets section.
        
        Args:
            context: Context dictionary from retriever
            max_lines_per_file: Maximum lines to include per file
            
        Returns:
            Formatted code section
        """
        section = "## CODE SNIPPETS\n\n"
        
        chunks_by_file = context.get('chunks_by_file', {})
        
        for file_path, chunks in chunks_by_file.items():
            section += f"### {file_path}\n\n"
            
            current_lines = 0
            for chunk in chunks:
                # Estimate lines in chunk
                chunk_lines = len(chunk.content.split('\n'))
                
                if current_lines + chunk_lines > max_lines_per_file:
                    section += f"... (truncated - remaining chunks omitted to stay within {max_lines_per_file} line limit)\n\n"
                    break
                
                section += self._format_code_chunk(chunk, include_metadata=True)
                current_lines += chunk_lines
            
            section += "\n"
        
        return section
    
    def _create_dependencies_section(self, context: Dict[str, any]) -> str:
        """
        Create the dependencies and relationships section.
        
        Args:
            context: Context dictionary from retriever
            
        Returns:
            Formatted dependencies section
        """
        section = "## DEPENDENCIES & RELATIONSHIPS\n\n"
        
        focus_entities = context.get('focus_entities', [])
        if focus_entities:
            section += "**Key Components Analyzed:**\n"
            for entity in focus_entities:
                section += f"- `{entity}`\n"
            section += "\n"
        
        # Show query expansion (Cursor-style debugging)
        expanded_queries = context.get('expanded_queries', [])
        if expanded_queries and len(expanded_queries) > 1:
            section += "**Query Expansion:**\n"
            section += f"Original: `{expanded_queries[0]}`\n"
            for i, expanded in enumerate(expanded_queries[1:], 1):
                section += f"Variant {i}: `{expanded}`\n"
            section += "\n"
        
        # Show retrieval methods used
        retrieval_methods = context.get('retrieval_methods', {})
        if retrieval_methods:
            method_counts = {}
            for method in retrieval_methods.values():
                method_counts[method] = method_counts.get(method, 0) + 1
            
            section += "**Search Methods Used:**\n"
            for method, count in method_counts.items():
                method_display = {
                    'exact_match': 'Exact Function/Class Match',
                    'keyword': 'Keyword Search',
                    'semantic_0.30': 'Semantic Search (High)',
                    'semantic_0.21': 'Semantic Search (Medium)', 
                    'semantic_0.15': 'Semantic Search (Low)'
                }.get(method, method.replace('_', ' ').title())
                
                section += f"- {method_display}: {count} chunks\n"
            section += "\n"
        
        # Add similarity scores if available
        similarity_scores = context.get('similarity_scores', {})
        if similarity_scores:
            section += "**Relevance Scores:**\n"
            section += "Components ranked by relevance to your query:\n"
            
            # Sort by score (descending)
            sorted_scores = sorted(
                [(chunk.name or f"block_{chunk.start_line}", score) 
                 for chunk, score in similarity_scores.items()],
                key=lambda x: x[1], reverse=True
            )
            
            for name, score in sorted_scores[:5]:  # Top 5
                section += f"- `{name}`: {score:.3f}\n"
            section += "\n"
        
        return section
    
    def _create_query_section(self, query: str) -> str:
        """Create the user query section."""
        return f"## USER QUERY\n\n{query}\n\n"
    
    def _create_task_section(self, task_type: str = "analysis") -> str:
        """
        Create the task instructions section.
        
        Args:
            task_type: Type of task (analysis, debugging, refactoring, etc.)
            
        Returns:
            Formatted task section
        """
        tasks = {
            "analysis": """## TASK INSTRUCTIONS

Please analyze the provided code and respond with:

1. **Summary**: Brief overview of what the code does
2. **Key Components**: Main functions, classes, and their purposes  
3. **Data Flow**: How data moves through the system
4. **Dependencies**: What the code relies on and what relies on it
5. **Potential Issues**: Any concerns, bugs, or areas for improvement
6. **Recommendations**: Suggestions for optimization or best practices

Be specific and reference the actual code provided above.""",

            "debugging": """## TASK INSTRUCTIONS

Please help debug the provided code by:

1. **Issue Identification**: Identify potential bugs or problems
2. **Root Cause Analysis**: Explain why issues occur
3. **Impact Assessment**: What could break or malfunction
4. **Fix Recommendations**: Specific solutions with code examples
5. **Prevention**: How to avoid similar issues in the future

Focus on the specific query and reference the actual code provided.""",

            "refactoring": """## TASK INSTRUCTIONS

Please suggest refactoring improvements for the provided code:

1. **Code Quality**: Identify areas for improvement
2. **Design Patterns**: Suggest better architectural approaches
3. **Performance**: Highlight optimization opportunities  
4. **Maintainability**: Ways to make code more readable and maintainable
5. **Testing**: How to make the code more testable
6. **Specific Changes**: Concrete refactoring steps with examples

Reference the actual code and maintain functionality while improving quality.""",

            "explanation": """## TASK INSTRUCTIONS

Please provide a clear explanation of the provided code:

1. **Purpose**: What this code is designed to do
2. **How It Works**: Step-by-step explanation of the logic
3. **Key Concepts**: Important algorithms, patterns, or techniques used
4. **Input/Output**: What goes in and what comes out
5. **Context**: How this fits into the larger system
6. **Examples**: If helpful, provide usage examples

Make the explanation accessible and reference specific parts of the code."""
        }
        
        return tasks.get(task_type, tasks["analysis"])
    
    def build_prompt(self, context: Dict[str, any], query: str,
                    task_type: str = "analysis",
                    custom_system_prompt: Optional[str] = None,
                    include_dependencies: bool = True,
                    max_lines_per_file: int = 200) -> str:
        """
        Build a complete structured prompt.
        
        Args:
            context: Context dictionary from retriever
            query: User's query
            task_type: Type of analysis task
            custom_system_prompt: Optional custom system instructions
            include_dependencies: Whether to include dependency information
            max_lines_per_file: Maximum lines to include per file
            
        Returns:
            Complete formatted prompt
        """
        prompt_parts = []
        
        # 1. System Instructions
        prompt_parts.append(self._create_system_section(custom_system_prompt))
        
        # 2. Repository Context
        prompt_parts.append(self._create_context_section(context))
        
        # 3. Dependencies & Relationships (optional)
        if include_dependencies:
            prompt_parts.append(self._create_dependencies_section(context))
        
        # 4. Code Snippets
        prompt_parts.append(self._create_code_section(context, max_lines_per_file))
        
        # 5. User Query
        prompt_parts.append(self._create_query_section(query))
        
        # 6. Task Instructions
        prompt_parts.append(self._create_task_section(task_type))
        
        # Combine all parts
        full_prompt = "\n".join(prompt_parts)
        
        return full_prompt
    
    def build_focused_prompt(self, chunks: List[CodeChunk], query: str,
                           focus_entity: str = None) -> str:
        """
        Build a focused prompt for a specific function or class.
        
        Args:
            chunks: Relevant code chunks
            query: User's query
            focus_entity: Main entity being analyzed
            
        Returns:
            Focused prompt string
        """
        prompt = f"## FOCUSED CODE ANALYSIS\n\n"
        
        if focus_entity:
            prompt += f"**Primary Focus:** `{focus_entity}`\n\n"
        
        prompt += f"**Query:** {query}\n\n"
        
        prompt += "## RELEVANT CODE\n\n"
        
        for chunk in chunks:
            prompt += self._format_code_chunk(chunk, include_metadata=True)
        
        prompt += """## INSTRUCTIONS

Analyze the provided code and answer the specific query. Focus on:
- The primary entity mentioned in the query
- How it works and what it does
- Its relationships with other components
- Any relevant implementation details
- Potential issues or improvements

Be specific and reference the actual code provided."""
        
        return prompt
    
    def estimate_token_count(self, prompt: str) -> int:
        """
        Rough estimation of token count for the prompt.
        
        Args:
            prompt: The prompt string
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token for code
        return len(prompt) // 4
    
    def truncate_prompt(self, prompt: str, max_tokens: int = 16000) -> str:
        """
        Truncate prompt if it exceeds token limit.
        
        Args:
            prompt: Original prompt
            max_tokens: Maximum allowed tokens
            
        Returns:
            Truncated prompt if necessary
        """
        estimated_tokens = self.estimate_token_count(prompt)
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # Calculate how much to truncate
        reduction_ratio = max_tokens / estimated_tokens
        target_length = int(len(prompt) * reduction_ratio * 0.9)  # 10% buffer
        
        # Try to truncate at a section boundary
        sections = prompt.split('\n## ')
        truncated_sections = []
        current_length = 0
        
        for section in sections:
            if current_length + len(section) < target_length:
                truncated_sections.append(section)
                current_length += len(section)
            else:
                # Add truncation notice
                truncated_sections.append("... (content truncated to fit token limit)")
                break
        
        return '\n## '.join(truncated_sections)


if __name__ == "__main__":
    # Example usage
    from retriever import ContextRetriever
    
    # Mock context for demonstration
    mock_context = {
        "query": "How does the foo function work?",
        "focus_entities": ["foo", "bar"],
        "total_chunks": 3,
        "files_involved": ["utils.py", "main.py"],
        "chunks_by_file": {},
        "similarity_scores": {}
    }
    
    builder = PromptBuilder()
    
    # Build different types of prompts
    analysis_prompt = builder.build_prompt(mock_context, "How does foo() work?", "analysis")
    print("=== ANALYSIS PROMPT ===")
    print(f"Length: {len(analysis_prompt)} characters")
    print(f"Estimated tokens: {builder.estimate_token_count(analysis_prompt)}")
    
    debugging_prompt = builder.build_prompt(mock_context, "Why is foo() failing?", "debugging")
    print("\n=== DEBUGGING PROMPT ===")
    print(f"Length: {len(debugging_prompt)} characters")
