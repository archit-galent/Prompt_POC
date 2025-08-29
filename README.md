# Prompt Engineering Agent POC

A CLI-based tool that analyzes local codebases and constructs contextual prompts for LLM queries. This POC implements the full prompt engineering pipeline: codebase indexing, dependency graph mapping, context retrieval, and structured prompt construction.

## Features

- **Semantic Code Chunking**: Breaks code into function/class-level chunks using AST analysis
- **Vector Embeddings**: Uses OpenAI embeddings API with FAISS for similarity search
- **Dependency Graph**: Builds call graphs to understand function dependencies and relationships
- **Context Expansion**: Combines similarity search with dependency graph traversal
- **Structured Prompts**: Constructs well-formatted prompts with system instructions, code context, and task guidance

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see `requirements.txt`)

## Installation

1. Clone this repository:
```bash
git clone <repo-url>
cd prompt-engineering-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Basic Usage

```bash
python agent.py --repo ./test_repo --query "Explain how foo() works"
```

### Advanced Options

```bash
# Debugging task type
python agent.py --repo ./myproject --query "Why is the authentication failing?" --task debugging

# Force rebuild index
python agent.py --repo ./utils --query "How does caching work?" --rebuild

# Custom similarity threshold and chunk limits
python agent.py --repo . --query "What would break if CONFIG_X is missing?" --similarity-threshold 0.4 --max-chunks 20

# Show repository statistics
python agent.py --repo ./test_repo --stats
```

### Command Line Options

- `--repo`: Path to the repository to analyze (required)
- `--query`: Your question about the code (required)
- `--task`: Analysis type (`analysis`, `debugging`, `refactoring`, `explanation`) - default: `analysis`
- `--rebuild`: Force rebuild the index even if it exists
- `--similarity-threshold`: Minimum similarity for chunk retrieval (default: 0.3)
- `--max-chunks`: Maximum number of code chunks to include (default: 15)
- `--max-tokens`: Maximum tokens in generated prompt (default: 16000)
- `--stats`: Show repository statistics instead of processing query
- `--model`: OpenAI embedding model (default: `text-embedding-3-small`)

## Architecture

### Core Components

1. **`indexer.py`**: Code chunking and FAISS embedding storage
   - Parses code files using AST
   - Creates function/class-level chunks
   - Generates embeddings via OpenAI API
   - Stores in FAISS index for fast similarity search

2. **`graph.py`**: AST-based dependency graph building
   - Analyzes imports and function calls
   - Builds call graph relationships
   - Enables context expansion through dependencies

3. **`retriever.py`**: Context retrieval and expansion
   - Performs similarity search on user queries
   - Expands context using dependency relationships
   - Ranks and filters relevant code chunks

4. **`prompt_builder.py`**: Structured prompt construction
   - Creates multi-section prompts
   - Includes system instructions, code context, and task guidance
   - Handles token limits and truncation

5. **`agent.py`**: CLI entry point that orchestrates the pipeline

### Workflow

1. **Index Repository** (first run or with `--rebuild`):
   - Walk through code files
   - Extract functions/classes using AST
   - Generate embeddings and build FAISS index
   - Build dependency graph

2. **Process Query**:
   - Generate embedding for user query
   - Search FAISS index for similar code chunks
   - Expand context using dependency graph
   - Construct structured prompt

3. **Output**: Returns the complete prompt ready for LLM consumption

## Example: Testing with the Toy Repository

The project includes a `test_repo/` with sample code to demonstrate the agent:

```bash
# Basic analysis
python agent.py --repo ./test_repo --query "Explain how foo() works"

# Critical dependency analysis
python agent.py --repo ./test_repo --query "What would break if CONFIG_X is missing?"

# Function relationship analysis  
python agent.py --repo ./test_repo --query "How does DataProcessor use the foo function?"

# Debugging scenario
python agent.py --repo ./test_repo --query "Why might the batch processing fail?" --task debugging
```

### Sample Query Output

When you run:
```bash
python agent.py --repo ./test_repo --query "Explain how foo() works, and what would break if CONFIG_X is missing."
```

The agent will:
1. Find the `foo()` function in `utils/helpers.py`
2. Discover its usage in `utils/data_processor.py`
3. Identify the CONFIG_X dependency
4. Construct a comprehensive prompt with:
   - System instructions for code analysis
   - Repository context and file overview
   - Relevant code snippets with metadata
   - Dependency information
   - Specific task instructions

## Test Repository Structure

```
test_repo/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration with CONFIG_X dependency
├── utils/
│   ├── __init__.py
│   ├── helpers.py           # Core foo() function and utilities
│   └── data_processor.py    # DataProcessor class that uses foo()
└── main.py                  # Main application entry point
```

### Key Features Demonstrated

- **`foo()` function**: Core data processing function with grouping logic
- **CONFIG_X dependency**: Critical configuration that breaks the system if missing
- **Function relationships**: DataProcessor.process_batch() depends on foo()
- **Error handling**: Various failure modes and validation logic
- **Import dependencies**: Clear module relationships for graph analysis

## Output Format

The generated prompt includes:

1. **System Instructions**: Guidelines for the LLM
2. **Repository Context**: Overview of files and entities
3. **Dependencies & Relationships**: Relevance scores and key components
4. **Code Snippets**: Actual code with file paths and line numbers
5. **User Query**: The original question
6. **Task Instructions**: Specific guidance based on task type

## Token Management

- Estimates token count for prompts
- Automatically truncates if exceeding limits
- Prioritizes most relevant code chunks
- Maintains prompt structure while reducing size

## Indexing Performance

- First run: Indexes repository and saves to `.prompt_agent_index.*` files
- Subsequent runs: Loads existing index (much faster)
- Use `--rebuild` to force re-indexing after code changes

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**: Set your OpenAI API key environment variable
2. **"Repository not indexed"**: Index gets corrupted - use `--rebuild`
3. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
4. **Empty results**: Try lowering `--similarity-threshold` (e.g., 0.2)

### Debug Tips

- Use `--stats` to see repository indexing statistics
- Check FAISS index files in repo: `.prompt_agent_index.index` and `.prompt_agent_index.metadata`
- Test with the provided `test_repo` first to verify setup

## Limitations

- Currently optimized for Python code (AST parsing)
- Other languages use simpler line-based chunking
- Embedding costs scale with repository size
- Dependency graph is based on static analysis (no runtime information)

## Future Enhancements

- Support for more programming languages
- Local embedding models to reduce API costs
- More sophisticated dependency analysis
- Integration with IDE extensions
- Caching layer for repeated queries

## License

MIT License - see LICENSE file for details.
