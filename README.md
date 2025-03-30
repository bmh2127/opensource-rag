# Open Source RAG Pipeline

A flexible and interactive RAG (Retrieval Augmented Generation) pipeline that allows you to ask questions about PDF documents using various LLM models.

## Features

- Support for multiple LLM models:
  - GPT-4 (OpenAI)
  - Claude 3 Sonnet (Anthropic)
  - Llama2 (via Ollama)
  - Llama3 (via Ollama)
- Interactive question-answering mode
- Single question mode
- Efficient document processing with vector embeddings
- OpenAI embeddings for GPT and Claude models
- Ollama embeddings for Llama models

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT model and embeddings)
- Anthropic API key (for Claude model)
- Ollama installed locally (for Llama models)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd opensource-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

5. For Llama models, make sure Ollama is installed and running locally with the required models:
```bash
ollama pull llama2
ollama pull llama3
```

## Usage

The script can be used in two modes:

### Interactive Mode

Run the script in interactive mode to ask multiple questions about a PDF:

```bash
python main.py path/to/your.pdf gpt --interactive
```

In interactive mode:
- Type your questions and press Enter
- Type 'quit', 'exit', or 'q' to end the session
- Press Ctrl+C to exit at any time
- Empty questions are ignored

### Single Question Mode

Ask a single question about a PDF:

```bash
python main.py path/to/your.pdf gpt --question "What is this document about?"
```

### Available Models

- `gpt`: GPT-4 (requires OpenAI API key)
- `claude`: Claude 3 Sonnet (requires Anthropic API key)
- `llama2`: Llama2 (requires Ollama)
- `llama3`: Llama3 (requires Ollama)

### Command Line Arguments

- `pdf_path`: Path to the PDF file to analyze (required)
- `model`: Model to use for answering questions (required, choices: gpt, claude, llama2, llama3)
- `--interactive`, `-i`: Run in interactive mode
- `--question`, `-q`: Single question to ask about the PDF (ignored in interactive mode)

## Example

```bash
# Interactive mode with GPT-4
python main.py documents/research.pdf gpt --interactive

# Single question with Claude
python main.py documents/research.pdf claude --question "What are the key findings?"

# Single question with Llama2
python main.py documents/research.pdf llama2 --question "Summarize the main points"
```

## Notes

- The script uses OpenAI embeddings for both GPT and Claude models for optimal performance
- Ollama embeddings are used for Llama models
- The vector store is kept in memory during interactive sessions for faster subsequent questions
- Make sure your PDF files are readable and not corrupted
- Large PDFs may take longer to process initially

## License


## Contributing
