# Open Source RAG with LangChain

This project demonstrates a Retrieval Augmented Generation (RAG) system using LangChain to process and query PDF documents. The example uses Richard Feynman's lectures on physics to showcase how to build a question-answering system that can provide accurate responses based on document content.

## Features

- PDF document processing using LangChain's PyPDFLoader
- Vector storage using DocArrayInMemorySearch
- Embeddings generation using Ollama
- Question-answering chain using LangChain's components
- Support for streaming responses
- Batch processing capabilities

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Ollama installed and running locally

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd opensource-rag
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

1. Place your PDF documents in the project directory.

2. Run the Jupyter notebook:
```bash
jupyter notebook notebook.ipynb
```

3. Follow the notebook cells to:
   - Load and process your PDF documents
   - Create embeddings and vector store
   - Set up the question-answering chain
   - Query your documents

## Project Structure

- `notebook.ipynb`: Main Jupyter notebook containing the RAG implementation
- `requirements.txt`: Project dependencies
- `feynman.pdf`: Example PDF document (Feynman's lectures)

## Dependencies

- langchain
- langchain-openai
- langchain-ollama
- langchain-community
- docarray
- pydantic
- python-dotenv
- jupyter

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## License


## Contributing
