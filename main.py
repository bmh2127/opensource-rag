import os
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Load environment variables
load_dotenv()

# Define available models
MODELS = {
    "gpt": "gpt-4",
    "claude": "claude-3-sonnet-20240229",
    "llama2": "llama2",
    "llama3": "llama3"
}

def get_model(model_name: str):
    """Get the appropriate model based on the model name."""
    if model_name == "gpt":
        return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=MODELS["gpt"])
    elif model_name == "claude":
        return ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model=MODELS["claude"])
    else:
        return OllamaLLM(model=MODELS[model_name])

def get_embeddings(model_name: str):
    """Get the appropriate embeddings model based on the model name."""
    # Use OpenAI embeddings for both GPT and Claude
    if model_name in ["gpt", "claude"]:
        return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        return OllamaEmbeddings(model=MODELS[model_name])

def create_rag_chain(pdf_path: str, model_name: str):
    """Create the RAG chain with the specified model."""
    # Load and split the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    # Get the appropriate model and embeddings
    model = get_model(model_name)
    embeddings = get_embeddings(model_name)
    
    # Create vector store
    vectorstore = DocArrayInMemorySearch.from_documents(
        pages,
        embedding=embeddings
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever()
    
    # Create prompt template
    template = """
    Answer the question based on the context below. If you don't know the answer, just reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    
    # Create parser
    parser = StrOutputParser()
    
    # Create the chain
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question")
        }
        | prompt
        | model
        | parser
    )
    
    return chain

def interactive_session(chain):
    """Run an interactive session for asking questions about the PDF."""
    print("\nInteractive session started. Type 'quit' to exit.")
    print("Ask questions about the PDF content:")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nEnding interactive session...")
                break
                
            if not question:
                continue
                
            # Get the answer
            answer = chain.invoke({"question": question})
            
            # Print the results
            print(f"\nAnswer: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n\nEnding interactive session...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'quit' to exit.")

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline for PDF question answering")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("model", choices=["gpt", "claude", "llama2", "llama3"], help="Model to use for answering questions")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--question", "-q", help="Single question to ask about the PDF (ignored in interactive mode)")
    
    args = parser.parse_args()
    
    # Create the RAG chain
    chain = create_rag_chain(args.pdf_path, args.model)
    
    if args.interactive:
        interactive_session(chain)
    else:
        if not args.question:
            print("Error: Please provide a question using --question or run in interactive mode with --interactive")
            return
            
        # Get the answer
        answer = chain.invoke({"question": args.question})
        
        # Print the results
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
