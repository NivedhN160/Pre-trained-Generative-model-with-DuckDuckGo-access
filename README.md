# Neural GPT Retrieval-QA Chatbot

This project is a local Retrieval-Augmented Generation (RAG) chatbot powered by a fine-tuned GPT-Neo 125M model combined with vector search over custom documents and live internet search using DuckDuckGo. It provides informative and context-aware answers interactively.

## Features

- Fine-tuned GPT-Neo 125M for domain-specific knowledge
- Local document retrieval using FAISS vector store with Sentence Transformers embeddings
- Dynamic web search integration via DuckDuckGo (`ddgs` package) for up-to-date information
- Simple command-line chat interface for easy interaction

## Setup

### Requirements

- Python 3.8+
- PyTorch
- [Transformers](https://huggingface.co/docs/transformers)
- LangChain & LangChain-Community
- Sentence Transformers
- FAISS
- DDGS (DuckDuckGo Search API wrapper)

### Installation

Install dependencies via pip:

pip install -r requirements.txt

Or individually:

pip install torch transformers langchain langchain-community sentence-transformers faiss-cpu ddgs

### Files

- `fine_tune_neo.py`: Script to fine-tune GPT-Neo 125M on custom data.
- `retrieval_qa_with_websearch.py`: Retrieval QA pipeline combining local retrieval and DuckDuckGo web search.
- `main.py`: Unified chat interface script to run the chatbot interactively.
- `book.txt`: Sample document for local knowledge retrieval.
- `README.md`: This file.

## Usage

Run the chatbot interface:

python main.py

Type your questions and get answers powered by local and web retrieval!

Type `exit` to quit.

## Notes

- Keep your fine-tuned model folder `fine_tuned_neo` and `book.txt` in the project directory.
- Ensure you have internet access for live web search.
- The model generates responses on CPU by default; GPU support depends on your PyTorch installation.

## License

This project is for personal and research use.
