# PDF Q&A RAG Machine

A Streamlit application that allows users to upload PDFs, generate embeddings for them, and perform question-and-answer tasks based on the content of the uploaded PDFs using a Retrieval-Augmented Generation (RAG) approach.

## Features

- Upload PDF documents and generate embeddings.
- Select uploaded PDFs for question-answering.
- Delete PDFs and their corresponding embeddings.
- Ask questions based on the selected PDF, retrieving relevant answers.

## Requirements

- Python 3.8 or later
- Streamlit
- LangChain
- Ollama
- Other necessary libraries

## Installation

1. Clone the repository

   ```bash
   git clone https://github.com/KroNicalKODER/PDF-CHAT-RAG
   cd pdf-qna-rag-machine
   ```

2. Install the dependecies
  ```bash
  pip install -r requirements.txt
  ```

3. Run The App

  ```bash
  streamlit run app.py
  ```
