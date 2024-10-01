import os
import streamlit as st
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Directory to store PDFs and embeddings
PDF_DIRECTORY = './pdfs/'
EMBEDDINGS_DIRECTORY = './embeddings/'

# Create directories if they don't exist
os.makedirs(PDF_DIRECTORY, exist_ok=True)
os.makedirs(EMBEDDINGS_DIRECTORY, exist_ok=True)

# Streamlit interface
st.title("PDF Q&A RAG Machine")

# Function to generate embeddings for a PDF
def process_pdf(file_path, file_name):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    final_docs = text_splitter.split_documents(docs)

    # Create embeddings
    embedding = OllamaEmbeddings(model="llama3.2")
    db = FAISS.from_documents(final_docs, embedding)

    # Remove the .pdf extension for the embedding filename
    embedding_filename = os.path.splitext(file_name)[0]  # Get the filename without extension

    # Save embeddings
    db.save_local(os.path.join(EMBEDDINGS_DIRECTORY, embedding_filename))

# Function to delete a PDF and its embeddings
def delete_pdf(file_name):
    pdf_path = os.path.join(PDF_DIRECTORY, file_name)
    embedding_path = os.path.join(EMBEDDINGS_DIRECTORY, os.path.splitext(file_name)[0])  # Remove extension for embeddings

    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if os.path.exists(embedding_path):
        shutil.rmtree(embedding_path)  # Remove the entire embeddings folder

# Upload PDF section
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf is not None:
    # Save uploaded PDF to directory
    pdf_path = os.path.join(PDF_DIRECTORY, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    # Generate embeddings
    process_pdf(pdf_path, uploaded_pdf.name)
    st.success(f"Embeddings created and stored for {uploaded_pdf.name}")

# Display available PDFs
st.subheader("Available PDFs")
pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]

if pdf_files:
    selected_pdf = st.selectbox("Select a PDF to use for Q&A", pdf_files)

    # Allow deletion of selected PDF
    if st.button(f"Delete {selected_pdf}"):
        delete_pdf(selected_pdf)
        st.success(f"{selected_pdf} has been deleted.")

    # Option to ask questions based on selected PDF
    if selected_pdf:
        query = st.text_input("Ask a question based on the selected PDF")

        # Generate button
        if st.button("Generate"):
            if query:
                # Load embeddings and search for relevant information
                embedding = OllamaEmbeddings(model="llama3.2")
                db = FAISS.load_local(os.path.join(EMBEDDINGS_DIRECTORY, os.path.splitext(selected_pdf)[0]), embedding, allow_dangerous_deserialization=True)

                retriever = db.as_retriever(search_kwargs={"k": 10})
                llama = Ollama(model="llama3.2")

                # Create a prompt for Q&A
                prompt = ChatPromptTemplate.from_template("""
                    Based on the following context, explain the concept of '{query}' if it is mentioned in the context.
                    <context>
                    {context}
                    </context>
                    If the context includes a section heading related to '{query}', focus on that part.
                """)

                document_chain = create_stuff_documents_chain(llm=llama, prompt=prompt)

                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llama, 
                    chain_type="stuff",
                    retriever=retriever,
                )

                # Get the response from the QA chain
                response = qa_chain.invoke(query)
                st.write("Answer:", response['result'])
            else:
                st.warning("Please enter a query to generate an answer.")
else:
    st.write("No PDFs available. Please upload one to start.")
