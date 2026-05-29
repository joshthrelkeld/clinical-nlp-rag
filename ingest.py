import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Load all PDFs from the data/ folder
data_dir = "data"
all_docs = []

for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        filepath = os.path.join(data_dir, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from {filename}")
        all_docs.extend(pages)

print(f"\nTotal pages loaded: {len(all_docs)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_docs)
print(f"Split into {len(chunks)} chunks")

# Embed and store in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("\nVectorstore created and saved to chroma_db/")