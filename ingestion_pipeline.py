import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_files(docs_path="docs"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist.")
    
    loader = DirectoryLoader(
        docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    return loader.load()

def split_documents(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=0
    )
    return text_splitter.split_documents(documents)

def embed_and_store(chunks, persist_directory="./db/chromadb"):
    # Using a free, local Hugging Face embedding model instead of OpenAI
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Local Vector store has been successfully created and saved.")
    return vector_store

def main():
    print("Starting ingestion pipeline...")
    documents = load_files("docs")
    chunks = split_documents(documents)
    embed_and_store(chunks)

if __name__ == "__main__":
    main()