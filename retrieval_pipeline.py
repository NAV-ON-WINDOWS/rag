from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

persist_directory = "./db/chromadb"

# Re-initialize the exact same local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Recreate the vector store connection
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Fetch top 5 chunks based on cosine similarity
retriever = db.as_retriever(search_kwargs={"k": 5})

user_query = "In what year did Tesla begin production of the Roadster?"

print(f"User Query: {user_query}\n")

# This is where it stops! It grabs the context but doesn't pass it to an LLM.
retrieved_docs = retriever.invoke(user_query)

for i, doc in enumerate(retrieved_docs, 1):
    print(f"--- Document {i} ---")
    print(doc.page_content)
    print("\n")