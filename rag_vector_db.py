# ================================================
# RAG — Auto Multi-File Vector DB Pipeline
# ================================================
# HOW TO USE:
# 1. Add any .txt file in this folder
# 2. Run: python rag_vector_db.py
# 3. New files will be added to DB automatically!
# ================================================
# Install dependencies (first time only):
# pip install langchain langchain-community langchain-text-splitters
#             chromadb sentence-transformers langchain-huggingface
#             langchain-chroma
# ================================================

import os
import hashlib
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#Importing Chroma DB vector store from langchain_chroma (local file-based DB)
from langchain_chroma import Chroma

#Importing PGVector vector store from langchain_community (PostgreSQL-based DB)
# from langchain_community.vectorstores import PGVector

# ------------------------------------------------
# CONFIG — Control everything from here
# ------------------------------------------------
FOLDER_PATH   = "."            # load all .txt files from current folder
CHROMA_DB     = "./chroma_db"  # local path to save vector DB
CHUNK_SIZE    = 200            # max characters per chunk
CHUNK_OVERLAP = 30             # overlap between chunks

# ------------------------------------------------
# STEP 1 — Load All .txt Files from Folder
# ------------------------------------------------
print("📂 Step 1: Scanning folder for .txt files...")

loader = DirectoryLoader(
    FOLDER_PATH,
    glob="docs/*.txt",                        # only .txt files
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True
)

documents = loader.load()

if not documents:
    print("❌ No .txt files found! Please add files to the folder.")
    exit()

# Print which files were loaded
loaded_files = set(doc.metadata['source'] for doc in documents)
print(f"\n✅ {len(loaded_files)} file(s) found:")
for f in loaded_files:
    print(f"   📄 {os.path.basename(f)}")

# ------------------------------------------------
# STEP 2 — Split Documents into Chunks
# ------------------------------------------------
print("\n✂️  Step 2: Splitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(documents)
print(f"✅ {len(chunks)} chunks created (across all files)")

# ------------------------------------------------
# STEP 3 — Generate Unique IDs (Duplicate Fix)
# ------------------------------------------------
# Each chunk gets a unique ID based on its content
# This ensures no duplicates on repeated runs
ids = []
for chunk in chunks:
    unique_str = chunk.metadata['source'] + chunk.page_content
    chunk_id   = hashlib.md5(unique_str.encode()).hexdigest()
    ids.append(chunk_id)

# ------------------------------------------------
# STEP 4 — Embed + Store in ChromaDB
# ------------------------------------------------
print("\n🔢 Step 3: Embedding chunks and storing in ChromaDB...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Using — ChromaDB
vectorstore = Chroma(
    persist_directory=CHROMA_DB,
    embedding_function=embeddings
)

# Using — PGVector
# CONNECTION_STRING = "postgresql+psycopg2://username:password@localhost:5432/dbname"

# vectorstore = PGVector(
#     connection_string=CONNECTION_STRING,
#     embedding_function=embeddings,
#     collection_name="my_docs"
# )

# Check which chunks are already in DB — skip existing, add only new
existing     = vectorstore.get()
existing_ids = set(existing['ids'])
new_chunks   = []
new_ids      = []

for chunk, cid in zip(chunks, ids):
    if cid not in existing_ids:
        new_chunks.append(chunk)
        new_ids.append(cid)

if new_chunks:
    vectorstore.add_documents(new_chunks, ids=new_ids)
    print(f"✅ {len(new_chunks)} new chunks added to DB!")
else:
    print("ℹ️  No new data found — all chunks already exist in DB!")

total = vectorstore.get()
print(f"📊 Total chunks in DB: {len(total['ids'])}")

# ------------------------------------------------
# STEP 5 — Query the Vector DB
# ------------------------------------------------
print("\n🔍 Step 4: Running test queries...\n")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}  # return top 2 most relevant chunks
)

queries = [
    "Where does Abhishek work?",
    "What is IFM?",
    "What is RAG?"
]

for query in queries:
    print(f"❓ Query: {query}")
    results = retriever.invoke(query)
    print(f"📌 Result: {results[0].page_content}")
    print("-" * 50)

# ------------------------------------------------
# DONE
# ------------------------------------------------
print("\n🎉 Done! Vector DB is ready and up to date.")
print("📁 Add a new .txt file and re-run — it will be added automatically!")
print("\nNext Step: Connect this retriever to an LLM → RetrievalQA Chain!")