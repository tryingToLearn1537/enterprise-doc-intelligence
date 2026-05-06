import chromadb
from chromadb.config import Settings
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_RESULTS
from src.embeddings import get_embedding, get_embeddings_batch


def get_chroma_client():
    """
    Creates and returns a ChromaDB client.
    The client is our connection to the vector database.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH  # Saves the database to disk at this path
    )
    return client


def get_or_create_collection(client):
    """
    Gets an existing collection or creates a new one.
    A collection is like a table in a normal database.
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity for search
    )
    return collection


def store_chunks(chunks: list, doc_name: str):
    """
    Takes a list of text chunks and stores them in ChromaDB.
    Each chunk gets: an ID, its text, and its vector embedding.
    """
    print(f"💾 Storing {len(chunks)} chunks in vector database...")

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Generate embeddings for all chunks at once (faster than one by one)
    print("🔄 Generating embeddings for all chunks...")
    embeddings = get_embeddings_batch(chunks)

    # Create a unique ID for each chunk
    # Example: "report.pdf_chunk_0", "report.pdf_chunk_1", etc.
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

    # Store everything in ChromaDB
    collection.upsert(  # upsert = update if exists, insert if not
        ids=ids,
        documents=chunks,       # The actual text
        embeddings=embeddings,  # The vector fingerprints
        metadatas=[{"doc_name": doc_name, "chunk_index": i} 
                   for i in range(len(chunks))]  # Extra info about each chunk
    )

    print(f"✅ Successfully stored {len(chunks)} chunks!")


def search_similar_chunks(query: str) -> list:
    """
    Takes a question, finds the most relevant chunks from the database.
    Returns a list of the top matching text chunks.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Check if collection has any data
    if collection.count() == 0:
        return []

    # Convert question to vector
    query_embedding = get_embedding(query)

    # Find the closest chunks to the query vector
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K_RESULTS, collection.count())  # Get top 5 matches
    )

    # Extract just the text from the results
    chunks = results["documents"][0]  # [0] because we sent one query
    return chunks


def clear_collection():
    """
    Deletes all stored chunks. Used when a new PDF is uploaded.
    """
    client = get_chroma_client()
    # Get list of existing collections
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️ Vector store cleared!")
    else:
        print("ℹ️ No existing collection to clear — skipping.")


def get_collection_count() -> int:
    """
    Returns how many chunks are currently stored.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    return collection.count()


# --- TEST ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.pdf_processor import process_pdf
    import fitz

    # Step 1: Create a test PDF
    print("📝 Creating test PDF...")
    test_pdf_path = "test_vector.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), """
    Artificial Intelligence is transforming the enterprise landscape.
    Machine learning models can process vast amounts of data quickly.
    Natural language processing enables computers to understand human text.
    TCS is a leading IT services company headquartered in Mumbai.
    Data scientists use Python, SQL, and machine learning frameworks daily.
    RAG systems combine retrieval and generation for accurate answers.
    Vector databases store embeddings for semantic similarity search.
    """)
    doc.save(test_pdf_path)
    doc.close()

    # Step 2: Clear any old data
    clear_collection()

    # Step 3: Process and store
    chunks = process_pdf(test_pdf_path)
    store_chunks(chunks, "test_vector.pdf")

    # Step 4: Search
    print(f"\n🔍 Testing search...")
    print(f"Total chunks stored: {get_collection_count()}")

    query = "What does TCS do?"
    results = search_similar_chunks(query)

    print(f"\nQuery: '{query}'")
    print(f"Top result:")
    print(f"  {results[0]}")

    # Cleanup
    os.remove(test_pdf_path)
    clear_collection()

    print("\n🎉 vector_store.py is working correctly!")