"""Offline ingestion: embed ./data into a local Chroma vector store.

Extra deps (not needed by the Streamlit app):
    pip install chromadb llama-index-vector-stores-chroma llama-index-embeddings-huggingface
"""
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
STORAGE_DIR = "./storage"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def main():
    documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()

    db = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = db.get_or_create_collection("vector")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Chunk + embed + upsert into Chroma.
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=20), embed_model],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents)

    # Persist an index over the same store for later querying.
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"[ingest] {len(documents)} documents -> {len(nodes)} nodes in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
