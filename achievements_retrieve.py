# retrieve.py
# ------------
import argparse
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings
import os

# Load environment variables
azure_api_key = os.getenv("AZURE_OPENAI_APIKEY")
llm_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
ellm_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_LLM")
embed_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING")
neo4j_uri = os.getenv("NEO4J_CONNECTION_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_pass = os.getenv("NEO4J_PASSWORD")

# Initialize LLM & embedding
llm = AzureOpenAI(
    model="gpt-4o-mini",
    deployment_name="gpt-4o-mini",
    api_key=azure_api_key,
    api_version=llm_api_version,
    azure_endpoint=ellm_endpoint,
    timeout=1220,
)
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-large",
    deployment_name="text-embedding-3-large",
    api_key=azure_api_key,
    azure_endpoint=embed_endpoint,
    api_version=llm_api_version,
    dimension=3072,
)

# Global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Initialize local Chroma client
client = chromadb.PersistentClient(path="./notebooks/chroma_db_2")

# Load vector index directly from Chroma
vector_collection = client.get_or_create_collection("my-vectorstore")
vector_store = ChromaVectorStore(chroma_collection=vector_collection)
vector_index = VectorStoreIndex.from_vector_store(vector_store)

# Load graph index with Chroma vector store and local property store
graph_collection = client.get_or_create_collection("my-graphstore")
graph_vector_store = ChromaVectorStore(chroma_collection=graph_collection)
prop_graph_store = SimplePropertyGraphStore.from_persist_dir("./notebooks/graph_storage")
graph_index = PropertyGraphIndex.from_existing(
    property_graph_store=prop_graph_store,
    vector_store=graph_vector_store,
    embed_kg_nodes=True
)


def query_vector(query: str):
    response = vector_index.as_query_engine().query(query)
    print(response)


def query_graph(query: str):
    response = graph_index.as_query_engine().query(query)
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query local vector and property-graph indexes via Chroma.")
    parser.add_argument("--vector", type=str, help="Run a query against the vector index")
    parser.add_argument("--graph", type=str, help="Run a query against the property graph index")
    args = parser.parse_args()

    if args.vector:
        query_vector(args.vector)
    if args.graph:
        query_graph(args.graph)
