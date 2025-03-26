import os
from typing import List, Dict, Any
import numpy as np

# Microsoft's GraphRAG package
from graphrag import GraphRAG
from graphrag.indexing import DRIFIndexer
from graphrag.retriever import GraphRetriever
from graphrag.models import OpenAIEmbeddings, AzureOpenAI

def setup_graphrag_system():
    """Initialize a GraphRAG system using Microsoft's package"""
    # Set up API credentials
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    # For Azure OpenAI
    # os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-api-key"
    # os.environ["AZURE_OPENAI_ENDPOINT"] = "your-azure-endpoint"
    
    # Initialize embedding model
    embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")
    
    # Initialize LLM
    llm = AzureOpenAI(
        deployment_name="gpt-35-turbo",
        temperature=0.0
    )
    
    # Initialize indexer with DRIF
    indexer = DRIFIndexer(
        embedding_model=embeddings,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Initialize retriever
    retriever = GraphRetriever(
        similarity_top_k=5,
        graph_hops=2,
        include_text=True
    )
    
    # Create GraphRAG system
    graphrag_system = GraphRAG(
        indexer=indexer,
        retriever=retriever,
        llm=llm
    )
    
    return graphrag_system

def add_documents(graphrag_system, documents):
    """Add documents to the GraphRAG system"""
    # Process and index documents
    graphrag_system.index(documents)
    
    # Build the knowledge graph
    graphrag_system.build_graph(
        similarity_threshold=0.7,
        create_sequential_edges=True,
        create_semantic_edges=True
    )
    
    print(f"Indexed {len(documents)} documents and built knowledge graph")
    print(f"Graph has {graphrag_system.graph.number_of_nodes()} nodes")

def save_graphrag_system(graphrag_system, directory):
    """Save the GraphRAG system"""
    graphrag_system.save(directory)
    print(f"Saved GraphRAG system to {directory}")

def load_graphrag_system(directory):
    """Load a saved GraphRAG system"""
    graphrag_system = GraphRAG.load(directory)
    print(f"Loaded GraphRAG system from {directory}")
    return graphrag_system

def query_graphrag(graphrag_system, query_text, use_graph=True):
    """Query the GraphRAG system"""
    # Set retrieval mode
    graphrag_system.retriever.use_graph = use_graph
    
    # Process query
    response = graphrag_system.query(query_text)
    
    return {
        "response": response.answer,
        "source_nodes": response.source_nodes,
        "graph_enhanced": use_graph
    }

# Example usage
if __name__ == "__main__":
    # Initialize system
    graphrag = setup_graphrag_system()
    
    # Example documents
    documents = [
        {
            "text": "GraphRAG is Microsoft's framework for graph-based retrieval augmented generation.",
            "metadata": {"source": "microsoft-docs", "id": "doc1"}
        },
        {
            "text": "DRIF indexing enhances document retrieval by creating optimized indices.",
            "metadata": {"source": "microsoft-docs", "id": "doc2"}
        }
    ]
    
    # Add documents
    add_documents(graphrag, documents)
    
    # Save system
    save_graphrag_system(graphrag, "./graphrag_system")
    
    # Load system
    loaded_graphrag = load_graphrag_system("./graphrag_system")
    
    # Query the system
    result = query_graphrag(loaded_graphrag, "How does GraphRAG improve retrieval?")
    print(f"Response: {result['response']}")