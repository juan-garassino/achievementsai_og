# ingest.py
# ------------
import os
import nest_asyncio
import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings
)
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.node_parser.slide import SlideNodeParser
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.core.schema import MetadataMode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import (
    DynamicLLMPathExtractor,
    SchemaLLMPathExtractor,
    PropertyGraphIndex
)

# apply asyncio patch
nest_asyncio.apply()

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

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# --- Vector Index Ingestion ---
vector_reader = SimpleDirectoryReader(input_dir="../data/pdf")
vector_docs = vector_reader.load_data()
#slide_parser = SlideNodeParser.from_defaults(chunk_size=200, window_size=5, llm=llm)
transformations = [
    SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model),
    TitleExtractor(llm=llm, metadata_mode=MetadataMode.ALL),
    # SummaryExtractor(llm=llm, metadata_mode=MetadataMode.ALL),
    # QuestionsAnsweredExtractor(llm=llm, questions=3, metadata_mode=MetadataMode.ALL),
    # KeywordExtractor(llm=llm, keywords=5, metadata_mode=MetadataMode.ALL),
    embed_model,
]
# Create the ingestion pipeline
pipeline = IngestionPipeline(transformations=transformations)
vector_nodes = pipeline.run(documents=vector_docs)

vector_collection = chroma_client.get_or_create_collection("my-vectorstore")
vector_store = ChromaVectorStore(chroma_collection=vector_collection)
vector_storage = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex(nodes=vector_nodes, storage_context=vector_storage)
vector_index.storage_context.persist(persist_dir="./vector_storage")
print("Vector index persisted to './vector_storage'.")

# --- Graph Index Ingestion ---
graph_reader = SimpleDirectoryReader(input_dir="../data/testing")
graph_docs = graph_reader.load_data()
pipeline = IngestionPipeline(transformations=[
    SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model),
    TitleExtractor(llm=llm, metadata_mode=MetadataMode.ALL),
    embed_model
])
graph_nodes = pipeline.run(documents=graph_docs)

# Allowed entities and relations
allowed_entity_types = [
    "PERSON","PLACE","ORGANIZATION","EDUCATION","SKILL","ROLE",
    "PROJECT","TIME_PERIOD","LANGUAGE","TOOL","FRAMEWORK","METHOD",
    "INTEREST","PUBLICATION","EVENT","CERTIFICATION","TECHNOLOGY",
    "DISCIPLINE","ETHICAL_VALUE"
]
allowed_relation_types = [
    "WORKED_AT","STUDIED_AT","GRADUATED_FROM","TAUGHT_AT","LIVED_IN",
    "VISITED","COLLABORATED_WITH","FOUNDED","COFOUNDED","PARTICIPATED_IN",
    "SPECIALIZED_IN","USED","TAUGHT","AUTHORED","MENTORED",
    "INSPIRED_BY","HAS_SKILL","SPEAKS_LANGUAGE","PART_OF","DEVELOPED",
    "DEPLOYED","AWARDED","ASSOCIATED_WITH"
]

# Define relation constraints per entity type
schema = {
    "PERSON": [
        "WORKED_AT","STUDIED_AT","GRADUATED_FROM","TAUGHT_AT","COLLABORATED_WITH",
        "LIVED_IN","VISITED","SPEAKS_LANGUAGE","HAS_SKILL","USED","TAUGHT",
        "FOUNDED","COFOUNDED","PARTICIPATED_IN","SPECIALIZED_IN","WORKED_ON",
        "DEVELOPED","DEPLOYED","AWARDED","AUTHORED","MENTORED","INSPIRED_BY"
    ],
    "ORGANIZATION": ["PART_OF","ASSOCIATED_WITH","HOSTED"],
    "PROJECT": ["WORKED_ON","DEVELOPED","DEPLOYED","PART_OF"],
    "ROLE": ["PART_OF"],
    "SKILL": ["USED","TAUGHT"],
    "TOOL": ["USED"],
    "FRAMEWORK": ["USED"],
    "TECHNOLOGY": ["USED","DEPLOYED"],
    "PLACE": ["PART_OF","VISITED","LIVED_IN"],
    "EDUCATION": ["STUDIED_AT","GRADUATED_FROM"],
    "CERTIFICATION": ["HOLDS_CERTIFICATION"],
    "LANGUAGE": ["SPEAKS_LANGUAGE"],
    "DISCIPLINE": ["SPECIALIZED_IN","INTERESTED_IN"],
    "PUBLICATION": ["AUTHORED","FEATURED_IN"],
    "INTEREST": ["INTERESTED_IN"],
    "EVENT": ["PARTICIPATED_IN"]
}

# KG extractors
dynamic_extractor = DynamicLLMPathExtractor(
    llm=llm, max_triplets_per_chunk=20, num_workers=4,
    allowed_entity_types=allowed_entity_types,
    allowed_relation_types=allowed_relation_types
)
schema_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=tuple(allowed_entity_types),
    possible_relations=tuple(allowed_relation_types),
    kg_validation_schema=schema,
    strict=True,
    num_workers=4,
    max_triplets_per_chunk=10
)

# Chroma vector store for graph embeddings
graph_collection = chroma_client.get_or_create_collection("my-graphstore")
graph_vector_store = ChromaVectorStore(chroma_collection=graph_collection)

# Persist property graph: in-memory store
simple_graph_store = SimplePropertyGraphStore()
graph_index = PropertyGraphIndex(
    nodes=graph_nodes,
    embed_model=embed_model,
    property_graph_store=simple_graph_store,
    vector_store=graph_vector_store,
    kg_extractors=[dynamic_extractor],
    show_progress=True
)
graph_index.storage_context.persist(persist_dir="./graph_storage")
print("Graph index persisted to './graph_storage'.")

# Also push to Neo4j
neo4j_store = Neo4jPropertyGraphStore(
    url=neo4j_uri, username=neo4j_user, password=neo4j_pass
)
PropertyGraphIndex(
    nodes=vector_nodes,
    embed_model=embed_model,
    property_graph_store=neo4j_store,
    vector_store=graph_vector_store,
    kg_extractors=[schema_extractor],
    show_progress=True
)
print("Graph index pushed to Neo4j.")