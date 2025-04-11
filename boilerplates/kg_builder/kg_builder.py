#!/usr/bin/env python
"""
Biographical Knowledge Graph Builder

This script creates a knowledge graph from biographical documents using LlamaIndex
and stores it in a web-hosted Neo4j database.
Optimized for curriculum vitae and biographical information.
"""

import os
import sys
from typing import List, Optional, Dict, Any
import time
import getpass
try:
    from termcolor import colored
except ImportError:
    # Fallback if termcolor isn't installed
    def colored(text, color=None, **kwargs):
        return text

# LlamaIndex imports
try:
    print(colored("├── [*] Loading required modules", "cyan"))
    print(colored("│   └── [?] Importing LlamaIndex components for knowledge graph construction.", "yellow"))
    
    from llama_index.core import Document, Settings
    from llama_index.core.indices import PropertyGraphIndex
    from llama_index.core.indices.property_graph import (
        SimpleLLMPathExtractor,
        ImplicitPathExtractor, 
        DynamicLLMPathExtractor
    )
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.azure_openai import AzureOpenAI
    
    print(colored("│   └── [+] Successfully imported LlamaIndex components", "green"))
except ImportError as e:
    print(colored("│   └── [!] Required packages are not installed", "red"))
    print(colored(f"│       └── [!] Error: {str(e)}", "red"))
    print(colored("│       └── [?] The script requires LlamaIndex and related packages to function.", "yellow"))
    print("Required packages are not installed. Please install them with:")
    print("pip install llama-index llama-index-graph-stores-neo4j llama-index-llms-openai")
    sys.exit(1)

# Neo4j connection parameters
print(colored("├── [*] Setting up configuration parameters", "cyan"))
print(colored("│   └── [?] Loading database connection parameters from environment variables or using defaults.", "yellow"))

# Check for Neo4j credentials in environment variables
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

print(colored("│   └── [+] Configuration parameters loaded", "green"))

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """Load documents from text files in a directory."""
    print(colored("├── [*] Loading documents from directory", "cyan"))
    print(colored("│   └── [?] Scanning the specified directory for text files to process.", "yellow"))
    documents = []
    try:
        for filename in os.listdir(directory_path):
            if filename.endswith((".txt", ".md", ".pdf", ".docx")):
                filepath = os.path.join(directory_path, filename)
                print(colored(f"│   ├── [*] Processing {filepath}", "cyan"))
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        text = file.read()
                        documents.append(Document(text=text))
                        print(colored(f"│   │   └── [+] Loaded document from {filepath}", "green"))
                except UnicodeDecodeError:
                    print(colored(f"│   │   └── [!] Unicode decode error with {filepath}, trying different encoding", "yellow"))
                    try:
                        with open(filepath, 'r', encoding='latin-1') as file:
                            text = file.read()
                            documents.append(Document(text=text))
                            print(colored(f"│   │   └── [+] Loaded document from {filepath} with latin-1 encoding", "green"))
                    except Exception as e:
                        print(colored(f"│   │   └── [!] Failed to load {filepath}: {e}", "red"))
        print(colored(f"│   └── [+] Successfully loaded {len(documents)} documents", "green"))
    except Exception as e:
        print(colored(f"│   └── [!] Error loading documents: {e}", "red"))
        print(colored(f"│       └── [?] There was an issue accessing or reading files from the directory.", "yellow"))
    return documents

def load_document_from_file(file_path: str) -> Optional[Document]:
    """Load a single document from a text file."""
    print(colored(f"├── [*] Loading document from file: {file_path}", "cyan"))
    print(colored("│   └── [?] Reading a single document file for processing.", "yellow"))
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            print(colored(f"│   └── [+] Successfully loaded document from {file_path}", "green"))
            return Document(text=text)
    except UnicodeDecodeError:
        print(colored(f"│   ├── [!] Unicode decode error, trying different encoding", "yellow"))
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
                print(colored(f"│   └── [+] Successfully loaded document from {file_path} with latin-1 encoding", "green"))
                return Document(text=text)
        except Exception as e:
            print(colored(f"│   └── [!] Error loading document: {e}", "red"))
            return None
    except Exception as e:
        print(colored(f"│   └── [!] Error loading document from {file_path}: {e}", "red"))
        print(colored(f"│       └── [?] There was an issue reading the specified file.", "yellow"))
        return None

def initialize_llm(llm_type: str = "auto") -> Any:
    """Initialize the language model based on available credentials."""
    print(colored("├── [*] Initializing language model", "cyan"))
    print(colored("│   └── [?] Setting up the LLM that will be used for knowledge extraction.", "yellow"))
    
    # First, check if Azure OpenAI credentials are available
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-gs")
    
    # Then check if regular OpenAI credentials are available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # If llm_type is auto, determine based on available credentials
    if llm_type == "auto":
        print(colored("│   ├── [*] Auto-detecting LLM type based on available credentials", "cyan"))
        if all([azure_api_key, azure_api_version, azure_endpoint]):
            llm_type = "azure"
            print(colored("│   │   └── [+] Detected Azure OpenAI credentials", "green"))
        elif openai_api_key:
            llm_type = "openai"
            print(colored("│   │   └── [+] Detected OpenAI credentials", "green"))
        else:
            print(colored("│   │   └── [!] No OpenAI or Azure OpenAI credentials found", "red"))
            raise ValueError("No OpenAI or Azure OpenAI credentials found in environment variables")
    
    # Initialize the chosen LLM
    if llm_type == "azure":
        print(colored("│   ├── [*] Setting up Azure OpenAI", "cyan"))
        if not all([azure_api_key, azure_api_version, azure_endpoint]):
            print(colored("│   │   └── [!] Missing Azure OpenAI credentials", "red"))
            raise ValueError("Missing Azure OpenAI credentials in environment variables")
        
        print(colored(f"│   │   └── [?] Using Azure OpenAI with deployment {azure_deployment}.", "yellow"))
        try:
            llm = AzureOpenAI(
                model="gpt-4o",
                deployment_name=azure_deployment,
                api_key=azure_api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
                temperature=0.0
            )
            print(colored("│   └── [+] Azure OpenAI initialized successfully", "green"))
            return llm
        except Exception as e:
            print(colored(f"│   └── [!] Error initializing Azure OpenAI: {e}", "red"))
            print(colored("│       └── [?] Falling back to OpenAI if credentials are available", "yellow"))
            if not openai_api_key:
                raise
            llm_type = "openai"
    
    if llm_type == "openai":
        print(colored("│   ├── [*] Setting up OpenAI", "cyan"))
        if not openai_api_key:
            print(colored("│   │   └── [!] Missing OpenAI API key", "red"))
            raise ValueError("Missing OpenAI API key in environment variables")
        
        print(colored("│   │   └── [?] Using OpenAI with gpt-3.5-turbo model.", "yellow"))
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
        print(colored("│   └── [+] OpenAI initialized successfully", "green"))
        return llm
    
    else:
        print(colored(f"│   └── [!] Unknown or unsupported LLM type: {llm_type}", "red"))
        raise ValueError(f"Unknown or unsupported LLM type: {llm_type}")

def get_neo4j_credentials():
    """Get Neo4j credentials from user if not provided in environment variables."""
    global NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    
    if not NEO4J_URI:
        print(colored("│   ├── [*] Neo4j URI not found in environment variables", "yellow"))
        NEO4J_URI = input("Enter Neo4j URI (e.g., neo4j+s://xyz.databases.neo4j.io): ")
    
    if not NEO4J_USERNAME:
        print(colored("│   ├── [*] Neo4j username not found in environment variables", "yellow"))
        NEO4J_USERNAME = input("Enter Neo4j username: ")
    
    if not NEO4J_PASSWORD:
        print(colored("│   ├── [*] Neo4j password not found in environment variables", "yellow"))
        NEO4J_PASSWORD = getpass.getpass("Enter Neo4j password: ")
    
    return NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def initialize_neo4j_graph_store():
    """Initialize the Neo4j graph store with proper credentials."""
    print(colored("├── [*] Initializing graph database connection", "cyan"))
    print(colored("│   └── [?] Setting up the connection to the Neo4j database.", "yellow"))
    
    uri, username, password = get_neo4j_credentials()
    
    if not all([uri, username, password]):
        print(colored("│   └── [!] Missing Neo4j credentials", "red"))
        raise ValueError("Neo4j credentials are required")
    
    print(colored(f"│   ├── [*] Connecting to Neo4j at {uri}", "cyan"))
    try:
        store = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=uri,
        )
        print(colored("│   └── [+] Neo4j connection established successfully", "green"))
        return store
    except Exception as e:
        print(colored(f"│   └── [!] Failed to connect to Neo4j: {e}", "red"))
        raise

def initialize_biographical_extractors(llm, extractor_types: List[str], **kwargs) -> List[Any]:
    """Initialize path extractors optimized for biographical data."""
    print(colored("├── [*] Initializing biographical knowledge extractors", "cyan"))
    print(colored("│   └── [?] Setting up specialized extractors for biographical texts.", "yellow"))
    
    extractors = []
    
    # Set default parameters
    num_workers = kwargs.get("num_workers", 4)
    max_paths = kwargs.get("max_paths", 10)
    
    # Define biographical entity and relation types
    bio_entity_types = [
        "PERSON", "ORGANIZATION", "LOCATION", "SKILL", "EDUCATION", 
        "PROJECT", "ACHIEVEMENT", "ROLE", "PUBLICATION", "COURSE", 
        "INTEREST", "EVENT", "TECHNOLOGY", "CERTIFICATION"
    ]
    
    bio_relation_types = [
        "WORKED_AT", "STUDIED_AT", "LIVED_IN", "HAS_SKILL", "EARNED",
        "CONTRIBUTED_TO", "KNOWS", "PUBLISHED", "TAUGHT", "INTERESTED_IN",
        "PARTICIPATED_IN", "HELD_ROLE", "AWARDED_BY", "RELATED_TO",
        "CERTIFIED_IN", "MENTORED", "MANAGED", "FOUNDED", "VOLUNTEERED_AT",
        "SPECIALIZES_IN", "COLLABORATED_WITH", "SPEAKS", "DEVELOPED"
    ]
    
    # Use any user-provided overrides
    entity_types = kwargs.get("entity_types", bio_entity_types)
    relation_types = kwargs.get("relation_types", bio_relation_types)

    print(colored(f"│   ├── [*] Configuring {len(extractor_types)} extractors", "cyan"))
    for extractor_type in extractor_types:
        if extractor_type == "simple":
            print(colored("│   │   ├── [*] Setting up SimpleLLMPathExtractor", "cyan"))
            print(colored("│   │   │   └── [?] This extractor uses a simple approach to identify relationships between entities in the text.", "yellow"))
            extractors.append(
                SimpleLLMPathExtractor(
                    llm=llm,
                    max_paths_per_chunk=max_paths,
                    num_workers=num_workers,
                    show_progress=True
                )
            )
            print(colored("│   │   │   └── [+] SimpleLLMPathExtractor initialized", "green"))
        
        elif extractor_type == "implicit":
            print(colored("│   │   ├── [*] Setting up ImplicitPathExtractor", "cyan"))
            print(colored("│   │   │   └── [?] This extractor identifies implicit relationships that may not be explicitly stated in the text.", "yellow"))
            extractors.append(ImplicitPathExtractor())
            print(colored("│   │   │   └── [+] ImplicitPathExtractor initialized", "green"))
        
        elif extractor_type == "dynamic":
            print(colored("│   │   ├── [*] Setting up DynamicLLMPathExtractor", "cyan"))
            print(colored("│   │   │   └── [?] This extractor dynamically identifies entities and relationships based on the biographical context.", "yellow"))
            extractors.append(
                DynamicLLMPathExtractor(
                    llm=llm,
                    max_triplets_per_chunk=max_paths,
                    num_workers=num_workers,
                    allowed_entity_types=entity_types,
                    allowed_relation_types=relation_types,
                )
            )
            print(colored("│   │   │   └── [+] DynamicLLMPathExtractor initialized", "green"))
        
        elif extractor_type == "schema":
            print(colored("│   │   ├── [!] SchemaLLMPathExtractor not available with Pydantic v2", "yellow"))
            print(colored("│   │   │   └── [?] Using DynamicLLMPathExtractor as a fallback.", "yellow"))
            extractors.append(
                DynamicLLMPathExtractor(
                    llm=llm,
                    max_triplets_per_chunk=max_paths,
                    num_workers=num_workers,
                    allowed_entity_types=entity_types,
                    allowed_relation_types=relation_types,
                )
            )
            print(colored("│   │   │   └── [+] DynamicLLMPathExtractor initialized as fallback", "green"))
    
    if not extractors:
        print(colored("│   ├── [!] No valid extractors specified, using default extractors", "yellow"))
        extractors.append(
            DynamicLLMPathExtractor(
                llm=llm,
                max_triplets_per_chunk=max_paths,
                num_workers=num_workers,
                allowed_entity_types=entity_types,
                allowed_relation_types=relation_types,
            )
        )
        extractors.append(ImplicitPathExtractor())
        print(colored("│   │   └── [+] Default extractors initialized", "green"))
    
    print(colored(f"│   └── [+] Successfully initialized {len(extractors)} biographical extractors", "green"))
    return extractors

def create_biographical_kg(llm_type: str = "auto", extractor_types: List[str] = None, **kwargs):
    """Create a biographical knowledge graph and store it in Neo4j."""
    print(colored("├── [*] Starting biographical knowledge graph creation workflow", "cyan"))
    print(colored("│   └── [?] This process will extract biographical information from documents and create a structured knowledge graph.", "yellow"))
    
    # Set default extractors if none provided
    if not extractor_types:
        extractor_types = ["dynamic", "implicit"]
    
    try:
        # Initialize LLM
        print(colored("│   ├── [*] Setting up language model", "cyan"))
        llm = initialize_llm(llm_type)
        if not llm:
            print(colored("│   │   └── [!] Failed to initialize language model", "red"))
            return None
        
        Settings.llm = llm
        print(colored("│   │   └── [+] Language model configured as global setting", "green"))
        
        # Initialize graph store
        print(colored("│   ├── [*] Setting up graph database", "cyan"))
        graph_store = initialize_neo4j_graph_store()
        if not graph_store:
            print(colored("│   │   └── [!] Failed to initialize graph store", "red"))
            return None
        print(colored("│   │   └── [+] Graph database connection established", "green"))
        
        # Initialize biographical extractors
        print(colored("│   ├── [*] Configuring knowledge extractors", "cyan"))
        kg_extractors = initialize_biographical_extractors(llm, extractor_types, **kwargs)
        if not kg_extractors:
            print(colored("│   │   └── [!] No valid extractors configured", "red"))
            return None
        print(colored("│   │   └── [+] Knowledge extractors configured successfully", "green"))
        
        # Load documents based on provided options
        print(colored("│   ├── [*] Loading input documents", "cyan"))
        documents = []
        
        # Check for file option
        file_path = kwargs.get("file")
        if file_path:
            print(colored(f"│   │   ├── [*] Loading from file: {file_path}", "cyan"))
            document = load_document_from_file(file_path)
            if document:
                documents = [document]
                print(colored("│   │   │   └── [+] Document loaded successfully", "green"))
        
        # Check for directory option
        dir_path = kwargs.get("dir")
        if dir_path and not documents:
            print(colored(f"│   │   ├── [*] Loading from directory: {dir_path}", "cyan"))
            documents = load_documents_from_directory(dir_path)
            print(colored(f"│   │   │   └── [+] Loaded {len(documents)} documents from directory", "green"))
        
        if not documents:
            print(colored("│   │   └── [!] No documents found", "red"))
            print(colored("│   │       └── [?] The workflow requires at least one document to process. Please provide a valid file or directory.", "yellow"))
            return None
        
        print(colored(f"│   │   └── [+] Successfully loaded {len(documents)} documents for processing", "green"))
        
        print(colored(f"│   ├── [*] Building knowledge graph from documents", "cyan"))
        print(colored(f"│   │   └── [?] This step will analyze the documents and extract entities and relationships. This may take some time.", "yellow"))
        
        # Create the knowledge graph index
        try:
            start_time = time.time()
            index = PropertyGraphIndex.from_documents(
                documents,
                property_graph_store=graph_store,
                kg_extractors=kg_extractors,
                show_progress=True,
                embed_kg_nodes=False,  # Disable embedding to reduce compatibility issues
            )
            elapsed_time = time.time() - start_time
            print(colored(f"│   │   └── [+] Knowledge graph built successfully in {elapsed_time:.2f} seconds", "green"))
        except Exception as e:
            print(colored(f"│   │   └── [!] Error building knowledge graph: {e}", "red"))
            import traceback
            traceback.print_exc()
            print(colored("│   │       └── [?] Trying alternative approach with fewer extractors...", "yellow"))
            
            # Try again with just the implicit extractor which is most reliable
            try:
                kg_extractors = [ImplicitPathExtractor()]
                index = PropertyGraphIndex.from_documents(
                    documents,
                    property_graph_store=graph_store,
                    kg_extractors=kg_extractors,
                    show_progress=True,
                    embed_kg_nodes=False,
                )
                print(colored("│   │       └── [+] Knowledge graph built successfully with fallback method", "green"))
            except Exception as e2:
                print(colored(f"│   │       └── [!] Fallback method also failed: {e2}", "red"))
                return None
        
        # Test the knowledge graph with a biographical query
        print(colored("│   ├── [*] Testing knowledge graph with sample queries", "cyan"))
        print(colored("│   │   └── [?] Running test queries to verify the graph was built correctly.", "yellow"))
        
        try:
            query_engine = index.as_query_engine(include_text=True)
            
            test_queries = [
                "What skills does this person have?",
                "Where did this person work?",
                "What educational background does this person have?"
            ]
            
            for i, query in enumerate(test_queries):
                print(colored(f"│   │   ├── [*] Running test query {i+1}: '{query}'", "cyan"))
                response = query_engine.query(query)
                print(colored(f"│   │   │   └── [+] Response received: {str(response)[:100]}...", "green"))
        except Exception as e:
            print(colored(f"│   │   └── [!] Error testing knowledge graph: {e}", "red"))
            print(colored("│   │       └── [?] This may be due to compatibility issues with the query engine, but the graph was still built.", "yellow"))
        
        # Example of retrieving paths from the graph
        print(colored("│   ├── [*] Retrieving knowledge graph paths", "cyan"))
        print(colored("│   │   └── [?] This demonstrates how to retrieve specific information from the graph.", "yellow"))
        
        try:
            retriever = index.as_retriever(include_text=True, similarity_top_k=3)
            nodes = retriever.retrieve("skills and professional experience")
            for i, node in enumerate(nodes):
                print(colored(f"│   │   ├── [*] Retrieved node {i+1}", "cyan"))
                print(colored(f"│   │   │   └── [+] {str(node)[:100]}...", "green"))
        except Exception as e:
            print(colored(f"│   │   └── [!] Error retrieving from knowledge graph: {e}", "red"))
        
        print(colored(f"│   └── [+] Biographical knowledge graph created and stored in Neo4j successfully!", "green"))
        
        return index
    
    except Exception as e:
        print(colored(f"│   └── [!] Error creating knowledge graph: {e}", "red"))
        print(colored(f"│       └── [?] An unexpected error occurred during the workflow. Check the traceback for details.", "yellow"))
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point for the script with command line arguments."""
    print(colored("├── [*] Starting Biographical Knowledge Graph Builder", "cyan"))
    print(colored("│   └── [?] This tool extracts biographical information from documents and creates a structured knowledge graph in a graph database.", "yellow"))
    
    import argparse
    
    print(colored("│   ├── [*] Parsing command line arguments", "cyan"))
    parser = argparse.ArgumentParser(description="Create a biographical knowledge graph using Neo4j and LlamaIndex")
    parser.add_argument("--llm", choices=["openai", "azure", "auto"], default="auto", help="LLM provider to use")
    parser.add_argument("--neo4j-uri", default="neo4j+s://2dc091a7.databases.neo4j.io", help="Neo4j URI (e.g., neo4j+s://example.databases.neo4j.io)")
    parser.add_argument("--neo4j-username", help="Neo4j username")
    parser.add_argument("--neo4j-password", help="Neo4j password")
    
    # Document options
    parser.add_argument("--file", help="Path to a specific biographical text file to process")
    parser.add_argument("--dir", help="Path to a directory of biographical text files to process")
    
    # Extractor options
    parser.add_argument(
        "--extractors", 
        nargs="+", 
        choices=["simple", "implicit", "dynamic", "schema"], 
        default=["dynamic", "implicit"],
        help="List of extractors to use (default: dynamic implicit)"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel extraction")
    parser.add_argument("--max-paths", type=int, default=15, help="Maximum paths per chunk")
    
    args = parser.parse_args()
    print(colored("│   │   └── [+] Command line arguments parsed successfully", "green"))
    
    # Update global variables if provided as command line arguments
    global NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    if args.neo4j_uri:
        NEO4J_URI = args.neo4j_uri
    if args.neo4j_username:
        NEO4J_USERNAME = args.neo4j_username
    if args.neo4j_password:
        NEO4J_PASSWORD = args.neo4j_password
    
    # Prepare kwargs for other components
    print(colored("│   ├── [*] Preparing workflow parameters", "cyan"))
    kwargs = {
        "file": args.file,
        "dir": args.dir,
        "num_workers": args.num_workers,
        "max_paths": args.max_paths,
    }
    print(colored("│   │   └── [+] Workflow parameters prepared", "green"))
    
    # Create the biographical knowledge graph
    print(colored("│   └── [*] Executing knowledge graph creation workflow", "cyan"))
    result = create_biographical_kg(args.llm, args.extractors, **kwargs)
    
    if result:
        print(colored("└── [+] Workflow completed successfully", "green"))
    else:
        print(colored("└── [!] Workflow completed with errors", "red"))

if __name__ == "__main__":
    print(colored("┌── [*] Initializing Biographical Knowledge Graph Builder", "cyan"))
    print(colored("│   └── [?] This script will extract entities and relationships from biographical documents and create a structured knowledge graph.", "yellow"))
    main()


# python kg_builder.py --extractors dynamic --file juan-complete-biography.md