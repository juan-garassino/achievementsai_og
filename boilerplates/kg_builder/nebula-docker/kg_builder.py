#!/usr/bin/env python
"""
Comprehensive Knowledge Graph Builder

This script creates a knowledge graph from biographical documents using LlamaIndex
and offers multiple database options (Neo4j or NebulaGraph, local or remote),
multiple LLM options (OpenAI or Azure OpenAI), and all path extractor types.
"""

import os
import sys
from typing import List, Optional, Dict, Any, Tuple, Union
import time
import getpass
import subprocess
try:
    from termcolor import colored
except ImportError:
    # Fallback if termcolor isn't installed
    def colored(text, color=None, **kwargs):
        return text

print(colored("┌── [*] Initializing Comprehensive Knowledge Graph Builder", "cyan"))
print(colored("│   └── [?] This script supports Neo4j/NebulaGraph, OpenAI/Azure, and all path extractors", "yellow"))

# Global variable declarations
# These need to be defined at the module level first
DB_MODE = "remote"  # remote or local
DB_TYPE = "neo4j"   # neo4j or nebula

# Neo4j parameters
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_LOCAL_PORT = os.environ.get("NEO4J_LOCAL_PORT", "7687")
NEO4J_DOCKER_NAME = os.environ.get("NEO4J_DOCKER_NAME", "neo4j")

# NebulaGraph parameters
NEBULA_SPACE = os.environ.get("NEBULA_SPACE", "llamaindex_property_graph")
NEBULA_HOST = os.environ.get("NEBULA_HOST", "localhost")
NEBULA_PORT = int(os.environ.get("NEBULA_PORT", "9669"))
NEBULA_USERNAME = os.environ.get("NEBULA_USERNAME", "root")
NEBULA_PASSWORD = os.environ.get("NEBULA_PASSWORD", "nebula")
NEBULA_DOCKER_NAME = os.environ.get("NEBULA_DOCKER_NAME", "nebula-graphd")

# Feature flags
NEBULA_AVAILABLE = False
SCHEMA_EXTRACTOR_AVAILABLE = False

# Check required packages
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
    
    # Neo4j 
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
    
    # OpenAI
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.azure_openai import AzureOpenAI
    
    # Check for NebulaGraph support
    try:
        from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
        NEBULA_AVAILABLE = True
        print(colored("│   ├── [+] NebulaGraph support available", "green"))
    except ImportError:
        NEBULA_AVAILABLE = False
        print(colored("│   ├── [!] NebulaGraph support not available", "yellow"))
        print(colored("│   │   └── [?] To enable NebulaGraph support, install: pip install llama-index-graph-stores-nebula", "yellow"))
    
    # Check for SchemaLLMPathExtractor support with Pydantic v2 compatibility
    try:
        # Try importing with fallback mechanism to handle potential Pydantic v2 incompatibilities
        try:
            from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
            SCHEMA_EXTRACTOR_AVAILABLE = True
            print(colored("│   ├── [+] SchemaLLMPathExtractor support available", "green"))
        except ImportError as e:
            if "model_validator" in str(e):
                # This is a Pydantic v2 compatibility issue
                print(colored("│   ├── [!] SchemaLLMPathExtractor not compatible with Pydantic v2", "yellow"))
                SCHEMA_EXTRACTOR_AVAILABLE = False
            else:
                raise
    except Exception:
        SCHEMA_EXTRACTOR_AVAILABLE = False
        print(colored("│   ├── [!] SchemaLLMPathExtractor not available", "yellow"))
        print(colored("│   │   └── [?] Will use DynamicLLMPathExtractor as a fallback", "yellow"))
    
    print(colored("│   └── [+] Successfully imported LlamaIndex components", "green"))
except ImportError as e:
    print(colored("│   └── [!] Required packages are not installed", "red"))
    print(colored(f"│       └── [!] Error: {str(e)}", "red"))
    print(colored("│       └── [?] The script requires LlamaIndex and related packages to function.", "yellow"))
    print("Required packages are not installed. Please install them with:")
    print("pip install llama-index llama-index-graph-stores-neo4j llama-index-llms-openai")
    print("For additional features: pip install llama-index-graph-stores-nebula")
    sys.exit(1)

# Connection parameters
print(colored("├── [*] Setting up configuration parameters", "cyan"))
print(colored("│   └── [?] Loading database connection parameters from environment variables or using defaults.", "yellow"))
print(colored("│   └── [+] Configuration parameters loaded", "green"))

def check_neo4j_local_status():
    """Check if Neo4j is running locally via Docker."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={NEO4J_DOCKER_NAME}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        return "Up" in result.stdout
    except Exception:
        return False

def check_nebula_local_status():
    """Check if NebulaGraph is running locally via Docker."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={NEBULA_DOCKER_NAME}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        return "Up" in result.stdout
    except Exception:
        return False

def check_nebula_space_exists(space_name):
    """Check if the specified NebulaGraph space exists."""
    try:
        # Use nebula-console to check if space exists
        result = subprocess.run([
            "docker", "exec", NEBULA_DOCKER_NAME, 
            "nebula-console", "-addr", "127.0.0.1", "-port", "9669", 
            "-u", NEBULA_USERNAME, "-p", NEBULA_PASSWORD, 
            "-e", f"SHOW SPACES"
        ], capture_output=True, text=True)
        return space_name in result.stdout
    except Exception:
        return False

def create_nebula_space(space_name):
    """Create a new NebulaGraph space if it doesn't exist."""
    try:
        # Create space command
        create_cmd = f"CREATE SPACE IF NOT EXISTS {space_name}(vid_type=FIXED_STRING(256))"
        
        # Run command via nebula-console
        result = subprocess.run([
            "docker", "exec", NEBULA_DOCKER_NAME, 
            "nebula-console", "-addr", "127.0.0.1", "-port", "9669", 
            "-u", NEBULA_USERNAME, "-p", NEBULA_PASSWORD, 
            "-e", create_cmd
        ], capture_output=True, text=True)
        
        if "created successfully" in result.stdout or "already exists" in result.stdout:
            return True
        else:
            print(colored(f"│   │   └── [!] Error creating space: {result.stdout}", "red"))
            return False
    except Exception as e:
        print(colored(f"│   │   └── [!] Error running nebula-console: {e}", "red"))
        return False

def start_neo4j_local():
    """Start Neo4j locally using Docker."""
    print(colored("├── [*] Starting local Neo4j instance with Docker", "cyan"))
    
    if check_neo4j_local_status():
        print(colored("│   └── [+] Local Neo4j is already running", "green"))
        return True
    
    try:
        # Check if the container exists but is stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={NEO4J_DOCKER_NAME}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            # Container exists, start it
            print(colored("│   ├── [*] Starting existing Neo4j container", "cyan"))
            subprocess.run(["docker", "start", NEO4J_DOCKER_NAME], check=True)
        else:
            # Container doesn't exist, create and start it
            print(colored("│   ├── [*] Creating new Neo4j container", "cyan"))
            subprocess.run([
                "docker", "run", 
                "--name", NEO4J_DOCKER_NAME,
                "-p", f"{NEO4J_LOCAL_PORT}:7687", 
                "-p", "7474:7474",
                "-e", "NEO4J_AUTH=neo4j/password",
                "-d", "neo4j:latest"
            ], check=True)
        
        # Wait for Neo4j to start
        print(colored("│   ├── [*] Waiting for Neo4j to initialize (this may take a few seconds)", "cyan"))
        for _ in range(10):
            if check_neo4j_local_status():
                print(colored("│   └── [+] Local Neo4j started successfully", "green"))
                print(colored("│       └── [?] Default credentials: username=neo4j, password=password", "yellow"))
                return True
            time.sleep(2)
        
        print(colored("│   └── [!] Neo4j container started but service might not be ready yet", "yellow"))
        return True
    except subprocess.CalledProcessError as e:
        print(colored(f"│   └── [!] Failed to start Neo4j with Docker: {e}", "red"))
        print(colored("│       └── [?] Make sure Docker is installed and running", "yellow"))
        return False
    except Exception as e:
        print(colored(f"│   └── [!] Unexpected error starting Neo4j: {e}", "red"))
        return False

def start_nebula_local():
    """Start NebulaGraph locally using Docker."""
    print(colored("├── [*] Starting local NebulaGraph instance with Docker", "cyan"))
    
    if check_nebula_local_status():
        print(colored("│   └── [+] Local NebulaGraph is already running", "green"))
        return True
    
    try:
        # Check if the container exists but is stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={NEBULA_DOCKER_NAME}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            # Container exists, start it
            print(colored("│   ├── [*] Starting existing NebulaGraph container", "cyan"))
            subprocess.run(["docker", "start", NEBULA_DOCKER_NAME], check=True)
        else:
            # Container doesn't exist, create and start it
            print(colored("│   ├── [*] Creating new NebulaGraph container", "cyan"))
            # Using the official docker run command for NebulaGraph
            subprocess.run([
                "docker", "run", 
                "--name", NEBULA_DOCKER_NAME,
                "-p", "9669:9669", 
                "-p", "19669:19669",
                "-p", "19670:19670",
                "-d", "vesoft/nebula-graphd:latest"
            ], check=True)
        
        # Wait for NebulaGraph to start
        print(colored("│   ├── [*] Waiting for NebulaGraph to initialize (this may take a few seconds)", "cyan"))
        for _ in range(10):
            if check_nebula_local_status():
                print(colored("│   └── [+] Local NebulaGraph started successfully", "green"))
                print(colored("│       └── [?] Default credentials: username=root, password=nebula", "yellow"))
                
                # Create space if it doesn't exist
                print(colored("│   ├── [*] Setting up NebulaGraph space", "cyan"))
                if create_nebula_space(NEBULA_SPACE):
                    print(colored(f"│   └── [+] NebulaGraph space '{NEBULA_SPACE}' is ready", "green"))
                    return True
                else:
                    print(colored(f"│   └── [!] Failed to create NebulaGraph space", "red"))
                    return False
                
            time.sleep(2)
        
        print(colored("│   └── [!] NebulaGraph container started but service might not be ready yet", "yellow"))
        return True
    except subprocess.CalledProcessError as e:
        print(colored(f"│   └── [!] Failed to start NebulaGraph with Docker: {e}", "red"))
        print(colored("│       └── [?] Make sure Docker is installed and running", "yellow"))
        return False
    except Exception as e:
        print(colored(f"│   └── [!] Unexpected error starting NebulaGraph: {e}", "red"))
        return False

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
    """Get Neo4j credentials based on the mode (local or remote)."""
    global NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    
    if DB_MODE == "local":
        print(colored("│   ├── [*] Using local Neo4j instance", "cyan"))
        # Set default local credentials
        if not NEO4J_URI:
            NEO4J_URI = f"bolt://localhost:{NEO4J_LOCAL_PORT}"
        if not NEO4J_USERNAME:
            NEO4J_USERNAME = "neo4j"
        if not NEO4J_PASSWORD:
            NEO4J_PASSWORD = "password"  # Default password for Docker Neo4j
        
        print(colored(f"│   │   └── [?] Connecting to local Neo4j at {NEO4J_URI}", "yellow"))
    else:  # remote mode
        print(colored("│   ├── [*] Using remote Neo4j instance", "cyan"))
        if not NEO4J_URI:
            print(colored("│   │   └── [?] Neo4j URI not found in environment variables", "yellow"))
            NEO4J_URI = input("Enter Neo4j URI (e.g., neo4j+s://xyz.databases.neo4j.io): ")
        
        if not NEO4J_USERNAME:
            print(colored("│   │   └── [?] Neo4j username not found in environment variables", "yellow"))
            NEO4J_USERNAME = input("Enter Neo4j username: ")
        
        if not NEO4J_PASSWORD:
            print(colored("│   │   └── [?] Neo4j password not found in environment variables", "yellow"))
            NEO4J_PASSWORD = getpass.getpass("Enter Neo4j password: ")
    
    return NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def get_nebula_credentials():
    """Get NebulaGraph credentials based on the mode (local or remote)."""
    global NEBULA_SPACE, NEBULA_HOST, NEBULA_PORT, NEBULA_USERNAME, NEBULA_PASSWORD
    
    if DB_MODE == "local":
        print(colored("│   ├── [*] Using local NebulaGraph instance", "cyan"))
        # Set default local credentials
        if not NEBULA_HOST:
            NEBULA_HOST = "localhost"
        if not NEBULA_PORT:
            NEBULA_PORT = 9669
        if not NEBULA_USERNAME:
            NEBULA_USERNAME = "root"
        if not NEBULA_PASSWORD:
            NEBULA_PASSWORD = "nebula"  # Default password for Docker NebulaGraph
        
        print(colored(f"│   │   └── [?] Connecting to local NebulaGraph at {NEBULA_HOST}:{NEBULA_PORT}", "yellow"))
    else:  # remote mode
        print(colored("│   ├── [*] Using remote NebulaGraph instance", "cyan"))
        if not NEBULA_HOST:
            print(colored("│   │   └── [?] NebulaGraph host not found in environment variables", "yellow"))
            NEBULA_HOST = input("Enter NebulaGraph host: ")
        
        if not NEBULA_PORT:
            print(colored("│   │   └── [?] NebulaGraph port not found in environment variables", "yellow"))
            NEBULA_PORT = int(input("Enter NebulaGraph port (default: 9669): ") or "9669")
        
        if not NEBULA_SPACE:
            print(colored("│   │   └── [?] NebulaGraph space not found in environment variables", "yellow"))
            NEBULA_SPACE = input("Enter NebulaGraph space name: ")
        
        if not NEBULA_USERNAME:
            print(colored("│   │   └── [?] NebulaGraph username not found in environment variables", "yellow"))
            NEBULA_USERNAME = input("Enter NebulaGraph username: ")
        
        if not NEBULA_PASSWORD:
            print(colored("│   │   └── [?] NebulaGraph password not found in environment variables", "yellow"))
            NEBULA_PASSWORD = getpass.getpass("Enter NebulaGraph password: ")
    
    return NEBULA_SPACE, NEBULA_HOST, NEBULA_PORT, NEBULA_USERNAME, NEBULA_PASSWORD

def initialize_graph_store():
    """Initialize the appropriate graph store (Neo4j or NebulaGraph)."""
    print(colored("├── [*] Initializing graph database connection", "cyan"))
    print(colored(f"│   └── [?] Setting up connection to the {DB_TYPE} database.", "yellow"))
    
    if DB_TYPE == "neo4j":
        return initialize_neo4j_graph_store()
    elif DB_TYPE == "nebula":
        if not NEBULA_AVAILABLE:
            print(colored("│   └── [!] NebulaGraph support not available. Please install the required package.", "red"))
            print(colored("│       └── [?] Run: pip install llama-index-graph-stores-nebula", "yellow"))
            raise ImportError("NebulaGraph support not available")
        return initialize_nebula_graph_store()
    else:
        print(colored(f"│   └── [!] Unsupported database type: {DB_TYPE}", "red"))
        raise ValueError(f"Unsupported database type: {DB_TYPE}")

def initialize_neo4j_graph_store():
    """Initialize the Neo4j graph store with proper credentials."""
    global DB_MODE
    
    # Start local Neo4j if needed
    if DB_MODE == "local":
        if not start_neo4j_local():
            print(colored("│   └── [!] Failed to start local Neo4j instance", "red"))
            if input("Would you like to switch to remote mode? (y/n): ").lower() == 'y':
                DB_MODE = "remote"
            else:
                raise ValueError("Local Neo4j setup failed")
    
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

def initialize_nebula_graph_store():
    """Initialize the NebulaGraph store with proper credentials."""
    global DB_MODE
    
    # Start local NebulaGraph if needed
    if DB_MODE == "local":
        if not start_nebula_local():
            print(colored("│   └── [!] Failed to start local NebulaGraph instance", "red"))
            if input("Would you like to switch to remote mode? (y/n): ").lower() == 'y':
                DB_MODE = "remote"
            else:
                raise ValueError("Local NebulaGraph setup failed")
    
    space, host, port, username, password = get_nebula_credentials()
    
    if not all([space, host, username, password]):
        print(colored("│   └── [!] Missing NebulaGraph credentials", "red"))
        raise ValueError("NebulaGraph credentials are required")
    
    print(colored(f"│   ├── [*] Connecting to NebulaGraph space '{space}'", "cyan"))
    try:
        # Current version of NebulaPropertyGraphStore API
        store = NebulaPropertyGraphStore(
            space=space,
            overwrite=True
        )
        print(colored("│   └── [+] NebulaGraph connection established successfully", "green"))
        return store
    except Exception as e:
        print(colored(f"│   └── [!] Failed to connect to NebulaGraph: {e}", "red"))
        
        # Debug information to help with connection issues
        print(colored(f"│   ├── [*] NebulaGraph connection details for debugging:", "yellow"))
        print(colored(f"│   │   ├── Space: {space}", "yellow"))
        print(colored(f"│   │   ├── Host: {host}", "yellow"))
        print(colored(f"│   │   ├── Port: {port}", "yellow"))
        print(colored(f"│   │   └── Username: {username}", "yellow"))
        
        # Check if the issue might be the API has changed
        if "hosts" in str(e):
            print(colored("│   ├── [!] API parameter issue detected", "red"))
            print(colored("│   │   └── [?] The NebulaGraph API appears to have changed. Try updating the package.", "yellow"))
        
        # Inspect API signature
        try:
            import inspect
            sig = inspect.signature(NebulaPropertyGraphStore.__init__)
            print(colored("│   ├── [*] NebulaPropertyGraphStore API signature:", "yellow"))
            print(colored(f"│   │   └── {sig}", "yellow"))
        except Exception:
            pass
        
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
    
    # Define a curriculum validation schema for logical connections
    curriculum_validation_schema = {
        "PERSON": ["WORKED_AT", "STUDIED_AT", "LIVED_IN", "HAS_SKILL"],
        "ORGANIZATION": ["AWARDED_BY", "RELATED_TO"],
        "SKILL": ["RELATED_TO"],
        "EDUCATION": ["RELATED_TO"]
    }
    
    # Use any user-provided overrides
    entity_types = kwargs.get("entity_types", bio_entity_types)
    relation_types = kwargs.get("relation_types", bio_relation_types)
    validation_schema = kwargs.get("schema_validation", curriculum_validation_schema)

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
            if SCHEMA_EXTRACTOR_AVAILABLE:
                print(colored("│   │   ├── [*] Setting up SchemaLLMPathExtractor", "cyan"))
                print(colored("│   │   │   └── [?] This extractor uses a predefined schema to validate and extract relationships, ensuring logical consistency.", "yellow"))
                strict = kwargs.get("schema_strict", True)
                
                try:
                    extractors.append(
                        SchemaLLMPathExtractor(
                            llm=llm,
                            possible_entities=entity_types,
                            possible_relations=relation_types,
                            kg_validation_schema=validation_schema,
                            strict=strict,
                            num_workers=num_workers,
                            max_triplets_per_chunk=max_paths
                        )
                    )
                    print(colored("│   │   │   └── [+] SchemaLLMPathExtractor initialized", "green"))
                except Exception as e:
                    print(colored(f"│   │   │   └── [!] Error initializing SchemaLLMPathExtractor: {e}", "red"))
                    print(colored("│   │   │       └── [?] Falling back to DynamicLLMPathExtractor", "yellow"))
                    extractors.append(
                        DynamicLLMPathExtractor(
                            llm=llm,
                            max_triplets_per_chunk=max_paths,
                            num_workers=num_workers,
                            allowed_entity_types=entity_types,
                            allowed_relation_types=relation_types,
                        )
                    )
            else:
                print(colored("│   │   ├── [!] SchemaLLMPathExtractor not available", "yellow"))
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
    """Create a biographical knowledge graph and store it in the selected database."""
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
        graph_store = initialize_graph_store()
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
                embed_kg_nodes=False  # Disable embedding to reduce compatibility issues
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
        
        print(colored(f"│   └── [+] Biographical knowledge graph created and stored in {DB_TYPE} successfully!", "green"))
        
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
    parser = argparse.ArgumentParser(description="Create a biographical knowledge graph using LlamaIndex")
    parser.add_argument("--db", choices=["neo4j", "nebula"], default="neo4j", help="Database type to use")
    parser.add_argument("--mode", choices=["local", "remote"], default="remote", help="Database mode (local or remote)")
    parser.add_argument("--llm", choices=["openai", "azure", "auto"], default="auto", help="LLM provider to use")
    
    # Neo4j options
    parser.add_argument("--neo4j-uri", help="Neo4j URI (for remote: neo4j+s://example.databases.neo4j.io, for local: bolt://localhost:7687)")
    parser.add_argument("--neo4j-username", help="Neo4j username")
    parser.add_argument("--neo4j-password", help="Neo4j password")
    parser.add_argument("--neo4j-port", help="Neo4j local port (for local mode)")
    parser.add_argument("--neo4j-docker-name", help="Docker container name for Neo4j (for local mode)")
    
    # NebulaGraph options
    parser.add_argument("--nebula-host", help="NebulaGraph host")
    parser.add_argument("--nebula-port", type=int, help="NebulaGraph port")
    parser.add_argument("--nebula-space", help="NebulaGraph space name")
    parser.add_argument("--nebula-username", help="NebulaGraph username")
    parser.add_argument("--nebula-password", help="NebulaGraph password")
    parser.add_argument("--nebula-docker-name", help="Docker container name for NebulaGraph (for local mode)")
    
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
    parser.add_argument("--schema-strict", action="store_true", help="Use strict schema validation for SchemaLLMPathExtractor")
    
    args = parser.parse_args()
    print(colored("│   │   └── [+] Command line arguments parsed successfully", "green"))
    
    # Update global variables based on command line arguments
    global DB_TYPE, DB_MODE
    global NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_LOCAL_PORT, NEO4J_DOCKER_NAME
    global NEBULA_SPACE, NEBULA_HOST, NEBULA_PORT, NEBULA_USERNAME, NEBULA_PASSWORD, NEBULA_DOCKER_NAME
    
    DB_TYPE = args.db
    DB_MODE = args.mode
    
    # Neo4j settings
    if args.neo4j_uri:
        NEO4J_URI = args.neo4j_uri
    if args.neo4j_username:
        NEO4J_USERNAME = args.neo4j_username
    if args.neo4j_password:
        NEO4J_PASSWORD = args.neo4j_password
    if args.neo4j_port:
        NEO4J_LOCAL_PORT = args.neo4j_port
    if args.neo4j_docker_name:
        NEO4J_DOCKER_NAME = args.neo4j_docker_name
    
    # NebulaGraph settings
    if args.nebula_space:
        NEBULA_SPACE = args.nebula_space
    if args.nebula_host:
        NEBULA_HOST = args.nebula_host
    if args.nebula_port:
        NEBULA_PORT = args.nebula_port
    if args.nebula_username:
        NEBULA_USERNAME = args.nebula_username
    if args.nebula_password:
        NEBULA_PASSWORD = args.nebula_password
    if args.nebula_docker_name:
        NEBULA_DOCKER_NAME = args.nebula_docker_name
    
    # Prepare kwargs for other components
    print(colored("│   ├── [*] Preparing workflow parameters", "cyan"))
    kwargs = {
        "file": args.file,
        "dir": args.dir,
        "num_workers": args.num_workers,
        "max_paths": args.max_paths,
        "schema_strict": args.schema_strict,
    }
    print(colored("│   │   └── [+] Workflow parameters prepared", "green"))
    
    # Create the biographical knowledge graph
    print(colored("│   └── [*] Executing knowledge graph creation workflow", "cyan"))
    result = create_biographical_kg(args.llm, args.extractors, **kwargs)
    
    if result:
        print(colored("└── [+] Workflow completed successfully", "green"))
        
        # Display browser link for local Neo4j Browser
        if DB_MODE == "local" and DB_TYPE == "neo4j":
            print(colored("├── [*] Access your graph", "cyan"))
            print(colored(f"│   └── [?] Open Neo4j Browser at http://localhost:7474", "yellow"))
            print(colored(f"│       └── [?] Login with: {NEO4J_USERNAME} / {NEO4J_PASSWORD}", "yellow"))
        
        # Display information for NebulaGraph
        if DB_TYPE == "nebula":
            print(colored("├── [*] Access your NebulaGraph", "cyan"))
            if DB_MODE == "local":
                print(colored(f"│   └── [?] Your graph is stored in NebulaGraph space: {NEBULA_SPACE}", "yellow"))
                print(colored(f"│       └── [?] You can access it using NebulaGraph Console or Studio", "yellow"))
            else:
                print(colored(f"│   └── [?] Your graph is stored in remote NebulaGraph space: {NEBULA_SPACE}", "yellow"))
                print(colored(f"│       └── [?] Access using your NebulaGraph management tools", "yellow"))
    else:
        print(colored("└── [!] Workflow completed with errors", "red"))

if __name__ == "__main__":
    main()