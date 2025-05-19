"""
GraphRAG ReAct Agent Workflow with Graph Curation and Persistence
----------------------------------------------------------------
This module combines the GraphRAG implementation with a ReAct agent workflow,
with special focus on graph curation capabilities that actually persist changes to Neo4j.
"""

import os
import re
import json
import asyncio
from typing import Any, List, Dict, Optional, Union, Tuple, Set
import numpy as np
from collections import defaultdict

# LlamaIndex imports
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.tools import FunctionTool, ToolOutput, ToolSelection, BaseTool
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import ActionReasoningStep, ObservationReasoningStep
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context, Workflow, StartEvent, StopEvent, Event, step
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.graph_stores.types import EntityNode, Relation, KG_NODES_KEY, KG_RELATIONS_KEY

# Import our GraphRAG implementation
from graph_rag import (
    GraphRAGExtractor,
    GraphRAGStore,
    GraphRAGQueryEngine,
    parse_triplets,
    KG_TRIPLET_EXTRACT_TMPL
)


# Define workflow events
class PrepEvent(Event):
    pass


class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


# Extended GraphRAGStore with persistence methods for curation
class CuratingGraphRAGStore(GraphRAGStore):
    """
    Extended GraphRAGStore with additional methods for graph curation
    that persist changes to Neo4j.
    """

    def delete_entity(self, entity_name: str) -> bool:
        """
        Delete an entity and all its relationships from the Neo4j database.

        Args:
            entity_name: Name of the entity to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete all relationships where this entity is either source or target
            # This is a Cypher query to delete relationships
            delete_rels_query = """
            MATCH (n {name: $entity_name})-[r]-()
            DELETE r
            """
            self.query(delete_rels_query, {"entity_name": entity_name})

            # Delete the entity node
            delete_node_query = """
            MATCH (n {name: $entity_name})
            DELETE n
            """
            self.query(delete_node_query, {"entity_name": entity_name})

            return True
        except Exception as e:
            print(f"Error deleting entity {entity_name}: {str(e)}")
            return False

    def add_entity(self, entity: EntityNode) -> bool:
        """
        Add a new entity to the Neo4j database.

        Args:
            entity: The EntityNode to add

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert entity properties to a format suitable for Neo4j
            properties = entity.properties.copy()

            # Add entity name as a property
            properties["name"] = entity.name

            # Create Cypher query to create the node with its label and properties
            create_query = """
            CREATE (n:`{label}` $props)
            RETURN n
            """.format(label=entity.label)

            # Execute the query
            self.query(create_query, {"props": properties})

            return True
        except Exception as e:
            print(f"Error adding entity {entity.name}: {str(e)}")
            return False

    def add_relationship(self, relation: Relation) -> bool:
        """
        Add a new relationship to the Neo4j database.

        Args:
            relation: The Relation to add

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create Cypher query to create the relationship
            create_rel_query = """
            MATCH (source {name: $source_id}), (target {name: $target_id})
            CREATE (source)-[r:`{label}` $props]->(target)
            RETURN r
            """.format(label=relation.label)

            # Convert relation properties to a format suitable for Neo4j
            properties = relation.properties.copy()

            # Execute the query
            self.query(
                create_rel_query,
                {
                    "source_id": relation.source_id,
                    "target_id": relation.target_id,
                    "props": properties
                }
            )

            return True
        except Exception as e:
            print(f"Error adding relationship between {relation.source_id} and {relation.target_id}: {str(e)}")
            return False

    def update_entity(self, old_name: str, entity: EntityNode) -> bool:
        """
        Update an existing entity in the Neo4j database.

        Args:
            old_name: The current name of the entity to update
            entity: The updated EntityNode

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert entity properties to a format suitable for Neo4j
            properties = entity.properties.copy()

            # Update the entity
            update_query = """
            MATCH (n {name: $old_name})
            SET n = $props, n.name = $new_name, n:`{label}`
            """.format(label=entity.label)

            # Execute the query
            self.query(
                update_query,
                {
                    "old_name": old_name,
                    "props": properties,
                    "new_name": entity.name
                }
            )

            return True
        except Exception as e:
            print(f"Error updating entity {old_name} to {entity.name}: {str(e)}")
            return False

    def merge_entities_with_persistence(self, entity1: EntityNode, entity2: EntityNode, merged_entity: EntityNode) -> bool:
        """
        Merge two entities in the Neo4j database, maintaining relationships.

        Args:
            entity1: First entity to merge
            entity2: Second entity to merge
            merged_entity: The resulting merged entity

        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Create the new merged entity
            self.add_entity(merged_entity)

            # 2. Redirect all relationships where entity1 or entity2 was the source
            redirect_source_rels_query = """
            MATCH (old)-[r]->(target)
            WHERE old.name = $old_name1 OR old.name = $old_name2
            MATCH (new {name: $new_name})
            CREATE (new)-[r2:${label} $props]->(target)
            DELETE r
            """ # .replace("${label}", r.label) # This was a bug, label should be dynamic per relationship type

            # 3. Redirect all relationships where entity1 or entity2 was the target
            redirect_target_rels_query = """
            MATCH (source)-[r]->(old)
            WHERE old.name = $old_name1 OR old.name = $old_name2
            MATCH (new {name: $new_name})
            CREATE (source)-[r2:${label} $props]->(new)
            DELETE r
            """ # .replace("${label}", r.label) # This was a bug, label should be dynamic per relationship type

            # Get all existing relationship types and properties connected to old nodes
            # This needs to be more robust, getting specific relationships and their types
            # The original code iterated `self.get_triplets()` which is inefficient here.
            # Better: query relationships of old nodes directly.

            # Query for outgoing relationships from entity1 or entity2
            outgoing_rels_query = """
            MATCH (old)-[r]->(target)
            WHERE old.name = $old_name1 OR old.name = $old_name2
            RETURN old.name AS source_name, type(r) AS rel_type, properties(r) AS rel_props, target.name AS target_name
            """
            results_outgoing = self.query(outgoing_rels_query, {"old_name1": entity1.name, "old_name2": entity2.name})

            for rel_data in results_outgoing:
                dynamic_redirect_source_query = redirect_source_rels_query.replace("${label}", rel_data['rel_type'])
                self.query(
                    dynamic_redirect_source_query,
                    {
                        "old_name1": entity1.name, # Keep these to ensure correct "old" node matching
                        "old_name2": entity2.name,
                        "new_name": merged_entity.name,
                        "props": rel_data['rel_props']
                        # "source_name_to_match": rel_data['source_name'] # Could be used to be more specific
                    }
                )

            # Query for incoming relationships to entity1 or entity2
            incoming_rels_query = """
            MATCH (source)-[r]->(old)
            WHERE old.name = $old_name1 OR old.name = $old_name2
            RETURN source.name AS source_name, type(r) AS rel_type, properties(r) AS rel_props, old.name AS target_name
            """
            results_incoming = self.query(incoming_rels_query, {"old_name1": entity1.name, "old_name2": entity2.name})

            for rel_data in results_incoming:
                dynamic_redirect_target_query = redirect_target_rels_query.replace("${label}", rel_data['rel_type'])
                self.query(
                    dynamic_redirect_target_query,
                    {
                        "old_name1": entity1.name, # Keep these
                        "old_name2": entity2.name,
                        "new_name": merged_entity.name,
                        "props": rel_data['rel_props']
                        # "target_name_to_match": rel_data['target_name'] # Could be used to be more specific
                    }
                )

            # 4. Delete the original entities
            self.delete_entity(entity1.name)
            self.delete_entity(entity2.name)

            return True
        except Exception as e:
            print(f"Error merging entities {entity1.name} and {entity2.name}: {str(e)}")
            return False


# Utility functions for graph curation

def calculate_embedding_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings."""
    if emb1 is None or emb2 is None:
        return 0.0

    # Ensure embeddings are numpy arrays
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_duplicate_entities(
    graph_store: CuratingGraphRAGStore,
    embedding_model: BaseEmbedding,
    similarity_threshold: float = 0.85
) -> List[Tuple[EntityNode, EntityNode, float]]:
    """
    Find potential duplicate entities in the graph based on name similarity and embeddings.

    Returns:
        List of tuples (entity1, entity2, similarity_score)
    """
    # Get all entities from the graph
    # This might be inefficient for large graphs; consider querying only nodes.
    # For now, using get_all_nodes if available, or fallback to triplets.
    try:
        all_raw_nodes = graph_store.get_all_nodes() # Assumes this method exists and returns raw node data
        entities_dict = {}
        for node_dict in all_raw_nodes:
            # Adapt this based on the actual structure returned by get_all_nodes
            # This is a guess at the structure
            node_name = node_dict.get('name')
            if node_name:
                 # We need to construct EntityNode objects. This requires labels and properties.
                 # This part needs access to how EntityNode is constructed from raw graph data.
                 # For simplicity, let's assume we can get EntityNode objects directly or
                 # the GraphRAGStore provides a method to get EntityNode objects.
                 # The original code relied on triplets, let's stick to that for consistency
                 # if get_all_nodes() is not directly providing full EntityNode objects.
                 pass # Fallback to triplets if complex node reconstruction is needed
    except AttributeError: # If get_all_nodes is not defined
        pass


    triplets = graph_store.get_triplets()
    entities = {} # Use a dict to store unique entities by name
    for src_node, _, dst_node in triplets:
        if src_node.name not in entities:
            entities[src_node.name] = src_node
        if dst_node.name not in entities:
            entities[dst_node.name] = dst_node

    entity_list = list(entities.values())

    if len(entity_list) < 2:
        return []

    # Prepare texts for embedding (name + description if available)
    entity_texts_to_embed = []
    valid_entities_for_embedding = []
    for e in entity_list:
        # Ensure embedding is not already present and is a valid float array
        # The original code set e.embedding later. Let's assume we always re-embed for comparison.
        desc = e.properties.get('entity_description', '') if e.properties else ''
        entity_texts_to_embed.append(f"{e.name}: {desc}")
        valid_entities_for_embedding.append(e)

    if not valid_entities_for_embedding:
        return []

    embeddings = embedding_model.get_text_embedding_batch(entity_texts_to_embed)

    # Store embeddings with entities (or use them directly)
    # Temporary storage of embedding on the entity object for this function's scope
    for i, entity in enumerate(valid_entities_for_embedding):
        entity.embedding = embeddings[i] # Assuming entity objects are mutable for this

    potential_duplicates = []
    # Compare entities (now that they all have fresh embeddings for this run)
    for i in range(len(valid_entities_for_embedding)):
        for j in range(i + 1, len(valid_entities_for_embedding)):
            entity_i = valid_entities_for_embedding[i]
            entity_j = valid_entities_for_embedding[j]

            if entity_i.name == entity_j.name: # Should not happen if entities dict works
                continue

            # Ensure embeddings are available for comparison
            if not hasattr(entity_i, 'embedding') or not hasattr(entity_j, 'embedding'):
                continue

            sim_score = calculate_embedding_similarity(
                entity_i.embedding,
                entity_j.embedding
            )

            if sim_score > similarity_threshold:
                potential_duplicates.append(
                    (entity_i, entity_j, sim_score)
                )
    return potential_duplicates


def verify_entity_duplication(
    entity1: EntityNode,
    entity2: EntityNode,
    llm: LLM
) -> bool:
    """
    Use LLM to verify if two entities are actually duplicates.

    Returns:
        True if entities should be merged, False otherwise
    """
    entity1_desc = entity1.properties.get('entity_description', 'No description') if entity1.properties else 'No description'
    entity2_desc = entity2.properties.get('entity_description', 'No description') if entity2.properties else 'No description'

    prompt = f"""
    I need to determine if these two entities refer to the same real-world entity:

    Entity 1 Name: {entity1.name}
    Entity 1 Label: {entity1.label}
    Entity 1 Description: {entity1_desc}

    Entity 2 Name: {entity2.name}
    Entity 2 Label: {entity2.label}
    Entity 2 Description: {entity2_desc}

    Considering their names, labels, and descriptions, are they referring to the exact same real-world concept or object?
    Respond with only "YES" if they are the same entity and should be merged, or "NO" if they are different entities.
    """

    response = llm.complete(prompt)
    return response.text.strip().upper() == "YES"


def merge_entities(
    graph_store: CuratingGraphRAGStore,
    entity1: EntityNode,
    entity2: EntityNode,
    llm: LLM
) -> Optional[EntityNode]:
    """
    Merge two entities by:
    1. Choosing the better name
    2. Combining descriptions
    3. Updating all relationships to point to the merged entity

    Returns:
        The merged entity, or None if merge fails
    """
    entity1_desc = entity1.properties.get('entity_description', 'No description') if entity1.properties else 'No description'
    entity2_desc = entity2.properties.get('entity_description', 'No description') if entity2.properties else 'No description'

    prompt = f"""
    I need to merge these two entities that refer to the same real-world entity:

    Entity 1:
    Name: {entity1.name}
    Label: {entity1.label}
    Description: {entity1_desc}

    Entity 2:
    Name: {entity2.name}
    Label: {entity2.label}
    Description: {entity2_desc}

    Please provide the following for the new, merged entity:
    1. The best name to use (the most accurate, complete, or canonical name).
    2. The most appropriate entity label/type (choose one from the existing labels or suggest a better one if applicable).
    3. A comprehensive, merged description that combines the key information from both, avoiding redundancy.

    Format your answer EXACTLY as follows, with each item on a new line:
    NAME: [chosen best name]
    LABEL: [chosen best label]
    DESCRIPTION: [merged comprehensive description]
    """

    response = llm.complete(prompt)
    response_text = response.text

    name_match = re.search(r'NAME: (.*)', response_text, re.IGNORECASE)
    label_match = re.search(r'LABEL: (.*)', response_text, re.IGNORECASE)
    desc_match = re.search(r'DESCRIPTION: (.*)', response_text, re.IGNORECASE | re.DOTALL) # DOTALL for multiline desc

    merged_name, merged_label, merged_desc = None, None, None

    if name_match:
        merged_name = name_match.group(1).strip()
    if label_match:
        merged_label = label_match.group(1).strip()
    if desc_match:
        merged_desc = desc_match.group(1).strip()

    if not all([merged_name, merged_label, merged_desc]):
        print(f"Warning: LLM output parsing for entity merge details failed for '{entity1.name}' and '{entity2.name}'. Using fallback logic.")
        print(f"LLM Response:\n{response_text}")
        # Fallback logic (simpler merge)
        merged_name = entity1.name if len(entity1.name) >= len(entity2.name) else entity2.name # Prefer longer name
        merged_label = entity1.label or entity2.label # Prefer first non-empty label
        merged_desc = f"{entity1_desc}. {entity2_desc}".strip()
        if merged_desc == ".": merged_desc = "Merged entity."
    else:
        print(f"LLM suggested merge: Name='{merged_name}', Label='{merged_label}'")


    # Create merged properties
    merged_properties = (entity1.properties or {}).copy()
    merged_properties.update(entity2.properties or {})
    merged_properties['entity_description'] = merged_desc
    if 'name' in merged_properties: # Name is not a regular property for EntityNode
        del merged_properties['name']


    # Use one of the embeddings, or re-embed the new merged description if desired
    # For simplicity, take embedding of entity1 if available
    merged_embedding = entity1.embedding if hasattr(entity1, 'embedding') and entity1.embedding is not None else None
    if merged_embedding is None and hasattr(entity2, 'embedding'):
        merged_embedding = entity2.embedding

    merged_entity = EntityNode(
        name=merged_name,
        label=merged_label,
        properties=merged_properties,
        embedding=merged_embedding # Pass existing embedding or None
    )

    success = graph_store.merge_entities_with_persistence(entity1, entity2, merged_entity)

    if not success:
        print(f"Warning: Failed to persist entity merge of {entity1.name} and {entity2.name} into {merged_entity.name}")
        return None

    return merged_entity


def discover_missing_relationships(
    graph_store: CuratingGraphRAGStore,
    llm: LLM,
    max_relationships_to_find: int = 5
) -> List[Tuple[EntityNode, EntityNode, str, str]]: # (src, dst, rel_type, rel_desc)
    """
    Discover potential relationships between existing entities that aren't currently connected.

    Returns:
        List of tuples (source_entity, target_entity, relation_type, description)
    """
    triplets = graph_store.get_triplets()
    entities_dict = {}
    existing_relationships_set = set() # Store as (src_name, dst_name)

    for src_node, rel, dst_node in triplets:
        if src_node.name not in entities_dict:
            entities_dict[src_node.name] = src_node
        if dst_node.name not in entities_dict:
            entities_dict[dst_node.name] = dst_node
        existing_relationships_set.add((src_node.name, dst_node.name))
        existing_relationships_set.add((dst_node.name, src_node.name)) # Consider undirected for check

    entity_list = list(entities_dict.values())

    if len(entity_list) < 2:
        return []

    entity_pairs_to_check = []
    # Create candidate pairs that don't have direct relationships
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)): # Avoid self-loops and duplicate pairs (j > i)
            src_entity = entity_list[i]
            dst_entity = entity_list[j]
            if (src_entity.name, dst_entity.name) not in existing_relationships_set and \
               (dst_entity.name, src_entity.name) not in existing_relationships_set:
                entity_pairs_to_check.append((src_entity, dst_entity))

    # If there are too many potential pairs, sample a subset to avoid excessive LLM calls
    if len(entity_pairs_to_check) > 20: # Configurable limit
        import random
        random.shuffle(entity_pairs_to_check)
        entity_pairs_to_check = entity_pairs_to_check[:20]

    discovered_relationships = []
    if not entity_pairs_to_check:
        return []

    # Prepare a context of all entities for the LLM
    entity_context_parts = []
    for idx, entity in enumerate(entity_list):
        desc = entity.properties.get('entity_description', 'No description') if entity.properties else 'No description'
        entity_context_parts.append(f"{idx+1}. Name: {entity.name}, Label: {entity.label}, Description: {desc}")
    full_entity_context = "\n".join(entity_context_parts)


    for src_entity, dst_entity in entity_pairs_to_check:
        src_desc = src_entity.properties.get('entity_description', 'No description') if src_entity.properties else 'No description'
        dst_desc = dst_entity.properties.get('entity_description', 'No description') if dst_entity.properties else 'No description'

        prompt = f"""
        I have a knowledge graph with the following entities:
        --- ENTITY CONTEXT START ---
        {full_entity_context}
        --- ENTITY CONTEXT END ---

        I am considering if a meaningful relationship exists between these two specific entities:

        Source Entity:
        Name: {src_entity.name}
        Label: {src_entity.label}
        Description: {src_desc}

        Target Entity:
        Name: {dst_entity.name}
        Label: {dst_entity.label}
        Description: {dst_desc}

        Based on their descriptions and general knowledge, if these two entities are related in a significant way, please describe the relationship.
        Format your answer EXACTLY as follows:
        RELATION_TYPE: [A concise, verb-based label for the relationship, e.g., "is part of", "competes with", "founded by"]
        DESCRIPTION: [A brief explanation of why this relationship exists or what it means]

        If you believe there is no clear, direct, or significant relationship, or if the relationship is too generic or trivial, respond ONLY with:
        RELATION_TYPE: none
        """

        response = llm.complete(prompt)
        response_text = response.text.strip()

        rel_type_match = re.search(r'RELATION_TYPE: (.*)', response_text, re.IGNORECASE)
        if rel_type_match:
            relation_type = rel_type_match.group(1).strip()
            if relation_type.lower() != "none" and relation_type != "":
                desc_match = re.search(r'DESCRIPTION: (.*)', response_text, re.IGNORECASE | re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else "Discovered by LLM."

                discovered_relationships.append(
                    (src_entity, dst_entity, relation_type, description)
                )
                if len(discovered_relationships) >= max_relationships_to_find:
                    break
    return discovered_relationships


def add_discovered_relationships(
    graph_store: CuratingGraphRAGStore,
    discovered_relationships: List[Tuple[EntityNode, EntityNode, str, str]]
) -> int:
    """
    Add discovered relationships to the graph.

    Returns:
        Number of relationships successfully added
    """
    successful_adds = 0
    for src_entity, dst_entity, relation_type, description in discovered_relationships:
        relation_props = {
            "relationship_description": description,
            "discovered_by_curation": True,
            "confidence_score": 0.80 # Example default confidence
        }
        relation = Relation(
            source_id=src_entity.name,
            target_id=dst_entity.name,
            label=relation_type.upper().replace(" ", "_"), # Standardize label format
            properties=relation_props
        )
        if graph_store.add_relationship(relation):
            print(f"Added new relationship: ({src_entity.name})-[{relation.label}]->({dst_entity.name})")
            successful_adds += 1
        else:
            print(f"Failed to add relationship: ({src_entity.name})-[{relation.label}]->({dst_entity.name})")
    return successful_adds


# Tool functions for GraphRAG operations
def add_document_to_graph(
    document_text: str,
    graph_store: Any = None, # CuratingGraphRAGStore
    index: Any = None, # PropertyGraphIndex
    # kg_extractor is part of index
    # llm is part of index
    title: str = "",
) -> str:
    """
    Add a document to the knowledge graph.

    Args:
        document_text: The text content of the document
        title: Optional title for the document

    Returns:
        A message indicating success or failure
    """
    if not graph_store or not index:
        return "Error: Graph store or index not initialized for 'add_document_to_graph' tool."
    try:
        document = Document(text=document_text, metadata={"title": title or "Untitled Document"})
        # Node parsing happens within index.insert() if not pre-parsed
        # Assuming index.insert handles node creation and triplet extraction
        index.insert(document) # This should trigger KG extraction and storage

        # GraphRAGStore might have its own community building logic triggered differently
        # or PropertyGraphIndex handles it.
        # If explicit community building is needed after inserts:
        if hasattr(graph_store, 'build_communities') and callable(graph_store.build_communities):
            graph_store.build_communities()

        return f"Successfully processed document titled '{document.metadata['title']}' for graph addition. Check graph_stats for updates."
    except Exception as e:
        return f"Failed to add document to graph: {str(e)}"


def query_graph(
    query: str,
    query_engine: Any = None, # GraphRAGQueryEngine
) -> str:
    """
    Query the knowledge graph using GraphRAG.

    Args:
        query: The query string

    Returns:
        The response from GraphRAG
    """
    if not query_engine:
        return "Error: Query engine not initialized for 'query_graph' tool."
    try:
        response = query_engine.query(query)
        return f"GraphRAG response: {response.response}" # Access .response for string output
    except Exception as e:
        return f"Failed to query graph: {str(e)}"


def graph_stats(
    graph_store: Any = None, # CuratingGraphRAGStore
) -> str:
    """
    Get statistics about the current knowledge graph.

    Returns:
        Information about the graph structure
    """
    if not graph_store:
        return "Error: Graph store not initialized for 'graph_stats' tool."
    try:
        triplets = graph_store.get_triplets()
        num_triplets = len(triplets)

        entities = set()
        relation_types = defaultdict(int)
        for src, rel, dst in triplets:
            entities.add(src.name)
            entities.add(dst.name)
            relation_types[rel.label] += 1

        num_entities = len(entities)
        num_relation_types = len(relation_types)

        stats_str = f"Graph Statistics:\n"
        stats_str += f"- Total Unique Entities: {num_entities}\n"
        stats_str += f"- Total Relationships (Triplets): {num_triplets}\n"
        stats_str += f"- Number of Unique Relationship Types: {num_relation_types}\n"
        if relation_types:
            stats_str += "  Relationship Type Counts:\n"
            for rel_type, count in sorted(relation_types.items(), key=lambda item: item[1], reverse=True):
                stats_str += f"    - {rel_type}: {count}\n"

        # Community info if available
        if hasattr(graph_store, 'get_community_summaries') and callable(graph_store.get_community_summaries):
            community_summaries = graph_store.get_community_summaries()
            stats_str += f"- Number of Communities: {len(community_summaries)}\n"
        else:
            stats_str += "- Community information not available through this method.\n"

        return stats_str
    except Exception as e:
        return f"Failed to get graph statistics: {str(e)}"


def curate_graph(
    graph_store: Any = None, # CuratingGraphRAGStore
    # index: Any = None, # PropertyGraphIndex - not directly used here but for context
    llm: Any = None, # LLM
    embedding_model: Any = None, # BaseEmbedding
) -> str:
    """
    Curate the knowledge graph by finding duplicate entities and discovering new relationships.
    All changes are persisted to the Neo4j database.

    Returns:
        A report of the curation actions taken
    """
    if not all([graph_store, llm, embedding_model]):
        return "Error: graph_store, llm, or embedding_model not initialized for 'curate_graph' tool."
    try:
        report_parts = ["Graph Curation Report:\n"]
        action_taken = False

        # 1. Find and merge duplicate entities
        print("Starting duplicate entity detection...")
        potential_duplicates = find_duplicate_entities(
            graph_store=graph_store,
            embedding_model=embedding_model,
            similarity_threshold=0.85 # Make this configurable
        )
        print(f"Found {len(potential_duplicates)} potential duplicate pairs.")

        merged_count = 0
        if potential_duplicates:
            verified_duplicates_info = []
            for entity1, entity2, sim_score in potential_duplicates:
                print(f"Verifying duplication between '{entity1.name}' and '{entity2.name}' (Similarity: {sim_score:.2f})")
                if verify_entity_duplication(entity1, entity2, llm):
                    print(f"Confirmed duplicate: '{entity1.name}' and '{entity2.name}'. Attempting merge...")
                    merged_entity = merge_entities(graph_store, entity1, entity2, llm)
                    if merged_entity:
                        verified_duplicates_info.append(f"- Merged '{entity1.name}' and '{entity2.name}' into '{merged_entity.name}'.")
                        merged_count += 1
                        action_taken = True
                    else:
                        verified_duplicates_info.append(f"- Failed to merge '{entity1.name}' and '{entity2.name}'.")
                else:
                    print(f"LLM indicated '{entity1.name}' and '{entity2.name}' are not duplicates.")


            if merged_count > 0:
                report_parts.append(f"\nMerged {merged_count} duplicate entity pair(s):")
                report_parts.extend(verified_duplicates_info)
            elif potential_duplicates:
                 report_parts.append("\nNo duplicate entities confirmed for merging by LLM among potential candidates.")
            else: # Should not happen if potential_duplicates was non-empty
                 report_parts.append("\nNo potential duplicate entities found to process.")
        else:
            report_parts.append("\nNo potential duplicate entities found based on similarity.")


        # 2. Discover missing relationships
        print("\nStarting missing relationship discovery...")
        new_relationships_tuples = discover_missing_relationships(
            graph_store=graph_store,
            llm=llm,
            max_relationships_to_find=3 # Make configurable
        )
        print(f"Discovered {len(new_relationships_tuples)} potential new relationships.")

        if new_relationships_tuples:
            added_rel_count = add_discovered_relationships(graph_store, new_relationships_tuples)
            if added_rel_count > 0:
                report_parts.append(f"\nAdded {added_rel_count} new relationship(s):")
                for src, dst, rel_type, desc in new_relationships_tuples[:added_rel_count]: # Only report added ones
                    report_parts.append(f"- ({src.name})-[{rel_type.upper().replace(' ','_')}]->({dst.name}): {desc[:100]}...")
                action_taken = True
            else:
                report_parts.append("\nNo new relationships were successfully added from discovered candidates.")
        else:
            report_parts.append("\nNo new relationships discovered.")

        if action_taken:
            # Rebuild communities if graph structure changed
            if hasattr(graph_store, 'build_communities') and callable(graph_store.build_communities):
                print("\nRebuilding communities after curation...")
                graph_store.build_communities()
                report_parts.append("\nGraph communities have been updated.")
        else:
            report_parts.append("\nNo curation actions were performed on the graph.")

        return "\n".join(report_parts)
    except Exception as e:
        print(f"Error during graph curation: {str(e)}") # Log it too
        return f"Failed to curate graph: {str(e)}"


def fix_entity(
    entity_name: str,
    new_name: str = None,
    new_type: str = None, # label
    new_description: str = None,
    graph_store: Any = None, # CuratingGraphRAGStore
) -> str:
    """
    Fix an entity in the knowledge graph by updating its name, type, or description.

    Args:
        entity_name: The current name of the entity to fix
        new_name: New name for the entity (optional)
        new_type: New type/label for the entity (optional)
        new_description: New description for the entity (optional)

    Returns:
        A message indicating success or failure
    """
    if not graph_store:
        return "Error: Graph store not initialized for 'fix_entity' tool."
    if not any([new_name, new_type, new_description]):
        return "No changes specified for the entity. Provide new_name, new_type, or new_description."

    try:
        # Need a way to get a single entity. GraphRAGStore might not have this.
        # Workaround: Iterate triplets or query Neo4j directly if store allows.
        # This is inefficient but works with current CuratingGraphRAGStore.
        target_entity_node: Optional[EntityNode] = None
        # Attempt to fetch the node directly if possible
        # This assumes `get_node_by_name` or similar might exist or be added to GraphRAGStore
        if hasattr(graph_store, 'get_node_by_name'):
            raw_node_data = graph_store.get_node_by_name(entity_name) # Hypothetical method
            if raw_node_data:
                 # Reconstruct EntityNode. This is complex without knowing raw_node_data structure.
                 # For now, stick to triplet iteration if direct fetch is not clean.
                 pass

        # Fallback to triplet iteration:
        if not target_entity_node:
            triplets = graph_store.get_triplets()
            for src, _, dst in triplets:
                if src.name == entity_name:
                    target_entity_node = src
                    break
                if dst.name == entity_name:
                    target_entity_node = dst
                    break

        if target_entity_node is None:
            return f"Entity '{entity_name}' not found in the graph."

        # Create updated entity properties
        updated_properties = (target_entity_node.properties or {}).copy()
        if new_description is not None: # Allow empty string to clear description
            updated_properties['entity_description'] = new_description

        final_name = new_name if new_name is not None else target_entity_node.name
        final_label = new_type if new_type is not None else target_entity_node.label

        # The EntityNode itself doesn't store 'name' in properties typically
        if 'name' in updated_properties:
            del updated_properties['name']


        updated_entity = EntityNode(
            name=final_name,
            label=final_label,
            properties=updated_properties,
            embedding=target_entity_node.embedding # Preserve embedding
        )

        if graph_store.update_entity(entity_name, updated_entity): # old_name, new_entity_node
            if hasattr(graph_store, 'build_communities') and callable(graph_store.build_communities) and new_name and new_name != entity_name:
                graph_store.build_communities() # If name changes, communities might be affected
            return f"Successfully updated entity. Old name: '{entity_name}', New name: '{final_name}', New label: '{final_label}'."
        else:
            return f"Failed to update entity '{entity_name}' in the graph store."

    except Exception as e:
        return f"Error fixing entity '{entity_name}': {str(e)}"


def fix_relationship(
    source_entity: str,
    target_entity: str,
    old_relation_type: str, # Need old type to uniquely identify relationship if multiple exist
    new_relation_type: str = None,
    new_description: str = None,
    graph_store: Any = None, # CuratingGraphRAGStore
) -> str:
    """
    Fix a relationship in the knowledge graph by updating its type or description.

    Args:
        source_entity: The source entity name
        target_entity: The target entity name
        old_relation_type: The current type/label of the relationship to fix
        new_relation_type: New relationship type (optional)
        new_description: New relationship description (optional)

    Returns:
        A message indicating success or failure
    """
    if not graph_store:
        return "Error: Graph store not initialized for 'fix_relationship' tool."
    if not new_relation_type and new_description is None: # Allow empty string for new_description
        return "No changes specified. Provide new_relation_type or new_description."

    try:
        # Find the specific relationship. This is tricky with current GraphRAGStore.
        # We need to delete the old one and create a new one.
        # This assumes there's at most one relationship of old_relation_type between source and target.
        # If multiple, this logic needs to be more specific or tools more granular.

        # Step 1: Get properties of the old relationship (if we want to preserve some)
        # This requires a Cypher query.
        get_rel_props_query = """
        MATCH (s {name: $source_name})-[r:`{old_rel_type}`]->(t {name: $target_name})
        RETURN properties(r) as props
        LIMIT 1
        """.format(old_rel_type=old_relation_type) # Be careful with Cypher injection if old_relation_type is user input
                                                # Better to parameterize if possible, but labels are hard.

        result = graph_store.query(get_rel_props_query, {"source_name": source_entity, "target_name": target_entity})
        if not result or not result[0] or 'props' not in result[0]:
            return f"Relationship '{old_relation_type}' between '{source_entity}' and '{target_entity}' not found or has no properties."
        
        old_properties = result[0]['props']
        updated_properties = old_properties.copy()

        if new_description is not None:
            updated_properties['relationship_description'] = new_description
        if 'discovered_by_curation' in updated_properties: # Mark as manually fixed
            updated_properties['fixed_by_curation'] = True


        # Step 2: Delete the old relationship
        delete_rel_query = """
        MATCH (s {name: $source_name})-[r:`{old_rel_type}`]->(t {name: $target_name})
        DELETE r
        """.format(old_rel_type=old_relation_type)
        graph_store.query(delete_rel_query, {"source_name": source_entity, "target_name": target_entity})


        # Step 3: Add the new/updated relationship
        final_relation_type = new_relation_type if new_relation_type else old_relation_type
        final_relation_type = final_relation_type.upper().replace(" ","_")


        new_rel = Relation(
            source_id=source_entity,
            target_id=target_entity,
            label=final_relation_type,
            properties=updated_properties
        )

        if graph_store.add_relationship(new_rel):
            # Communities might not be strongly affected by rel property/type changes unless type changes significantly
            # if hasattr(graph_store, 'build_communities') and callable(graph_store.build_communities):
            #     graph_store.build_communities()
            return (f"Successfully updated relationship: "
                    f"({source_entity})-[{final_relation_type}]->({target_entity}).")
        else:
            # Attempt to rollback or indicate failure state
            return (f"Failed to add new/updated relationship after deleting old one. "
                    f"Graph might be in an inconsistent state for: "
                    f"({source_entity})-...->({target_entity}).")

    except Exception as e:
        return f"Error fixing relationship between '{source_entity}' and '{target_entity}': {str(e)}"


class GraphRAGReActAgent(Workflow):
    """
    ReAct agent that can interact with and maintain a GraphRAG knowledge base.
    """

    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        neo4j_config: Dict[str, str] = None,
        embedding_model: BaseEmbedding | None = None,
        # GraphRAG specific components if not fully encapsulated by neo4j_config init
        graph_store_instance: Optional[CuratingGraphRAGStore] = None,
        index_instance: Optional[PropertyGraphIndex] = None,
        query_engine_instance: Optional[GraphRAGQueryEngine] = None,
        kg_extractor_instance: Optional[GraphRAGExtractor] = None,
        extra_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.llm = llm or OpenAI(model="gpt-4") # Consider making model configurable
        self.embedding_model = embedding_model or OpenAIEmbedding()

        self.graph_store = graph_store_instance
        self.index = index_instance
        self.query_engine = query_engine_instance
        self.kg_extractor = kg_extractor_instance

        if neo4j_config and not self.graph_store: # Only initialize if not pre-supplied
            self._initialize_graphrag_components(neo4j_config)

        if not all([self.graph_store, self.index, self.query_engine, self.kg_extractor]):
            raise ValueError("GraphRAG components (store, index, query engine, extractor) are not properly initialized or provided.")

        graphrag_tools = self._create_graphrag_tools()
        self.tools = (tools or []) + graphrag_tools

        default_context = (
            "You are an AI assistant with access to a knowledge graph. "
            "You can add documents to the graph, query it for information, "
            "get statistics about it, and curate it by finding/merging duplicate entities, "
            "discovering new relationships, and fixing existing entities and relationships. "
            "When fixing relationships, you need to know the source entity, target entity, AND the current relationship type."
        )
        self.formatter = ReActChatFormatter.from_defaults(
            context=extra_context or default_context
        )
        self.output_parser = ReActOutputParser()

    def _initialize_graphrag_components(self, neo4j_config: Dict[str, str]) -> None:
        """Initialize GraphRAG components if not provided externally."""
        print("Initializing GraphRAG components with Neo4j config...")
        self.graph_store = CuratingGraphRAGStore(
            llm=self.llm, # Assuming GraphRAGStore needs an LLM for some internal ops
            username=neo4j_config.get("username", "neo4j"),
            password=neo4j_config.get("password", ""),
            url=neo4j_config.get("url", "bolt://localhost:7687"),
            database=neo4j_config.get("database", "neo4j") # Add database if used
        )

        self.kg_extractor = GraphRAGExtractor( # Using the imported one
            llm=self.llm,
            extract_prompt=KG_TRIPLET_EXTRACT_TMPL, # Imported constant
            max_paths_per_chunk=10, # Configurable
            parse_fn=parse_triplets # Imported function
        )

        # PropertyGraphIndex is the primary interface for building the graph from documents
        self.index = PropertyGraphIndex(
            nodes=[], # Start with an empty index; nodes are added via documents
            property_graph_store=self.graph_store,
            kg_extractors=[self.kg_extractor], # List of extractors
            # llm=self.llm, # index might also take an LLM for its ops
            # embedding_model=self.embedding_model # and embedding model
        )

        self.query_engine = GraphRAGQueryEngine( # Using the imported one
            graph_store=self.graph_store,
            index=self.index, # Query engine needs the index for some operations
            llm=self.llm,
            # embedding_model=self.embedding_model,
            similarity_top_k=5, # Configurable
            # other GraphRAGQueryEngine specific params
        )
        print("GraphRAG components initialized.")


    def _create_graphrag_tools(self) -> List[BaseTool]:
        """Create tools for interacting with GraphRAG."""
        # These tools now directly use self.graph_store, self.index etc.
        # The lambda functions pass these instance variables.
        tools_list = [
            FunctionTool.from_defaults(
                fn=lambda document_text, title="": add_document_to_graph(
                    document_text,
                    graph_store=self.graph_store,
                    index=self.index,
                    title=title
                ),
                name="add_document_to_graph",
                description="Adds a new document (text content) to the knowledge graph. Specify 'document_text' and optionally 'title'."
            ),
            FunctionTool.from_defaults(
                fn=lambda query: query_graph(query, query_engine=self.query_engine),
                name="query_graph",
                description="Queries the knowledge graph using natural language. Specify 'query'."
            ),
            FunctionTool.from_defaults(
                fn=lambda: graph_stats(graph_store=self.graph_store),
                name="graph_stats",
                description="Gets statistics about the current state of the knowledge graph (entities, relationships, etc.)."
            ),
            FunctionTool.from_defaults(
                fn=lambda: curate_graph(
                    graph_store=self.graph_store,
                    llm=self.llm,
                    embedding_model=self.embedding_model
                ),
                name="curate_graph",
                description="Performs automated curation on the graph: finds and merges duplicate entities, and discovers potential new relationships."
            ),
            FunctionTool.from_defaults(
                fn=lambda entity_name, new_name=None, new_type=None, new_description=None: fix_entity(
                    entity_name, new_name, new_type, new_description,
                    graph_store=self.graph_store
                ),
                name="fix_entity",
                description="Updates an existing entity in the graph. Specify 'entity_name' (current name) and at least one of: 'new_name', 'new_type' (label), or 'new_description'."
            ),
            FunctionTool.from_defaults(
                fn=lambda source_entity, target_entity, old_relation_type, new_relation_type=None, new_description=None: fix_relationship(
                    source_entity, target_entity, old_relation_type, new_relation_type, new_description,
                    graph_store=self.graph_store
                ),
                name="fix_relationship",
                description="Updates an existing relationship. Specify 'source_entity', 'target_entity', 'old_relation_type' (current type), and at least one of: 'new_relation_type' or 'new_description'."
            ),
        ]
        return tools_list

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        await ctx.set("sources", [])
        memory = await ctx.get("memory", default=None)
        if not memory:
            # Ensure memory is compatible with self.llm if it uses it internally
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        user_input_content = ev.input # Assuming ev.input is the raw string
        if isinstance(user_input_content, dict) and "input" in user_input_content: # Adapt if input is structured
            user_input_content = user_input_content["input"]

        user_msg = ChatMessage(role="user", content=user_input_content)
        memory.put(user_msg)
        await ctx.set("current_reasoning", [])
        await ctx.set("memory", memory)
        return PrepEvent()

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: PrepEvent
    ) -> InputEvent:
        memory = await ctx.get("memory")
        chat_history = memory.get()
        current_reasoning = await ctx.get("current_reasoning", default=[])
        llm_input_messages = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input_messages)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> Union[ToolCallEvent, StopEvent, PrepEvent]: # Added PrepEvent for error case
        llm_messages_for_api = ev.input
        current_reasoning = await ctx.get("current_reasoning", default=[])
        memory = await ctx.get("memory")

        # Streaming response from LLM
        full_response_content = ""
        try:
            response_gen = await self.llm.astream_chat(llm_messages_for_api)
            async for response_chunk in response_gen:
                delta = response_chunk.delta or ""
                full_response_content += delta
                ctx.write_event_to_stream(StreamEvent(delta=delta))

            # Construct a final ChatMessage from the aggregated content for parsing
            final_llm_message = ChatMessage(role="assistant", content=full_response_content)

            # Parse the full response
            reasoning_step = self.output_parser.parse(final_llm_message) # Pass ChatMessage
            current_reasoning.append(reasoning_step)
            await ctx.set("current_reasoning", current_reasoning)


            if reasoning_step.is_done:
                memory.put(ChatMessage(role="assistant", content=reasoning_step.response))
                await ctx.set("memory", memory)
                sources = await ctx.get("sources", default=[])
                return StopEvent(
                    result={
                        "response": reasoning_step.response,
                        "sources": [s.content for s in sources if hasattr(s, 'content')], # Extract content
                        "reasoning": [str(r) for r in current_reasoning], # Stringify reasoning steps
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                # LlamaIndex ToolSelection expects kwargs as a dict.
                # ReActOutputParser typically puts arguments in reasoning_step.action_input (which should be a dict)
                tool_kwargs = reasoning_step.action_input
                if not isinstance(tool_kwargs, dict):
                    # Attempt to parse if it's a JSON string, or handle error
                    try:
                        tool_kwargs = json.loads(str(tool_kwargs)) if isinstance(tool_kwargs, str) else {}
                    except json.JSONDecodeError:
                        print(f"Warning: Tool input '{tool_kwargs}' is not a valid JSON dict. Attempting to use as is.")
                        # This might still fail if the tool expects a dict.
                        # Fallback or error needed if tool_kwargs isn't a dict.
                        # For now, let it proceed and potentially fail at tool call.
                        pass


                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="tool_" + reasoning_step.action, # Make a unique ID
                            tool_name=reasoning_step.action,
                            tool_kwargs=tool_kwargs if isinstance(tool_kwargs, dict) else {},
                        )
                    ]
                )
            else: # Should not happen if is_done or ActionReasoningStep
                print(f"Unexpected reasoning step type: {type(reasoning_step)}")
                current_reasoning.append(ObservationReasoningStep(observation="Unexpected reasoning step from LLM output."))
                await ctx.set("current_reasoning", current_reasoning)
                return PrepEvent() # Re-prompt

        except Exception as e:
            print(f"Error processing LLM output or parsing reasoning: {e}")
            # Log the full_response_content that caused the error for debugging
            print(f"LLM response content that caused error: {full_response_content}")
            current_reasoning.append(
                ObservationReasoningStep(
                    observation=f"Error processing LLM response or parsing reasoning: {e}. Please try rephrasing or check tool inputs."
                )
            )
            await ctx.set("current_reasoning", current_reasoning)
            return PrepEvent() # Re-prompt the LLM

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> PrepEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.name: tool for tool in self.tools} # Use .metadata.name
        current_reasoning = await ctx.get("current_reasoning", default=[])
        sources = await ctx.get("sources", default=[])

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                obs_text = f"Tool '{tool_call.tool_name}' not found. Available tools: {list(tools_by_name.keys())}"
                print(obs_text)
                current_reasoning.append(ObservationReasoningStep(observation=obs_text))
                continue
            try:
                # Ensure tool_kwargs is a dictionary
                tool_kwargs = tool_call.tool_kwargs if isinstance(tool_call.tool_kwargs, dict) else {}
                print(f"Calling tool: {tool.metadata.name} with args: {tool_kwargs}")
                tool_output: ToolOutput = tool(**tool_kwargs)
                sources.append(tool_output) # ToolOutput object
                obs_text = str(tool_output.content)
                print(f"Tool Output ({tool.metadata.name}): {obs_text[:200]}...") # Log snippet
                current_reasoning.append(ObservationReasoningStep(observation=obs_text))
            except Exception as e:
                obs_text = f"Error calling tool {tool.metadata.name} with args {tool_call.tool_kwargs}: {e}"
                print(obs_text)
                current_reasoning.append(ObservationReasoningStep(observation=obs_text))

        await ctx.set("sources", sources)
        await ctx.set("current_reasoning", current_reasoning)
        return PrepEvent()


async def main():
    """Example usage of GraphRAG ReAct Agent with curation capabilities."""
    # Ensure OPENAI_API_KEY is set in your environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    if not os.getenv("NEO4J_PASSWORD"): # Add check for Neo4j password
        print("Error: NEO4J_PASSWORD environment variable not set (or other Neo4j credentials).")
        return

    neo4j_config = {
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD"),
        "url": os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j")
    }

    # Initialize LLM and Embedding Model (can be configured further)
    llm = OpenAI(model="gpt-3.5-turbo-0125") # Cheaper and faster for testing, GPT-4 for complex tasks
    embedding_model = OpenAIEmbedding()

    # Initialize the agent
    # The agent will internally initialize GraphRAG components using neo4j_config
    agent_workflow = GraphRAGReActAgent(
        llm=llm,
        embedding_model=embedding_model,
        neo4j_config=neo4j_config,
        timeout=300, # Increased timeout for potentially long curation steps
        verbose=True # LlamaIndex verbosity
    )

    # Agent expects a Context object for run. Let's reuse one.
    workflow_ctx = Context(agent_workflow) # workflow_ctx, not ctx to avoid conflict

    async def run_agent_command(command: str):
        print(f"\n>>> USER: {command}")
        response_event = await agent_workflow.run(input=command, ctx=workflow_ctx)
        # The result is in response_event.output if it's a StopEvent
        if response_event and hasattr(response_event, 'output') and response_event.output:
            final_response = response_event.output.get("response", "No response content.")
            print(f"\n<<< AGENT: {final_response}")
            # sources = response_event.output.get("sources", [])
            # if sources:
            #     print(f"--- Sources ({len(sources)}) ---")
            #     for i, src in enumerate(sources):
            #         print(f"Source {i+1}: {str(src)[:200]}...") # Print content of ToolOutput
            # reasoning = response_event.output.get("reasoning", [])
            # if reasoning:
            #     print(f"--- Reasoning ({len(reasoning)}) ---")
            #     for i, step_str in enumerate(reasoning):
            #         print(f"Step {i+1}: {step_str}")
            return final_response
        else:
            print("\n<<< AGENT: No final response or error in workflow execution.")
            return "Workflow did not complete as expected."


    # Example interaction sequence
    initial_documents = [
        "Chevron is an American multinational energy corporation. It is one of the successor companies of Standard Oil and is headquartered in San Ramon, California. Chevron engages in every aspect of the oil and natural gas industries, including hydrocarbon exploration and production; refining, marketing and transport; chemicals manufacturing and sales; and power generation.",
        "Chevron Corp, formerly Standard Oil of California, is a major player in the global energy sector, involved in oil, gas, and geothermal energy projects. Its headquarters are in California, USA.",
        "ExxonMobil is one of the largest publicly traded international energy and petrochemical companies. It was formed on November 30, 1999, by the merger of Exxon and Mobil. ExxonMobil often competes with Chevron in various global markets.",
        "The global energy sector is dominated by several key companies, including Chevron, BP (British Petroleum), Shell, and ExxonMobil. These companies focus on the exploration, production, refining, and distribution of oil and gas products worldwide."
    ]

    for i, doc_text in enumerate(initial_documents):
        await run_agent_command(f"Add this document to the graph, titled 'Doc {i+1}': '{doc_text}'")

    await run_agent_command("What are the current statistics of our knowledge graph?")

    await run_agent_command("Please curate the graph. Find duplicate entities and discover potential new relationships.")

    await run_agent_command("What are the graph statistics now after curation?")

    # Example of fixing an entity. Let's assume 'Chevron Corp' was merged into 'Chevron'
    # or we want to ensure a specific description for 'Chevron'.
    # The LLM might need to query first to know exact names if they changed.
    await run_agent_command("Fix the entity named 'Chevron' by updating its description to clearly state it was a successor to Standard Oil and its primary focus is oil and gas.")

    # Example of fixing a relationship.
    # First, let's see what relationships 'Chevron' has.
    await run_agent_command("What direct relationships does 'Chevron' have in the graph?")
    # Based on the output, the user (or LLM in a more advanced setup) would formulate the fix command.
    # Assuming a relationship like (Chevron)-[COMPETES_WITH]->(ExxonMobil) exists:
    await run_agent_command("Fix the relationship between 'Chevron' and 'ExxonMobil' of type 'COMPETES_WITH' by updating its description to 'Chevron and ExxonMobil are major global competitors in oil and gas exploration, production, and sales.'")

    await run_agent_command("What can you tell me about Chevron and its relationships to other entities in the graph?")

    await run_agent_command("Show me the graph statistics one last time.")


if __name__ == "__main__":
    # This is a placeholder for the graph_rag module
    # In a real scenario, this module would contain the definitions for
    # GraphRAGExtractor, GraphRAGStore, GraphRAGQueryEngine, parse_triplets, KG_TRIPLET_EXTRACT_TMPL
    # For this script to be runnable as-is, these need to be stubbed or defined.

    class MockLLM: # Basic mock for non-OpenAI dependent parts of GraphRAGStore
        def complete(self, prompt): return type('obj', (object,), {'text': ''})()

    class GraphRAGStore(Neo4jPropertyGraphStore): # Extend the real one
        def __init__(self, llm, **kwargs): # Add llm if your base class needs it
            self.llm_internal = llm # or pass to super if it accepts
            super().__init__(**kwargs)
            # self.client, self._driver, self._database are initialized by Neo4jPropertyGraphStore

        # If GraphRAGStore had these methods, they would be here.
        # For now, Neo4jPropertyGraphStore already has get_triplets.
        # build_communities and get_community_summaries are specific to GraphRAGStore's logic.
        def build_communities(self):
            print("Mock GraphRAGStore: build_communities() called.")
            # In a real implementation, this would run community detection algorithms (e.g., Louvain)
            # and store community IDs and possibly summaries on nodes or as separate community nodes.
            # For now, we can simulate creating some community data if needed by other parts.
            # Example: Add a 'community_id' property to all nodes.
            try:
                # A simple way to assign a dummy community ID
                self.query("MATCH (n) SET n.community_id = 0")
                # Or more complex:
                # self.query("""
                # CALL gds.graph.project.estimate('*', '*') YIELD nodeCount
                # CALL gds.louvain.write({
                #   nodeProjection: '*',
                #   relationshipProjection: '*',
                #   writeProperty: 'community_id'
                # })
                # YIELD communityCount, modularity
                # RETURN communityCount, modularity;
                # """) # This requires GDS plugin
                print("Mock: Assigned dummy community IDs or simulated Louvain.")
            except Exception as e:
                print(f"Mock build_communities error (GDS might not be available): {e}")


        def get_community_summaries(self) -> List[str]:
            print("Mock GraphRAGStore: get_community_summaries() called.")
            # This would fetch nodes per community, summarize them (e.g., using LLM or keyword extraction).
            # For mock, return a fixed list or query distinct community_ids.
            try:
                results = self.query("MATCH (n) WHERE n.community_id IS NOT NULL RETURN DISTINCT n.community_id AS id ORDER BY id")
                if results:
                    return [f"Summary for mock community {res['id']}" for res in results]
                return ["Mock Community Summary 1", "Mock Community Summary 2"]
            except Exception as e:
                print(f"Mock get_community_summaries error: {e}")
                return ["Mock Community Summary (error fetching)"]


    class GraphRAGExtractor:
        def __init__(self, llm, extract_prompt, max_paths_per_chunk, parse_fn):
            self.llm = llm
            self.extract_prompt = extract_prompt
            self.max_paths_per_chunk = max_paths_per_chunk
            self.parse_fn = parse_fn
            print("Mock GraphRAGExtractor initialized.")

        def extract_triplets(self, text_chunk: str) -> List[Tuple[str, str, str]]:
            # This is a simplified mock. A real one would call the LLM.
            print(f"Mock GraphRAGExtractor: extracting triplets from chunk: '{text_chunk[:50]}...'")
            # Simulate LLM call and parsing
            # Example: "Alice (Person) works_at (Relationship) Google (Organization)."
            # Would be parsed into ("Alice", "works_at", "Google")
            # Based on KG_TRIPLET_EXTRACT_TMPL, it expects specific output.
            # Let's return a fixed dummy triplet for any input.
            if "chevron" in text_chunk.lower() and "exxonmobil" in text_chunk.lower():
                return [("Chevron", "competes with", "ExxonMobil")]
            if "chevron" in text_chunk.lower():
                return [("Chevron", "is a", "Energy Corporation")]
            return [("Subject", "related to", "Object")]

        # PropertyGraphIndex calls `extract_ সেরাKG_NodesAndRelationships(node)` which in turn would call this
        # if this extractor is used. The LlamaIndex `KnowledgeGraphKGExtractor` is the one normally used by `PropertyGraphIndex`.
        # To use this custom `GraphRAGExtractor`, the `PropertyGraphIndex` would need to be adapted or
        # this class needs to conform to the `BaseKGExtractor` interface expected by `PropertyGraphIndex`.
        # For simplicity, let's assume PropertyGraphIndex can use it or it's a conceptual placeholder.
        # The important part is that index.insert() triggers something like this.


    def parse_triplets(input_str: str) -> List[Tuple[str, str, str]]:
        # Mock parsing logic
        print(f"Mock parse_triplets called with: '{input_str[:50]}...'")
        triplets = []
        # Example: input_str might be "('Alice', 'works at', 'Google')\n('Google', 'located in', 'USA')"
        try:
            # A very naive parser for "(subj, rel, obj)" format
            for line in input_str.strip().split('\n'):
                if line.startswith("(") and line.endswith(")"):
                    parts = line[1:-1].split(',')
                    if len(parts) == 3:
                        subj = parts[0].strip().strip("'\"")
                        rel = parts[1].strip().strip("'\"")
                        obj = parts[2].strip().strip("'\"")
                        triplets.append((subj, rel, obj))
        except Exception:
            pass # Ignore parsing errors in mock
        if not triplets and input_str: # Fallback if parse fails but input exists
            return [("ParsedSubject", "parsedRelationship", "ParsedObject")]
        return triplets

    KG_TRIPLET_EXTRACT_TMPL = """
    Some text is provided below. Given the text, extract up to {max_knowledge_triplets}
    knowledge triplets in the form of (subject, predicate, object).
    Avoid stopwords.
    ---------------------
    Text: {text}
    ---------------------
    Triplets:
    """

    class GraphRAGQueryEngine:
        def __init__(self, graph_store, index, llm, similarity_top_k):
            self.graph_store = graph_store
            self.index = index
            self.llm = llm
            self.similarity_top_k = similarity_top_k
            print("Mock GraphRAGQueryEngine initialized.")

        def query(self, query_str: str):
            print(f"Mock GraphRAGQueryEngine: received query '{query_str}'")
            # Mock response generation. A real engine would:
            # 1. Understand the query (e.g., entity lookup, path finding, summarization).
            # 2. Retrieve relevant subgraphs/paths from Neo4j (using self.graph_store).
            # 3. Use self.index for vector search if needed.
            # 4. Synthesize an answer using self.llm.
            mock_response_text = f"This is a mock response from GraphRAGQueryEngine for your query: '{query_str}'. "
            if "chevron" in query_str.lower():
                mock_response_text += "Chevron is a major energy company. It competes with ExxonMobil."
            else:
                mock_response_text += "The graph contains information that might be relevant."

            class MockResponse:
                def __init__(self, text):
                    self.response = text # LlamaIndex responses often have a .response attribute

            return MockResponse(mock_response_text)

    # Apply mocks by replacing the global names if `graph_rag` module is not actually present
    # This is a hack for making the script runnable without the actual graph_rag.py
    # In a real project structure, you'd import from your `graph_rag` module.
    import sys
    sys.modules['graph_rag'] = type('module', (object,), {
        'GraphRAGExtractor': GraphRAGExtractor,
        'GraphRAGStore': GraphRAGStore,
        'GraphRAGQueryEngine': GraphRAGQueryEngine,
        'parse_triplets': parse_triplets,
        'KG_TRIPLET_EXTRACT_TMPL': KG_TRIPLET_EXTRACT_TMPL,
    })()

    # Now that mocks are in place, re-import to ensure the main script uses them if graph_rag.py is absent
    from graph_rag import (
        GraphRAGExtractor,
        GraphRAGStore,
        GraphRAGQueryEngine,
        parse_triplets,
        KG_TRIPLET_EXTRACT_TMPL
    )
    # The CuratingGraphRAGStore will inherit from the mocked GraphRAGStore.


    asyncio.run(main())