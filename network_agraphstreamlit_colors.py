import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import time

# Set page configuration
st.set_page_config(
    page_title="Knowledge Graph Visualization",
    layout="wide"
)

# Custom CSS for rounded corners and removing gray lines
st.markdown("""
<style>
    /* Style the container to have rounded corners */
    .css-1d391kg, .css-1kyxreq, div[data-testid="stVerticalBlock"] {
        border: none !important;
        outline: none !important;
    }
    
    /* Remove all gray outlines/borders from the visualization */
    .stApp {
        background-color: white;
    }
    
    /* Custom graph container styling */
    [data-testid="stDecoration"], [data-testid="stToolbar"] {
        display: none;
    }
    
    /* Style headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #333;
    }
    
    /* Style the visualization container */
    .network-graph {
        background-color: white;
        border-radius: 15px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        overflow: hidden;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Hide scrollbars */
    ::-webkit-scrollbar {
        display: none;
    }
    
    /* Add animation for loading effect */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
</style>
""", unsafe_allow_html=True)

# Add a title
st.title("Knowledge Graph Visualization")

# Create a loading spinner placeholder
loading_placeholder = st.empty()

# Show loading spinner
with loading_placeholder.container():
    with st.spinner("Loading Knowledge Graph..."):
        # Simulate loading time (the actual graph creation might take some time)
        time.sleep(1)

# Function to create the knowledge graph
def create_knowledge_graph():
    # Create nodes
    nodes = []
    
    # People (red-orange color: #FF5733)
    nodes.append(Node(id="Juan_Garassino", 
                      label="Juan Garassino", 
                      size=25,
                      color="#FF5733"))
    
    nodes.append(Node(id="father", 
                      label="Father", 
                      size=15,
                      color="#FF5733"))
    
    nodes.append(Node(id="mother", 
                      label="Mother", 
                      size=15,
                      color="#FF5733"))
    
    nodes.append(Node(id="Garassino_family", 
                      label="Garassino Family", 
                      size=15,
                      color="#FF5733"))
    
    # Locations (green color: #33FF57)
    nodes.append(Node(id="Argentina", 
                      label="Argentina", 
                      size=15,
                      color="#33FF57"))
                      
    nodes.append(Node(id="Buenos_Aires", 
                      label="Buenos Aires", 
                      size=15,
                      color="#33FF57"))
                      
    nodes.append(Node(id="Berlin", 
                      label="Berlin", 
                      size=15,
                      color="#33FF57"))
    
    # Technology concepts (purple color: #A233FF)
    nodes.append(Node(id="machine_learning", 
                      label="Machine Learning", 
                      size=18,
                      color="#A233FF"))
                      
    nodes.append(Node(id="AI", 
                      label="AI", 
                      size=18,
                      color="#A233FF"))
                      
    nodes.append(Node(id="data_management", 
                      label="Data Management", 
                      size=15,
                      color="#A233FF"))
                      
    nodes.append(Node(id="technology_implementation", 
                      label="Technology Implementation", 
                      size=15,
                      color="#A233FF"))
    
    # Skills (gold color: #FFD700)
    nodes.append(Node(id="analytical_thinking", 
                      label="Analytical Thinking", 
                      size=15,
                      color="#FFD700"))
                      
    nodes.append(Node(id="communication_style", 
                      label="Communication Style", 
                      size=15,
                      color="#FFD700"))
                      
    nodes.append(Node(id="spatial_reasoning", 
                      label="Spatial Reasoning", 
                      size=15,
                      color="#FFD700"))
                      
    nodes.append(Node(id="creative_problem_solving", 
                      label="Creative Problem-Solving", 
                      size=15,
                      color="#FFD700"))
                      
    nodes.append(Node(id="quantitative_reasoning", 
                      label="Quantitative Reasoning", 
                      size=15,
                      color="#FFD700"))
    
    # Concepts (blue color: #33A1FF)
    nodes.append(Node(id="Italian_heritage", 
                      label="Italian Heritage", 
                      size=15,
                      color="#33A1FF"))
                      
    nodes.append(Node(id="AI_innovation", 
                      label="AI Innovation", 
                      size=18,
                      color="#33A1FF"))
                      
    nodes.append(Node(id="ethical_AI", 
                      label="Ethical AI Development", 
                      size=18,
                      color="#33A1FF"))
                      
    nodes.append(Node(id="cross_disciplinary", 
                      label="Cross-Disciplinary Collaboration", 
                      size=18,
                      color="#33A1FF"))
                      
    nodes.append(Node(id="democratizing_AI", 
                      label="Democratizing AI Knowledge", 
                      size=15,
                      color="#33A1FF"))
    
    # Create edges
    edges = []
    
    # Family connections
    edges.append(Edge(source="Juan_Garassino", target="father", label="child_of"))
    edges.append(Edge(source="Juan_Garassino", target="mother", label="child_of"))
    edges.append(Edge(source="Juan_Garassino", target="Garassino_family", label="member_of"))
    edges.append(Edge(source="Garassino_family", target="Italian_heritage", label="has_origin"))
    
    # Location connections
    edges.append(Edge(source="Juan_Garassino", target="Argentina", label="born_in"))
    edges.append(Edge(source="Juan_Garassino", target="Buenos_Aires", label="lived_in"))
    edges.append(Edge(source="Juan_Garassino", target="Berlin", label="lives_in"))
    edges.append(Edge(source="Buenos_Aires", target="Argentina", label="part_of"))
    
    # Professional connections
    edges.append(Edge(source="Juan_Garassino", target="machine_learning", label="expertise_in"))
    edges.append(Edge(source="Juan_Garassino", target="AI", label="works_with"))
    edges.append(Edge(source="Juan_Garassino", target="data_management", label="skill_in"))
    edges.append(Edge(source="Juan_Garassino", target="technology_implementation", label="implements"))
    
    # Skills connections
    edges.append(Edge(source="Juan_Garassino", target="analytical_thinking", label="has_skill"))
    edges.append(Edge(source="Juan_Garassino", target="communication_style", label="has_trait"))
    edges.append(Edge(source="Juan_Garassino", target="spatial_reasoning", label="has_ability"))
    edges.append(Edge(source="Juan_Garassino", target="creative_problem_solving", label="employs"))
    edges.append(Edge(source="Juan_Garassino", target="quantitative_reasoning", label="uses"))
    
    # Concept connections
    edges.append(Edge(source="Juan_Garassino", target="AI_innovation", label="contributes_to"))
    edges.append(Edge(source="Juan_Garassino", target="ethical_AI", label="advocates_for"))
    edges.append(Edge(source="Juan_Garassino", target="cross_disciplinary", label="engages_in"))
    edges.append(Edge(source="Juan_Garassino", target="democratizing_AI", label="supports"))
    
    # Domain connections
    edges.append(Edge(source="machine_learning", target="AI", label="subset_of"))
    edges.append(Edge(source="ethical_AI", target="AI", label="aspect_of"))
    edges.append(Edge(source="democratizing_AI", target="AI_innovation", label="goal_of"))
    edges.append(Edge(source="cross_disciplinary", target="AI_innovation", label="enables"))
    
    # Configure the graph
    config = Config(
        width=750,
        height=600,
        directed=True, 
        physics=True,
        hierarchical=False,
        node_color="#97c2fc",
        node_size=10,
        node_font_size=14,
        edge_width=2,
        edge_curved=0.1,
        node_spacing=200,
        solver="forceAtlas2Based",
        edge_color="#888888",
        edge_font_size=12,
        edge_font_color="#555555"
    )
    
    return nodes, edges, config

# Create the knowledge graph
nodes, edges, config = create_knowledge_graph()

# Clear the loading spinner
loading_placeholder.empty()

# Create a container for the graph with styling
graph_container = st.container()
with graph_container:
    st.markdown('<div class="network-graph fade-in">', unsafe_allow_html=True)
    # Render the graph
    return_value = agraph(nodes=nodes, 
                          edges=edges, 
                          config=config)
    st.markdown('</div>', unsafe_allow_html=True)

# Add explanation with color legend
st.markdown("""
### About this Knowledge Graph

This interactive knowledge graph shows the relationships between various entities related to Juan Garassino.

**Node color legend:**
- <span style='color:#FF5733'>●</span> People
- <span style='color:#33A1FF'>●</span> Concepts
- <span style='color:#33FF57'>●</span> Locations
- <span style='color:#A233FF'>●</span> Technology
- <span style='color:#FFD700'>●</span> Skills

You can interact with the graph by:
- Dragging nodes to reposition them
- Zooming in and out using the mouse wheel or trackpad
- Panning by dragging the background
- Clicking on nodes to see more information
""", unsafe_allow_html=True)

# Sidebar information
st.sidebar.title("Visualization Controls")
st.sidebar.info("This application displays a knowledge graph network using the streamlit-agraph library.")

# Optional: Add controls in the sidebar
physics_enabled = st.sidebar.checkbox("Enable Physics Simulation", value=True)
if not physics_enabled:
    config.physics = False
    st.experimental_rerun()

# Optional: Add node search functionality
st.sidebar.markdown("### Search Nodes")
search_term = st.sidebar.text_input("Enter node name to search")
if search_term:
    matching_nodes = [node.label for node in nodes if search_term.lower() in node.label.lower()]
    if matching_nodes:
        st.sidebar.write("Matching nodes:")
        for node in matching_nodes:
            st.sidebar.write(f"- {node}")
    else:
        st.sidebar.write("No matching nodes found.")