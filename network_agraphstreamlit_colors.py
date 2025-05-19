import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import time

# Set page configuration
st.set_page_config(
    page_title="Knowledge Graph Visualization",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Container styling */
    .css-1d391kg, .css-1kyxreq, div[data-testid="stVerticalBlock"] {
        border: none !important;
        outline: none !important;
    }
    .stApp {
        background-color: white;
    }
    [data-testid="stDecoration"], [data-testid="stToolbar"] {
        display: none;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #333;
    }
    .network-graph {
        background-color: white;
        border-radius: 15px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        overflow: hidden;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    ::-webkit-scrollbar {
        display: none;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Knowledge Graph Visualization")

# Loading spinner
loading_placeholder = st.empty()
with loading_placeholder.container():
    with st.spinner("Loading Knowledge Graph..."):
        time.sleep(1)

# Create nodes and edges from your input data only
def create_knowledge_graph():
    nodes = [
        Node(id="Juan_Garassino", label="Juan Garassino", size=25, color="#FFC107"),
        Node(id="Argentina", label="Argentina", size=15, color="#8BC34A"),
        Node(id="Italian_heritage", label="Italian heritage", size=15, color="#4CAF50"),
        Node(id="Garassino_family", label="Garassino family", size=15, color="#FFC107"),
        Node(id="mother", label="Mother (Architect)", size=15, color="#FFC107"),
        Node(id="father", label="Father (Data Engineer)", size=15, color="#FFC107"),
    ]

    edges = [
        Edge(source="Juan_Garassino", target="Argentina", label="BORN_IN"),
        Edge(source="Juan_Garassino", target="Italian_heritage", label="HAS_HERITAGE"),
        Edge(source="Juan_Garassino", target="Garassino_family", label="HAS_FAMILY"),
        Edge(source="Juan_Garassino", target="mother", label="HAS_MOTHER"),
        Edge(source="Juan_Garassino", target="father", label="HAS_FATHER"),
    ]

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

nodes, edges, config = create_knowledge_graph()

loading_placeholder.empty()

# Render graph container
graph_container = st.container()
with graph_container:
    st.markdown('<div class="network-graph fade-in">', unsafe_allow_html=True)
    agraph(nodes=nodes, edges=edges, config=config)
    st.markdown('</div>', unsafe_allow_html=True)

# Explanation & legend
st.markdown("""
### About this Knowledge Graph

This knowledge graph displays relationships strictly based on the provided data.

**Node color legend:**
- <span style='color:#FF5733'>●</span> People & Family
- <span style='color:#33A1FF'>●</span> Cultural Identity
- <span style='color:#33FF57'>●</span> Location

Interact by dragging, zooming, panning, and clicking nodes.
""", unsafe_allow_html=True)

# Sidebar with controls
st.sidebar.title("Visualization Controls")
st.sidebar.info("This application displays a knowledge graph network using the streamlit-agraph library.")

physics_enabled = st.sidebar.checkbox("Enable Physics Simulation", value=True)
if not physics_enabled:
    config.physics = False
    st.experimental_rerun()

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
