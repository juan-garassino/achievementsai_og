import os
import streamlit as st
from PIL import Image

# ----- Sidebar Configuration from Version 1.0 -----
ROOT = os.path.dirname(os.path.abspath(__file__))
CV_PATH = os.path.join(ROOT, "assets", "JuanGarassinoCV-2024.pdf")
IMAGE_PATH = os.path.join(ROOT, "assets", "robot-assistant.png")

st.set_page_config(
    page_title="Chat with Juan's Curriculum 🤓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar content
st.sidebar.title("About This App")
st.sidebar.image(Image.open(IMAGE_PATH), use_container_width=True)

st.sidebar.markdown(
    """
I designed this application as a chatbot that interacts with a Retrieval-Augmented Generation (RAG) model.  
The RAG model is fine-tuned with information about me and my professional curriculum, 
enabling personalized and accurate responses about my background and expertise.
"""
)

st.sidebar.markdown("### 1. Key Features")
st.sidebar.markdown(
    """
- **A. Chatbot**: Engage with a sophisticated agent that leverages both vector search and graph queries.
- **B. RAG + Graph**: Combines Pinecone vector store with Neo4j property graph for deep insights.
- **C. Function Tools**: Send emails directly from the chat using a secure Gmail function.
- **D. Cloud-Ready**: Built for deployment on Streamlit Cloud or any containerized environment.
"""
)

st.sidebar.markdown("### 2. Biography")
st.sidebar.markdown(
    """
Juan Garassino is a machine learning and deep learning professional with four years of experience in data science and MLOps engineering. 
He transitioned from architecture to tech, bringing a unique creative problem-solving perspective to AI. 
His expertise spans computer vision, generative AI, automated ML pipelines, and smart cities applications.
"""
)

st.sidebar.markdown("### 3. Skills and Expertise")
st.sidebar.markdown(
    """
- **Machine Learning & Deep Learning**: Classification, prediction, and generative models.  
- **Computer Vision**: Object detection, segmentation, and 3D generative design.  
- **Generative AI**: GANs, diffusion, and LLM-driven content creation.  
- **Automated Pipelines**: Docker, Kubernetes, and cloud orchestration.  
- **Graph & Vector Search**: Neo4j and Pinecone integration for rich querying.
"""
)

st.sidebar.markdown("### 4. Projects & Links")
st.sidebar.markdown(
    """
- **[deepTechno](https://github.com/juan-garassino/deepTechno)**: Techno music synthesis with transformers.  
- **[deepSculpt](https://github.com/juan-garassino/deepSculpt)**: 3D GANs for architectural spaces.  
- **[MiniNetworks](https://github.com/juan-garassino)**: Educational small neural networks.
"""
)

st.sidebar.markdown("### 5. Testimonial")
st.sidebar.markdown(
    """
> "In a short time, Mr. Garassino has shown remarkable adaptability and leadership, teaching around the globe and driving impact in B2B and B2C programs."  
> **– Titiana Benassi, People Manager @ Le Wagon**
"""
)

st.sidebar.markdown("### 6. Download Resume")
with st.sidebar:
    with open(CV_PATH, "rb") as f:
        st.download_button(
            label="Download CV (PDF)",
            data=f,
            file_name="JuanGarassinoCV-2024.pdf",
            mime="application/pdf",
        )

st.sidebar.markdown("### 7. Contact Me")
cols = st.sidebar.columns(5)
for icon, link in zip(["👨‍💻", "💼", "📷", "📧", "📱"],
                      ["https://github.com/juan-garassino",
                       "https://www.linkedin.com/in/juan-garassino/",
                       "https://www.instagram.com/artista.artificial/",
                       "mailto:juan.garassino@gmail.com",
                       None]):
    if link:
        cols.pop(0).markdown(f"[{icon}]({link})")
    else:
        cols.pop(0).markdown(icon)
st.sidebar.markdown("<div style='text-align:center;'>+49 0152 24024860</div>", unsafe_allow_html=True)

# CSS for buttons
st.markdown(
    """
<style>
.stButton > button { width:100%; padding:0.5rem; margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ----- Environment & Secrets Configuration -----

# Load credentials from environment variables or Streamlit secrets
AZ_API_KEY = os.getenv("AZURE_OPENAI_APIKEY") or st.secrets["AZURE_OPENAI"]["AZURE_OPENAI_APIKEY"]
AZ_LLM_EP = os.getenv("AZURE_OPENAI_ENDPOINT_LLM") or st.secrets["AZURE_OPENAI"]["AZURE_OPENAI_ENDPOINT_LLM"]
AZ_EMB_EP = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING") or st.secrets["AZURE_OPENAI"]["AZURE_OPENAI_ENDPOINT_EMBEDDING"]
AZ_API_VER = os.getenv("AZURE_OPENAI_API_VERSION") or st.secrets["AZURE_OPENAI"]["AZURE_OPENAI_API_VERSION"]
PINECONE_KEY = os.getenv("PINECONE_API_KEY") or st.secrets["API_KEYS"]["PINECONE_API_KEY"]
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or st.secrets["NEO4J"]["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or st.secrets["NEO4J"]["NEO4J_PASSWORD"]
NEO4J_URI = os.getenv("NEO4J_CONNECTION_URI") or st.secrets["NEO4J"]["NEO4J_CONNECTION_URI"]
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD") or st.secrets["GMAIL_APP_PASSWORD"]["GMAIL_APP_PASSWORD"]


# ----- Agent Initialization and Tooling -----
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent
import nest_asyncio
from email_tool import send_gmail as _send_owner_email

# Apply asyncio patch for Jupyter compatibility
nest_asyncio.apply()

# Initialize LLM and Embeddings
llm = AzureOpenAI(
    model="gpt-4o-mini",
    deployment_name="gpt-4o-mini",
    api_key=AZ_API_KEY,
    azure_endpoint=AZ_LLM_EP,
    api_version=AZ_API_VER,
)
embeddings = AzureOpenAIEmbedding(
    model="text-embedding-3-large",
    deployment_name="text-embedding-3-large",
    api_key=AZ_API_KEY,
    azure_endpoint=AZ_EMB_EP,
    api_version=AZ_API_VER,
)
Settings.llm = llm
Settings.embed_model = embeddings

# Neo4j Graph Index
neo4j_store = Neo4jPropertyGraphStore(
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    url=NEO4J_URI,
)
graph_index = PropertyGraphIndex.from_existing(
    property_graph_store=neo4j_store,
    llm=llm,
    embed_model=embeddings,
)


# Pinecone Vector Index
vector_store = PineconeVectorStore(
    index_name="achievementsai",
    namespace="achievementsai",
    api_key=PINECONE_KEY
)

vector_index = VectorStoreIndex.from_vector_store(vector_store)

# Function for sending email

def send_gmail(subject: str, message: str) -> str:
    """Send a message to Juan. The recipient is fixed (see email_tool.py)."""
    return _send_owner_email(subject, message, app_password=GMAIL_APP_PASSWORD)

# Create query-engine tools
vector_qe = vector_index.as_query_engine()
graph_qe = graph_index.as_query_engine()

tools = [
    QueryEngineTool(
        query_engine=vector_qe,
        metadata=ToolMetadata(
            name="biography_semantic_search",
            description="General background queries about Juan's skills and experiences.",
        ),
    ),
    QueryEngineTool(
        query_engine=graph_qe,
        metadata=ToolMetadata(
            name="biography_relationship_query",
            description="Relationship-focused queries: projects, collaborations, timeline.",
        ),
    ),
    FunctionTool.from_defaults(
        fn=send_gmail,
        name="send_gmail",
        description="Send an email to Juan (the recipient is always Juan; include the visitor's contact details in the message).",
    ),
]


# System prompt for the agent
SYSTEM_PROMPT = (
    "You are an assistant that helps users find information about Juan's biography.\n"
    "For relationship queries, use biography_relationship_query.\n"
    "For general queries, use biography_semantic_search.\n"
    "You can also send emails using the send_gmail tool when appropriate.\n"
    "Introduce your capabilities on first interaction and suggest using tools."
)

# One agent (and chat memory) per browser session: a cache_resource agent would
# share its ChatMemoryBuffer, i.e. one visitor's conversation, with every visitor.
def load_agent():
    return ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        system_prompt=SYSTEM_PROMPT,
        memory=ChatMemoryBuffer.from_defaults(token_limit=3900),
    )

if "agent" not in st.session_state:
    st.session_state.agent = load_agent()
agent = st.session_state.agent

# ----- Streamlit UI Main Content -----
st.title("Chat with Juan's Curriculum 🤓")
st.write(
    "Welcome to my AI-powered chatbot! I designed this app to answer questions about my professional background. Feel free to use the chat to interact with my curriculum. If you have further questions after the chat, let's organize a meeting - I'll be happy to answer the rest of them!"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": (
            "Hello! I'm your assistant for Juan's biography. "
            "I can search his background, explore relationships, and even send an email to him. "
            "How can I help you today?"
        )}
    ]

if "button_pressed" not in st.session_state:
    st.session_state.button_pressed = False
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# If there are no messages except the initial greeting, show suggested questions
if len(st.session_state.messages) <= 1 and not st.session_state.button_pressed:
    suggested_questions = [
        "Can you describe Juan Garassino's skills in machine learning?",
        "What can you share about Juan's early years?",
        "How might Juan's experience as a ML Instructor benefit our company?",
        "What was Juan's role while he lived in New Zealand?",
        "Would Juan be a strong fit for our Generative AI team?",
    ]

    for question in suggested_questions:
        if st.button(question):
            st.session_state.button_pressed = True
            st.session_state.last_question = question
            st.rerun()

# If a button was just pressed, process the query
if st.session_state.button_pressed:
    st.session_state.messages.append({"role": "user", "content": st.session_state.last_question})
    with st.chat_message("assistant"):
        response = agent.chat(st.session_state.last_question)
        st.write(response.response)
    st.session_state.messages.append({"role": "assistant", "content": response.response})
    st.session_state.button_pressed = False
    st.session_state.last_question = ""

# Handle user input
if prompt := st.chat_input("Ask me about Juan…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = agent.chat(prompt)
        st.write(response.response)
    st.session_state.messages.append({"role": "assistant", "content": response.response})
