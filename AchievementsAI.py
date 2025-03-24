import os
import sys

# Check if running on Streamlit Cloud
if os.getenv("MY_APP_ENV") == "streamlit_cloud":
    print("Running on Streamlit Cloud")
    # Replace sqlite3 with pysqlite3
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
else:
    print("Not running on Streamlit Cloud")

import chromadb
import streamlit as st
from PIL import Image

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)

# from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic

from llama_index.core.node_parser import SentenceSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# from langchain_community.chat_message_histories import ChatMessageHistory
from llama_index.core.prompts import PromptTemplate

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "achievementsAI"
ROOT = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_PATH = os.path.join(ROOT, "data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-3-sonnet-20240229"  # "phi3"
LLM_TIMEOUT = 660.0
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
TOP_K = 10

# # Custom prompt template
# CUSTOM_PROMPT_TEMPLATE = (
#     "You are a helpful assistant designed to provide information about Juan Garassino's professional background. "
#     "Focus primarily on Juan's recent experiences in data science, machine learning, deep learning, and generative AI engineering. "
#     "Always refer to Juan in the third person. You are an assistant he developed, not Juan himself. "
#     "Respond positively and encouragingly, using clear and accessible language. "
#     "Keep answers concise and directly address the query. "
#     "If uncertain about any information, acknowledge that you don't know. "
#     "Your goal is to assist Juan in securing positions in data science, machine learning, deep learning, or generative AI engineering. "
#     "Emphasize Juan's skills and experiences in machine learning and generative AI as they are most relevant to his current career goals. "
#     "If mentioning Juan's architectural background, do so briefly and mainly to highlight transferable skills or unique perspectives it brings to his current field. "
    
#     "Do not mention Juan's background in architecture, architectural drawing or morphology. Do it only if you are asked to"
    
#     "Context: {context_str}\n"
#     "Human: {query_str}\n"
#     "Assistant: "
# )

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = (
    "You are a helpful assistant designed to provide information about Juan Garassino's professional background. "
    "Focus primarily on Juan's recent experiences in data science, machine learning, deep learning, and generative AI engineering. "
    "Always refer to Juan in the third person. You are an assistant he developed, not Juan himself. "
    "Respond positively and encouragingly, using clear and accessible language. "
    "Keep answers concise and directly address the query. "
    "If uncertain about any information, acknowledge that you don't know. "
    "Your goal is to assist Juan in securing positions in data science, machine learning, deep learning, or generative AI engineering. "
    "Emphasize Juan's skills and experiences in machine learning and generative AI as they are most relevant to his current career goals. "
    "Do not mention Juan's background in architecture, architectural drawing, or morphology unless specifically asked. "
    "Context: {context_str}\n"
    "Human: {query_str}\n"
    "Assistant: "
)

# Set up Chroma database and vector store
db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load documents
documents = SimpleDirectoryReader(DOCUMENTS_PATH).load_data()

# Configure settings
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

llm = Anthropic(
    model=LLM_MODEL,
    temperature=0.7,  # Adjust as needed
    api_key=ANTHROPIC_API_KEY,
    timeout=LLM_TIMEOUT,
    # streaming=True
)

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[
        SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    ],
)

# Create query engine with custom prompt
custom_prompt = PromptTemplate(CUSTOM_PROMPT_TEMPLATE)
query_engine = index.as_query_engine(
    similarity_top_k=TOP_K, text_qa_template=custom_prompt
)

# Sidebar content
st.sidebar.title("About This App")

st.sidebar.markdown(
    """
I designed this application as a chatbot that interacts with a Retrieval-Augmented Generation (RAG) model. 
The RAG model has been fine-tuned with information about me and my professional curriculum, allowing for personalized and accurate responses about my background and expertise.
"""
)

# Add an image to the sidebar

image_path = os.path.join(ROOT, "assets", "robot-assistant.png")

image = Image.open(image_path)  # Replace with the path to your image
st.sidebar.image(image, use_column_width=True)

st.sidebar.markdown("### 1. Key Features")
st.sidebar.markdown(
    """
- **A. Chatbot**: Engage with a sophisticated chatbot that provides information about me.
- **B. RAG Model**: Utilizing state-of-the-art Retrieval-Augmented Generation techniques for accurate responses.
- **C. Containerization**: Seamless integration and deployment using containerization technologies (e.g., Docker).
- **D. Cloud Deployment**: Easily deploy the application to the cloud for enhanced accessibility and scalability.
"""
)

st.sidebar.markdown("### 2. Biography")
st.sidebar.markdown(
    """
I'm a **machine learning** and **deep learning** professional with four years of experience in **data science** and **MLOps engineering**. My strong foundation in **Data Science** & **Data Engineering** has enabled me to develop expertise in building and deploying sophisticated **AI models**. I successfully transitioned from **architecture** to tech four years ago, which gives me a unique perspective and creative problem-solving approach in my ML projects. I specialize in **computer vision**, **generative AI**, and the development of **automated ML pipelines**, constantly pushing the boundaries of what's possible in AI.
"""
)

st.sidebar.markdown("### 3. Skills and Expertise")
st.sidebar.markdown(
    """
- **A. Machine Learning & Deep Learning**: Building and using advanced models for various tasks like classification and prediction.
- **B. Computer Vision**: Analyzing images and videos using AI techniques like object detection and image segmentation.
- **C. Generative AI**: Creating new content with models like GANs, LLMs, and diffusion models.
- **D. Automated ML Pipelines**: Setting up efficient systems to train, test, and deploy ML models.
- **E. 3D AI for Architecture**: Using AI to generate and improve building designs and spaces.
- **F. Smart Cities**: Applying AI to make urban areas more efficient and livable.
- **G. Data Engineering**: Building systems to handle and analyze large amounts of data, including containerization with Docker and orchestration with Kubernetes.
- **H. Cloud Services**: Using platforms like AWS and Google Cloud to run AI solutions at scale.
"""
)

st.sidebar.markdown("### 4. Project Links and Descriptions")
st.sidebar.markdown(
    """
- **[deepTechno](https://github.com/juan-garassino/deepTechno)**  
  A project focused on techno music synthesis using transformer architecture. It works with MIDI files and aims to expand into audio waveforms through VQ encoders.

- **[deepSculpt](https://github.com/juan-garassino/deepSculpt)**  
  A 3D generative adversarial network designed for architectural space generation. Started in 2020, this project integrates AI with architecture and design.

- **[MiniNetworks](https://github.com/juan-garassino)**  
  A collection of small neural networks created for educational purposes, covering RNNs, LSTMs, diffusion models, transformers, GANs, guided diffusion, and CNNs.

*All projects are continuously updated and developed.*
"""
)


st.sidebar.markdown("### 5. Testimonial")
st.sidebar.markdown(
    """'In a short time with our company, Mr. Garassino has shown remarkable enthusiasm, adaptability, and leadership. His expertise has taken him across the globe, teaching Data Science in cities like Berlin, Tokyo, Barcelona, and Amsterdam for our B2C Bootcamps. He’s also a respected colleague and lecturer for our B2B programs, making a significant impact in places like Malaysia, Switzerland, and Dubai.'
  
  **Titiana Benassi**  
  People Manager @ Le Wagon
"""
)


st.sidebar.markdown("### 6. Download Resume")

CV = os.path.join(ROOT, "assets", "JuanGarassinoCV-2024.pdf")

# Custom CSS to make the button full width and use theme colors
st.markdown(
    """
<style>
div[data-testid="stDownloadButton"] > button {
    width: 100%;
    padding: 0.5rem;
    background-color: var(--primary-color);
    color: var(--text-color);
    border: 1px solid var(--primary-color);
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s, color 0.3s, border-color 0.3s;
}
div[data-testid="stDownloadButton"] > button:hover {
    background-color: var(--background-color);
    color: var(--primary-color);
    border-color: var(--primary-color);
}
div[data-testid="stDownloadButton"] {
    width: 100%;
}
</style>
""",
    unsafe_allow_html=True,
)

# Use the sidebar for the download button
with st.sidebar:
    with open(CV, "rb") as file:
        st.download_button(
            label="Download PDF",
            data=file,
            file_name="JuanGarassinoCV-2024.pdf",
            mime="application/pdf",
            key="download_button",
        )

st.sidebar.markdown("### 7. Contact Me")

# Create a row of emojis with links
col1, col2, col3, col4, col5 = st.sidebar.columns(5)

with col1:
    st.markdown("[👨‍💻](https://github.com/juan-garassino)")

with col2:
    st.markdown("[💼](https://www.linkedin.com/in/juan-garassino/)")

with col3:
    st.markdown("[📷](https://www.instagram.com/artista.artificial/)")

with col4:
    st.markdown("[📧](mailto:juan.garassino@gmail.com)")

with col5:
    st.markdown("📱")  # Phone emoji without link

# Add phone number below the emojis
st.sidebar.markdown(
    "<div style='text-align: center;'>+49 0152 24024860</div>", unsafe_allow_html=True
)

# Custom CSS for button styling
st.markdown(
    """
<style>
.stButton > button {
    width: 100%;
    height: auto;
    white-space: normal;
    word-wrap: break-word;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px;
    margin-bottom: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("Chat with Juan's Curriculum 🤓")
st.write(
    "Welcome to my AI-powered chatbot! I designed this app to answer questions about my professional background. Feel free to use the chat to interact with my curriculum. If you have further questions after the chat, let's organize a meeting - I'll be happy to answer the rest of them!"
)

# Initialize chat history
history = StreamlitChatMessageHistory(key="chat_messages")

# Initialize session state
if "button_pressed" not in st.session_state:
    st.session_state.button_pressed = False
if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# Function to process query and update chat
def process_query(query):
    st.chat_message("user").write(query)
    history.add_user_message(query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.write("🧠 Thinking...")

        response = query_engine.query(query)

        response_placeholder.empty()
        response_placeholder.write(response.response)

        history.add_ai_message(response.response)


# Display chat history
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

# If there's no chat history and no button has been pressed, show suggested questions
if not history.messages and not st.session_state.button_pressed:
    # st.markdown("#### Suggested Questions:")
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
if st.session_state.button_pressed and not history.messages:
    process_query(st.session_state.last_question)
    st.session_state.button_pressed = False
    st.session_state.last_question = ""

# Chat input
if prompt := st.chat_input():
    process_query(prompt)
