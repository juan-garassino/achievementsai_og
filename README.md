# AchievementsAI

A Streamlit chatbot that answers questions about Juan Garassino's career, using a ReAct agent over a vector index and a knowledge graph.

<p align="center">
  <img src="assets/robot-assistant.png" alt="AchievementsAI assistant icon" width="220">
</p>

## What and why

A CV is a static document. This app lets a recruiter or student ask questions about a career in plain language ("What was Juan's role while he lived in New Zealand?") and get answers grounded in a curated biography. A LlamaIndex `ReActAgent` picks between semantic search and relationship queries, and can send an email to Juan from the chat.

## Highlights

- **Two retrieval tools** behind one agent: `biography_semantic_search` (Pinecone vector index) and `biography_relationship_query` (Neo4j property graph) in [`achievements_streamlit.py`](achievements_streamlit.py).
- **Azure OpenAI** `gpt-4o-mini` for the LLM and `text-embedding-3-large` for embeddings.
- **`send_gmail` function tool** so the agent can email Juan via Gmail SMTP.
- **Streamlit UI** with suggested questions, chat history, a sidebar bio and a CV download button.
- **Source corpus** of biography drafts in [`data/raw/`](data/raw) (Markdown and text).
- **Deployment scaffolding**: a Codespaces devcontainer and a `supervisord.conf` that runs Ollama and Streamlit side by side.

## Quickstart

The app cannot run without your own cloud services. It reads existing indexes: a Pinecone index and namespace named `achievementsai`, and a Neo4j graph that must already be populated. No script in this repo that populates them runs as-is (see Status).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env        # then fill in values; never commit .env
```

Environment variables read by the app (names only; each can alternatively come from `st.secrets`):
`AZURE_OPENAI_APIKEY`, `AZURE_OPENAI_ENDPOINT_LLM`, `AZURE_OPENAI_ENDPOINT_EMBEDDING`, `AZURE_OPENAI_API_VERSION`, `PINECONE_API_KEY`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_CONNECTION_URI`, `GMAIL_APP_PASSWORD`.

```bash
export $(grep -v '^#' .env | xargs)   # or use direnv
streamlit run achievements_streamlit.py
```

Verified in a CPU sandbox: `pip install -r requirements.txt` succeeds on Python 3.12 and `streamlit` and `llama_index.core` import. Launching the app was not verified because it needs live Azure, Pinecone, Neo4j and Gmail credentials. `.env.sample` lists more variables than the app uses (Anthropic, Ollama, GraphRAG).

## Project structure

```
achievements_streamlit.py   Streamlit app: sidebar, agent, tools, chat loop
achievements_ingestion.py   Draft ingestion script (Chroma + LlamaIndex), incomplete
data/raw/                   Biography source texts (.md, .txt)
assets/                     Robot icons, CV PDF (downloaded by the sidebar button)
models_store/               Three joblib models (iris, diabetes, blood sugar); not used by the app
supervisord.conf            Runs `ollama serve` and Streamlit in a container
.devcontainer/              Codespaces config (Python 3.11 image)
.env.sample                 Keys-only template
```

## Tech stack

Python, Streamlit, LlamaIndex, Azure OpenAI, Pinecone, Neo4j, ChromaDB, Hugging Face embeddings (ingestion draft), Ollama (container config only).

## Status

Work in progress. The repo contains no results or evaluation metrics.

- `achievements_ingestion.py` does not run: `SimpleDirectoryReader`, `PropertyGraphIndex`, `OpenAIEmbedding` and `chroma_collection` are used without being imported or defined, and it targets Chroma rather than the Pinecone and Neo4j stores the app reads.
- `supervisord.conf` and the devcontainer refer to `AchievementsAI.py`, which does not exist; the app file is `achievements_streamlit.py`.
- `requirements.txt` is unpinned and does not list `chromadb` or the Hugging Face embedding package used by the ingestion draft.
- No tests, no license file.
