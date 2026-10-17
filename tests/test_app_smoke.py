"""End-to-end smoke test of the Streamlit script with every backend mocked.

Skipped unless the app's deps are installed (`pip install -r requirements.txt`).
No network: LLM, embeddings, Neo4j, Pinecone and the agent are all fakes.
"""
import os
from unittest import mock

import pytest

pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("llama_index.graph_stores.neo4j")
pytest.importorskip("llama_index.vector_stores.pinecone")
pytest.importorskip("llama_index.llms.azure_openai")
pytest.importorskip("llama_index.embeddings.azure_openai")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = {k: "test" for k in (
    "AZURE_OPENAI_APIKEY", "AZURE_OPENAI_ENDPOINT_LLM", "AZURE_OPENAI_ENDPOINT_EMBEDDING",
    "AZURE_OPENAI_API_VERSION", "PINECONE_API_KEY", "NEO4J_USERNAME", "NEO4J_PASSWORD",
    "NEO4J_CONNECTION_URI", "GMAIL_APP_PASSWORD")}


def make_fake_llm():
    """A local LLM that always emits a final ReAct answer echoing the user turn."""
    from llama_index.core.llms import (CompletionResponse, CustomLLM, LLMMetadata,
                                       ChatMessage, ChatResponse)

    class FakeLLM(CustomLLM):
        calls: list = []

        @property
        def metadata(self):
            return LLMMetadata(is_chat_model=True)

        def _answer(self, text):
            return f"Thought: I can answer without tools.\nAnswer: answer to {text}"

        def chat(self, messages, **kw):
            user = [m for m in messages if m.role == "user"][-1].content
            self.calls.append(user)
            return ChatResponse(message=ChatMessage(role="assistant", content=self._answer(user)))

        async def achat(self, messages, **kw):
            return self.chat(messages, **kw)

        def complete(self, prompt, formatted=False, **kw):
            return CompletionResponse(text=self._answer(prompt))

        def stream_complete(self, prompt, formatted=False, **kw):
            yield self.complete(prompt)

    return FakeLLM()


@pytest.fixture
def app(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    llm = make_fake_llm()
    targets = {
        "llama_index.llms.azure_openai.AzureOpenAI": mock.MagicMock(return_value=llm),
        "llama_index.embeddings.azure_openai.AzureOpenAIEmbedding": mock.MagicMock(),
        "llama_index.graph_stores.neo4j.Neo4jPropertyGraphStore": mock.MagicMock(),
        "llama_index.vector_stores.pinecone.PineconeVectorStore": mock.MagicMock(),
        "llama_index.core.PropertyGraphIndex.from_existing": mock.MagicMock(),
        "llama_index.core.VectorStoreIndex.from_vector_store": mock.MagicMock(),
        "llama_index.core.Settings": mock.MagicMock(),
    }
    patches = [mock.patch(t, new) for t, new in targets.items()]
    for p in patches:
        p.start()
    import streamlit as st
    st.cache_resource.clear()  # cached backends would otherwise outlive this test's mocks
    at = AppTest.from_file(os.path.join(ROOT, "achievements_streamlit.py"), default_timeout=60)
    yield at, llm, targets
    for p in patches:
        p.stop()


def test_app_answers_chat_through_guard(app):
    # Real llama-index workflow ReActAgent (G-019) driven by a fake local LLM.
    at, llm, _ = app
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("hello").run()
    assert not at.exception, at.exception
    assert any("hello" in c for c in llm.calls)
    assert at.session_state.messages[-1]["content"] == "answer to hello"


def test_backends_not_rebuilt_on_rerun(app):
    # G-018 behaviourally: a second message must not reconnect to Neo4j.
    at, _, targets = app
    neo4j = targets["llama_index.graph_stores.neo4j.Neo4jPropertyGraphStore"]
    at.run()
    at.chat_input[0].set_value("one").run()
    at.chat_input[0].set_value("two").run()
    assert not at.exception
    assert neo4j.call_count == 1
