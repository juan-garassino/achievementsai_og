from types import SimpleNamespace

import chat_guard
from email_tool import RateLimiter


class FakeAgent:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def chat(self, q):
        self.calls.append(q)
        if self.fail:
            raise RuntimeError("upstream 500: secret-endpoint.example")
        return SimpleNamespace(response=f"echo {q}")


def test_answer_returns_text():
    lim = RateLimiter(per_key=5, global_max=5)
    agent = FakeAgent()
    assert chat_guard.answer(agent, "hi", "s1", limiter=lim) == "echo hi"
    assert agent.calls == ["hi"]


def test_session_cap_blocks_llm_call():
    # G-012: per-session cap enforced before the LLM is called.
    lim = RateLimiter(per_key=1, global_max=10)
    agent = FakeAgent()
    chat_guard.answer(agent, "a", "s1", limiter=lim)
    assert chat_guard.answer(agent, "b", "s1", limiter=lim) == chat_guard.LIMIT_MESSAGE
    assert agent.calls == ["a"]


def test_global_cap_blocks_new_sessions():
    # G-015: opening new sessions must not bypass the process-wide LLM budget.
    lim = RateLimiter(per_key=10, global_max=2)
    agent = FakeAgent()
    for sid in ("s1", "s2"):
        chat_guard.answer(agent, "q", sid, limiter=lim)
    assert chat_guard.answer(agent, "q", "s3", limiter=lim) == chat_guard.LIMIT_MESSAGE
    assert len(agent.calls) == 2


def test_default_limiter_is_process_wide():
    assert chat_guard.LLM_LIMITER.global_max == chat_guard.MAX_GLOBAL_TURNS
    assert chat_guard.LLM_LIMITER.per_key == chat_guard.MAX_TURNS_PER_SESSION


def test_agent_error_is_not_shown_to_visitor():
    # G-016: an agent exception crashed the page and showed its traceback/details.
    lim = RateLimiter(per_key=5, global_max=5)
    out = chat_guard.answer(FakeAgent(fail=True), "q", "s1", limiter=lim)
    assert out == chat_guard.ERROR_MESSAGE
    assert "secret-endpoint" not in out


def test_app_uses_chat_guard():
    src = open("achievements_streamlit.py", encoding="utf-8").read()
    assert "chat_guard.answer(" in src and "agent.chat(" not in src


def test_backends_built_once_per_process():
    # G-018: LLM clients, the Neo4j driver and the Pinecone index were rebuilt on
    # EVERY Streamlit rerun (each chat message), leaking Neo4j driver connections.
    import ast
    tree = ast.parse(open("achievements_streamlit.py", encoding="utf-8").read())
    cached = [f for f in tree.body if isinstance(f, ast.FunctionDef)
              and any("cache_resource" in ast.unparse(d) for d in f.decorator_list)]
    inside = {ast.unparse(c.func) for f in cached for c in ast.walk(f) if isinstance(c, ast.Call)}
    top = {ast.unparse(c.func) for s in tree.body if not isinstance(s, ast.FunctionDef)
           for c in ast.walk(s) if isinstance(c, ast.Call)}
    for ctor in ("Neo4jPropertyGraphStore", "PineconeVectorStore", "AzureOpenAI"):
        assert ctor in inside and ctor not in top, ctor


def test_vector_index_gets_explicit_embed_model():
    # Inside load_backends Settings.embed_model is not set yet; without an explicit
    # embed_model llama-index falls back to the default OpenAI embedding.
    src = open("achievements_streamlit.py", encoding="utf-8").read()
    assert "from_vector_store(vector_store, embed_model=embeddings)" in src


def test_workflow_adapter_runs_agent_with_session_memory():
    # G-019: llama-index >= 0.13 removed ReActAgent.from_tools / .chat; the app
    # crashed on a fresh install. The adapter drives the workflow agent's run().
    seen = []

    class FakeWorkflowAgent:
        def run(self, user_msg, memory):
            seen.append((user_msg, memory))

            async def done():
                return "workflow answer"
            return done()

    memory = object()
    wrapped = chat_guard.WorkflowChatAgent(FakeWorkflowAgent(), memory)
    assert wrapped.chat("hi").response == "workflow answer"
    assert seen == [("hi", memory)]
