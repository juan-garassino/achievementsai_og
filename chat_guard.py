"""Guarded LLM turn for the public chat app (no Streamlit import: unit-testable).

Every visitor message costs Azure OpenAI tokens, so each turn passes a
sliding-window limiter keyed by browser session AND a process-wide budget
(opening new sessions must not bypass the cap). Agent errors are logged
server-side and replaced by a generic message, so upstream details
(endpoints, deployment names, tracebacks) never reach the visitor.
"""
import asyncio
import logging
from types import SimpleNamespace

from email_tool import RateLimiter

MAX_TURNS_PER_SESSION = 30   # per session per hour
MAX_GLOBAL_TURNS = 300       # whole process per hour
LIMIT_MESSAGE = ("This chat has reached its message limit. "
                 "Please use the contact links in the sidebar to reach Juan.")
ERROR_MESSAGE = "Sorry, something went wrong answering that. Please try again later."

# Module-level: Streamlit keeps imported modules across reruns and sessions.
LLM_LIMITER = RateLimiter(per_key=MAX_TURNS_PER_SESSION, global_max=MAX_GLOBAL_TURNS, window=3600)

log = logging.getLogger(__name__)


def answer(agent, question: str, session_id: str, limiter: RateLimiter | None = None) -> str:
    """Return the agent's text answer, or a limit/error message. `agent` needs `.chat(str)`."""
    limiter = LLM_LIMITER if limiter is None else limiter
    if not limiter.allow(session_id):
        return LIMIT_MESSAGE
    try:
        return str(agent.chat(question).response)
    except Exception:  # noqa: BLE001 - never surface upstream details to visitors
        log.exception("agent.chat failed")
        return ERROR_MESSAGE


class WorkflowChatAgent:
    """`.chat(q) -> obj.response` over a llama-index workflow agent (>= 0.13).

    The workflow ReActAgent has no sync `.chat`; `run()` must be awaited inside
    a running loop. `memory` is this session's own buffer (never shared).
    """

    def __init__(self, agent, memory):
        self.agent, self.memory = agent, memory

    def chat(self, question: str):
        async def _run():
            return await self.agent.run(user_msg=question, memory=self.memory)
        return SimpleNamespace(response=str(asyncio.run(_run())))
