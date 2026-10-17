# GUARDIAN

Automated security & bug guardian for this repo. Branch: `claude/guardian`.

## Checks

| Area | Command |
|---|---|
| Dependencies | `pip-audit -r requirements.txt` (resolves unpinned deps, incl. transitive) |
| Static analysis | `bandit -r .` + manual grep for eval/exec, pickle/joblib loads, yaml.load, `shell=True`, `verify=False`, CORS/XSRF, 0.0.0.0 servers |
| Secrets (history) | `gitleaks detect --source . --redact -v` |
| Secrets (tree) | `gitleaks detect --no-git --source . --redact` + `git ls-files \| grep -i env` |
| Workflows | none present (`.github/workflows` absent) |
| Containers | `supervisord.conf`, `.devcontainer/devcontainer.json` reviewed (no Dockerfile) |
| Lint | `python -m pyflakes .` (no linter configured in repo) |
| Tests | `python -m pytest -q --cov=email_tool` (tests need only pytest; no network, SMTP mocked) |
| Abuse limits | email tool `RateLimiter` (per-session + global window) and `MAX_TURNS_PER_SESSION` LLM cap stay in place |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | high | LLM tool lets any visitor choose the email recipient: the public chat agent could be prompt-driven into sending arbitrary mail from the app's Gmail account (open relay / spam / phishing); subject header injection possible | achievements_streamlit.py | fixed (recipient fixed in `email_tool.py`, CR/LF stripped, length capped) | 2026-10-15 |
| G-002 | medium | Prefixes of API keys/passwords printed to stdout on every run (lands in hosting logs) | achievements_streamlit.py | fixed | 2026-10-15 |
| G-003 | low | Bug: chat history stored the agent response object instead of its text | achievements_streamlit.py | fixed | 2026-10-15 |
| G-004 | medium | Bug: ingestion script crashed with NameError (undefined `SimpleDirectoryReader`, `chroma_collection`, `PropertyGraphIndex`, `OpenAIEmbedding`) | achievements_ingestion.py | fixed (rewritten as runnable vector ingestion; broken graph section dropped) | 2026-10-15 |
| G-005 | low | Bug: supervisord/devcontainer launch a non-existent `AchievementsAI.py` | supervisord.conf, .devcontainer/devcontainer.json | fixed | 2026-10-15 |
| G-006 | low | Devcontainer ran Streamlit with CORS and XSRF protection disabled | .devcontainer/devcontainer.json | fixed (defaults restored) | 2026-10-15 |
| G-010 | medium | Privacy: agent built in `st.cache_resource` with a module-level `ChatMemoryBuffer`, so every visitor shared one conversation memory (cross-session leak) | achievements_streamlit.py | fixed (agent + memory per `st.session_state`) | 2026-10-15 |
| G-011 | medium | Email tool had no rate limit: a visitor (or prompt injection) could make the agent mail the owner without bound (spam, Gmail quota/account lockout) | email_tool.py, achievements_streamlit.py | fixed (3/session/hour, 20/process/hour) | 2026-10-17 |
| G-012 | medium | Unbounded LLM calls per browser session on a public app (Azure OpenAI cost abuse) | achievements_streamlit.py | fixed (30 turns/session; new sessions still bypass, see backlog) | 2026-10-17 |
| G-014 | low | Agent built with `verbose=True`: ReAct traces (visitor messages, email bodies) printed to hosting logs | achievements_streamlit.py | fixed (`verbose=False`) | 2026-10-17 |
| G-013 | low | `pillow` / `nest_asyncio` imported directly but only installed transitively | requirements.txt | fixed | 2026-10-17 |
| G-007 | medium | Transitive dep `nltk` (via llama-index-core) has advisory PYSEC-2026-3740, no fix version published | requirements.txt | open (re-checked 2026-10-17: nltk 3.10.3, still no fix) | 2026-10-15 |
| G-008 | low | `requirements.txt` fully unpinned (non-reproducible, supply-chain drift) | requirements.txt | open | 2026-10-15 |
| G-009 | low | `.joblib` (pickle) model files tracked; unused by the app, unsafe to load if untrusted | models_store/ | open | 2026-10-15 |

Secrets: gitleaks found 0 findings in history and tree (re-run 2026-10-17, 47 commits, all branches); `.env.sample` holds keys only, no values.

## Metrics

| metric | run #1 before | run #1 after | run #2 after |
|---|---|---|---|
| open vulns crit/high/med/low | 0/1/3/3 | 0/0/1/2 | 0/0/1/2 |
| findings fixed this run | - | 7 | 4 (G-011..G-014) |
| failing tests | n/a (no tests) | 0 | 0 |
| test count | 0 | 9 | 16 |
| coverage (`email_tool.py`) | n/a | 100% | 100% |
| lint errors (pyflakes) / bandit | 5 / - | 0 / 0 | 0 / 0 |

(Run #2 baseline counts G-011/G-012 as newly found: 0/0/3/4 -> 0/0/1/2.)

## Backlog

- Pin `requirements.txt` (or adopt a lock file) and re-run pip-audit; bump `nltk` when a fix ships (G-007, G-008).
- Remove or document `models_store/*.joblib` (G-009).
- Per-session caps (G-012) are bypassed by opening new sessions; a process-wide LLM budget or per-IP limit (reverse proxy) would close that.
- Chat UI does not render the user's message until the next rerun (append-only); minor UX.
- No CI workflow (needs a token with workflow scope; see Ideas).

## Learnings

- App is a Streamlit RAG chatbot with Azure OpenAI + Pinecone + Neo4j; nothing can run end-to-end without cloud credentials, so tests target the pure-Python pieces (`email_tool.py`) and static checks on the scripts.
- Modules imported by a Streamlit script persist across reruns and sessions, so module-level state in `email_tool.py` is process-wide (good for a global limiter, bad for per-user data).
- Any tool exposed to the agent is reachable by anonymous visitors through prompt injection; keep tool arguments narrow.

## Ideas

1. Split the app into `config.py` / `agent.py` / UI so agent wiring can be unit-tested with mocked LLMs.
2. Add a GitHub Actions workflow (pinned SHAs, `permissions: contents: read`) running pytest + pyflakes + gitleaks.
3. Cache only the expensive shared indexes (`st.cache_resource`) and keep the cheap per-session agent, to cut cold-start time.
4. Move `ask_agent` and the limits into a small importable module so the turn cap can be unit-tested behaviourally instead of by source inspection.

## Log

- 2026-10-15 run #1: first scan. Fixed open email relay in the agent tool (G-001), secret-prefix logging (G-002), three bugs (G-003..G-005), devcontainer CSRF/CORS defaults (G-006) and cross-visitor chat-memory sharing (G-010). Added 9 tests (pytest, SMTP mocked). pip-audit flagged transitive nltk (no fix yet). No secrets found.

- 2026-10-17 run #2: no PR review comments, no CI configured. Re-ran pip-audit (nltk still unfixed), bandit (0), gitleaks (0, history + tree), pyflakes (0). Added email rate limiting (G-011), per-session LLM turn cap (G-012), explicit pillow/nest_asyncio deps (G-013), agent verbose logging off (G-014). Tests 9 -> 16, all passing.
