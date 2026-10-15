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
| G-007 | medium | Transitive dep `nltk` (via llama-index-core) has advisory PYSEC-2026-3740, no fix version published | requirements.txt | open (no fix available) | 2026-10-15 |
| G-008 | low | `requirements.txt` fully unpinned (non-reproducible, supply-chain drift) | requirements.txt | open | 2026-10-15 |
| G-009 | low | `.joblib` (pickle) model files tracked; unused by the app, unsafe to load if untrusted | models_store/ | open | 2026-10-15 |

Secrets: gitleaks found 0 findings in history and tree; `.env.sample` holds keys only, no values.

## Metrics

| metric | before | after |
|---|---|---|
| open vulns crit/high/med/low | 0/1/3/3 | 0/0/1/2 |
| failing tests | n/a (no tests) | 0 |
| test count | 0 | 9 |
| coverage (`email_tool.py`) | n/a | 100% |
| lint errors (pyflakes) | 5 | 0 |

## Backlog

- Pin `requirements.txt` (or adopt a lock file) and re-run pip-audit; bump `nltk` when a fix ships (G-007, G-008).
- Remove or document `models_store/*.joblib` (G-009).
- Add rate limiting / per-session cap on the email tool and on LLM calls (cost abuse on a public app).
- `requirements.txt` lacks `pillow` / `nest_asyncio` explicitly (currently transitive).

## Learnings

- App is a Streamlit RAG chatbot with Azure OpenAI + Pinecone + Neo4j; nothing can run end-to-end without cloud credentials, so tests target the pure-Python pieces (`email_tool.py`) and static checks on the scripts.
- Any tool exposed to the agent is reachable by anonymous visitors through prompt injection; keep tool arguments narrow.

## Ideas

1. Split the app into `config.py` / `agent.py` / UI so agent wiring can be unit-tested with mocked LLMs.
2. Add a GitHub Actions workflow (pinned SHAs, `permissions: contents: read`) running pytest + pyflakes + gitleaks.
3. Cache only the expensive shared indexes (`st.cache_resource`) and keep the cheap per-session agent, to cut cold-start time.

## Log

- 2026-10-15 run #1: first scan. Fixed open email relay in the agent tool (G-001), secret-prefix logging (G-002), three bugs (G-003..G-005), devcontainer CSRF/CORS defaults (G-006) and cross-visitor chat-memory sharing (G-010). Added 9 tests (pytest, SMTP mocked). pip-audit flagged transitive nltk (no fix yet). No secrets found.
