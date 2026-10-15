import inspect
from unittest import mock

import email_tool


def test_recipient_is_not_caller_controlled():
    # G-001: the LLM tool must not accept an arbitrary recipient (open relay).
    assert "to_email" not in inspect.signature(email_tool.send_gmail).parameters


def test_send_goes_only_to_owner(monkeypatch):
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    with mock.patch("email_tool.smtplib.SMTP") as smtp:
        out = email_tool.send_gmail("Hi", "hello", app_password="x")
    server = smtp.return_value.__enter__.return_value
    sent = server.send_message.call_args[0][0]
    assert sent["To"] == email_tool.DEFAULT_RECIPIENT
    assert out == "Email sent successfully!"


def test_subject_header_injection_stripped():
    msg = email_tool.build_message("Hi\r\nBcc: victim@example.com", "x", "a@b.c", "d@e.f")
    assert "\n" not in msg["Subject"] and "\r" not in msg["Subject"]
    assert msg["Bcc"] is None


def test_message_length_capped():
    msg = email_tool.build_message("s", "a" * 10_000, "a@b.c", "d@e.f")
    assert len(msg.get_payload()[0].get_payload()) <= email_tool.MAX_MESSAGE_LEN


def test_unconfigured_does_not_connect(monkeypatch):
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    with mock.patch("email_tool.smtplib.SMTP") as smtp:
        assert email_tool.send_gmail("s", "m") == "Email is not configured."
    smtp.assert_not_called()


def test_app_does_not_print_secret_prefixes():
    # G-002: secret prefixes were printed to stdout (ends up in hosting logs).
    src = open("achievements_streamlit.py", encoding="utf-8").read()
    assert "[0:5]" not in src and "[:5]" not in src


def test_chat_history_stores_text_not_response_object():
    # G-003: history stored the AgentChatResponse object instead of its text.
    src = open("achievements_streamlit.py", encoding="utf-8").read()
    assert '"content": response})' not in src


def test_ingestion_script_has_no_undefined_names():
    # G-004: ingestion script referenced undefined names (NameError on run).
    import ast, builtins
    tree = ast.parse(open("achievements_ingestion.py", encoding="utf-8").read())
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
            defined |= {a.arg for a in node.args.args} if isinstance(node, ast.FunctionDef) else set()
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert used <= defined, used - defined


def test_agent_memory_is_per_session_not_shared():
    # G-010: a module-level ChatMemoryBuffer inside a cache_resource agent is
    # shared by every visitor (cross-session conversation leak).
    src = open("achievements_streamlit.py", encoding="utf-8").read()
    assert "@st.cache_resource" not in src.split("def load_agent")[0].splitlines()[-1]
    assert "\nmemory = ChatMemoryBuffer" not in src
    assert "st.session_state.agent" in src
