"""Contact-email tool exposed to the chat agent.

The agent is driven by untrusted user prompts, so the tool must not let a
visitor choose the recipient (that would turn the app's Gmail account into an
open mail relay). The recipient is fixed to the owner's contact address and the
subject is stripped of CR/LF to prevent header injection.
"""
import os
import smtplib
import threading
import time
from collections import defaultdict, deque
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

DEFAULT_SENDER = "achievements.ai.backend@gmail.com"
DEFAULT_RECIPIENT = "juan.garassino@gmail.com"  # same address as the app's public mailto link
MAX_SUBJECT_LEN = 200
MAX_MESSAGE_LEN = 5000
MAX_PER_SESSION = 3        # emails one visitor session may send per window
MAX_GLOBAL = 20            # emails the whole process may send per window
WINDOW_SECONDS = 3600


class RateLimiter:
    """Sliding-window limiter: at most `per_key` sends per key and `global_max`
    sends overall within `window` seconds. Thread-safe (Streamlit runs each
    session's script in its own thread)."""

    def __init__(self, per_key=MAX_PER_SESSION, global_max=MAX_GLOBAL, window=WINDOW_SECONDS):
        self.per_key, self.global_max, self.window = per_key, global_max, window
        self._keys = defaultdict(deque)
        self._all = deque()
        self._lock = threading.Lock()

    def _prune(self, q, now):
        while q and q[0] <= now - self.window:
            q.popleft()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._keys[key]
            self._prune(q, now)
            self._prune(self._all, now)
            if len(q) >= self.per_key or len(self._all) >= self.global_max:
                return False
            q.append(now)
            self._all.append(now)
            return True


# Module-level: shared by all sessions in the process (modules survive reruns).
LIMITER = RateLimiter()


def build_message(subject: str, message: str, sender: str, recipient: str) -> MIMEMultipart:
    subject = " ".join(str(subject).splitlines()).strip()[:MAX_SUBJECT_LEN]
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject or "(no subject)"
    msg.attach(MIMEText(str(message)[:MAX_MESSAGE_LEN], "plain"))
    return msg


def send_gmail(subject: str, message: str, app_password: str | None = None,
               session_id: str = "anonymous") -> str:
    """Send a message to Juan (the owner). The recipient cannot be chosen by the caller."""
    sender = os.getenv("GMAIL_SENDER", DEFAULT_SENDER)
    recipient = os.getenv("CONTACT_EMAIL", DEFAULT_RECIPIENT)
    password = app_password or os.getenv("GMAIL_APP_PASSWORD")
    if not password:
        return "Email is not configured."
    if not LIMITER.allow(session_id):
        return "Email rate limit reached; please try again later or use the contact links."
    msg = build_message(subject, message, sender, recipient)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
    return "Email sent successfully!"
