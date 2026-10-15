"""Contact-email tool exposed to the chat agent.

The agent is driven by untrusted user prompts, so the tool must not let a
visitor choose the recipient (that would turn the app's Gmail account into an
open mail relay). The recipient is fixed to the owner's contact address and the
subject is stripped of CR/LF to prevent header injection.
"""
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

DEFAULT_SENDER = "achievements.ai.backend@gmail.com"
DEFAULT_RECIPIENT = "juan.garassino@gmail.com"  # same address as the app's public mailto link
MAX_SUBJECT_LEN = 200
MAX_MESSAGE_LEN = 5000


def build_message(subject: str, message: str, sender: str, recipient: str) -> MIMEMultipart:
    subject = " ".join(str(subject).splitlines()).strip()[:MAX_SUBJECT_LEN]
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject or "(no subject)"
    msg.attach(MIMEText(str(message)[:MAX_MESSAGE_LEN], "plain"))
    return msg


def send_gmail(subject: str, message: str, app_password: str | None = None) -> str:
    """Send a message to Juan (the owner). The recipient cannot be chosen by the caller."""
    sender = os.getenv("GMAIL_SENDER", DEFAULT_SENDER)
    recipient = os.getenv("CONTACT_EMAIL", DEFAULT_RECIPIENT)
    password = app_password or os.getenv("GMAIL_APP_PASSWORD")
    if not password:
        return "Email is not configured."
    msg = build_message(subject, message, sender, recipient)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
    return "Email sent successfully!"
