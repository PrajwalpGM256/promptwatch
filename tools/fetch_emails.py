import argparse
import email
import imaplib
import os
import re
from email.header import decode_header, make_header
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

IMAP_HOST = "imap.gmail.com"

_HIDDEN_TAGS = frozenset({"script", "style", "head"})
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "tr", "li", "table", "section",
        "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
    }
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skipping = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _HIDDEN_TAGS:
            self._skipping = True
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _HIDDEN_TAGS:
            self._skipping = False
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self._chunks.append(data)

    def text(self) -> str:
        joined = "".join(self._chunks)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def _decode(value: str | None) -> str:
    """Decode an RFC 2047 encoded header into plain text."""
    if not value:
        return ""
    return str(make_header(decode_header(value)))


def _decoded_payload(part: Message) -> str:
    raw = part.get_payload(decode=True)
    if not isinstance(raw, bytes):
        return ""
    return raw.decode(part.get_content_charset() or "utf-8", errors="replace")


def _body_text(message: Message) -> str:
    """Extract readable text, preferring text/plain and falling back to HTML.

    Marketing mail is frequently HTML-only, which is exactly the job_alert and
    newsletter traffic the dataset needs, so the HTML path is not optional.
    """
    html_fallback = ""
    for part in message.walk():
        if part.get_filename() or part.get_content_maintype() == "multipart":
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            text = _decoded_payload(part)
            if text.strip():
                return text.strip()
        elif content_type == "text/html" and not html_fallback:
            html_fallback = _decoded_payload(part)

    if html_fallback:
        parser = _TextExtractor()
        parser.feed(html_fallback)
        return parser.text()
    return ""


def fetch(query: str, limit: int) -> list[tuple[str, str]]:
    """Fetch the most recent messages matching a Gmail search query.

    `query` uses Gmail's own search syntax via IMAP X-GM-RAW, so
    'subject:(interview) newer_than:1y' works exactly as in the Gmail UI.

    Returns:
        (subject, body) pairs, newest first.

    Raises:
        RuntimeError: if GMAIL_ADDRESS or GMAIL_APP_PASSWORD is unset, or if
            the search fails.
    """
    address = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not address or not password:
        raise RuntimeError(
            "Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env "
            "(Google Account > Security > App passwords; requires 2FA)."
        )

    with imaplib.IMAP4_SSL(IMAP_HOST) as imap:
        imap.login(address, password)
        imap.select("INBOX", readonly=True)
        imap.literal = query.encode("utf-8")  # type: ignore[assignment]
        status, data = imap.search("UTF-8", "X-GM-RAW")
        if status != "OK":
            raise RuntimeError(f"IMAP search failed for query {query!r}")

        ids = data[0].split()[-limit:]
        messages = []
        for message_id in reversed(ids):
            status, fetched = imap.fetch(message_id, "(RFC822)")
            if status != "OK":
                continue
            head = fetched[0]
            if not isinstance(head, tuple):
                continue
            parsed = email.message_from_bytes(head[1])
            messages.append((_decode(parsed.get("Subject")), _body_text(parsed)))
    return messages


def write_worksheet(messages: list[tuple[str, str]], path: Path) -> None:
    """Write messages to a labeling worksheet with a blank LABEL line each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, (subject, body) in enumerate(messages, start=1):
        blocks.append(
            f"### {index} ###\n"
            f"LABEL: \n"
            f"SUBJECT: {subject}\n"
            f"BODY:\n{body}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=fetch.__doc__)
    parser.add_argument("query", help="Gmail search query, e.g. 'subject:(interview)'")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("raw_emails/batch.txt"))
    args = parser.parse_args()

    found = fetch(args.query, args.limit)
    write_worksheet(found, args.out)
    print(f"{len(found)} messages -> {args.out}")
