import re
from html import unescape
from email.utils import getaddresses
from bs4 import BeautifulSoup

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def extract_body(msg):
    """Extract plain-text body from an email message.

    For multipart messages, prefers the ``text/plain`` part. Falls back to
    the first ``text/html`` part (with tags stripped) only if no plain-text
    payload exists. Concatenating both would leak HTML markup into the
    classifier and bias it toward HTML-bearing messages.
    """

    if msg.is_multipart():
        plain_texts: list[str] = []
        html_texts: list[str] = []

        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type not in ("text/plain", "text/html"):
                continue

            payload = part.get_payload(decode=True)
            if not payload:
                continue

            try:
                text = payload.decode(errors="ignore")
            except Exception:
                continue

            text = unescape(text)
            text = BeautifulSoup(text, "html.parser").get_text(" ")

            if content_type == "text/plain":
                plain_texts.append(text)
            else:
                html_texts.append(text)

        texts = plain_texts if plain_texts else html_texts
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            try:
                text = payload.decode(errors="ignore")
            except Exception:
                text = ""
            text = unescape(text)
            text = BeautifulSoup(text, "html.parser").get_text(" ")
            texts = [text]
        else:
            texts = []

    clean = " ".join(texts)
    clean = re.sub(r"[\r\n\t]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def all_recipients(msg):
    fields = []
    for h in ["From", "To", "Cc", "Bcc"]:
        fields.extend(getaddresses([msg.get(h, "")]))
    return ", ".join(sorted(set(addr for _, addr in fields if addr)))


def clean_text(text):
    """Sanitize text for safe CSV/spreadsheet export.

    - Strips control characters that break parsers.
    - Truncates at Excel's per-cell limit.
    - Prefixes formula-trigger characters with a single quote so they are
      treated as literal text rather than formulas (mitigates CSV injection).
    """
    if not isinstance(text, str):
        return text
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\u200B\u200C\u200D\u200E\u200F\uFEFF]", "", text)
    text = text[:32767]
    if text.startswith(_FORMULA_PREFIXES):
        text = "'" + text
    return text