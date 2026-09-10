"""Chat exports that come out as HTML rather than plain text.

WhatsApp only ever exports .txt. The apps that hand you HTML are Telegram
Desktop, Meta (Instagram and Facebook Messenger) and DiscordChatExporter, and
each lays a message out differently. The format is detected from markers in the
document rather than from the filename, because all three are usually just
called something like messages.html.

Anything unrecognised falls back to stripping the tags and running the plain
text parser over the result, which handles a WhatsApp export someone saved as a
web page.
"""

import re
from datetime import datetime

from bs4 import BeautifulSoup

from parser import Message, parse_text

TELEGRAM_DATE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2})")

# iMessage timestamps carry read and delivery receipts inline, e.g.
# "Jan 07, 2026  2:17:24 AM (Read by you after 8 hours, 5 minutes, 19 seconds)".
IMESSAGE_RECEIPT_RE = re.compile(
    r"\s*\((?:Read|Delivered|Sent|Edited)[^)]*\)\s*", re.I
)
IMESSAGE_DATE_FORMATS = ("%b %d, %Y %I:%M:%S %p", "%b %d, %Y %H:%M:%S")

# Wrappers whose nested content belongs to some other message.
IMESSAGE_NESTED = ("replies", "reply", "reply_context", "tapbacks")


def _class_list(value) -> list[str]:
    """BeautifulSoup hands back a string for a single class and a list for many."""
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    return list(value)


def _classes(tag) -> str:
    value = tag.get("class") or []
    if isinstance(value, str):
        return value
    return " ".join(value)


def _has(tag, *needles) -> bool:
    joined = _classes(tag)
    return any(n in joined for n in needles)


def _text_of(tag) -> str:
    """Visible text, with <br> kept as line breaks and reactions dropped."""
    if tag is None:
        return ""
    clone = BeautifulSoup(str(tag), "html.parser")
    for junk in clone.find_all(["ul", "script", "style"]):
        junk.decompose()
    for br in clone.find_all("br"):
        br.replace_with("\n")
    return clone.get_text().strip()


def _fix_mojibake(text: str) -> str:
    """Undo Meta's double encoding of non-ASCII characters.

    Meta's HTML exports are famously written as UTF-8 bytes reinterpreted as
    Latin-1, so an emoji arrives as a run of accented characters. Round tripping
    it back fixes emoji and accents; if it does not round trip cleanly the text
    was fine already and is returned untouched.
    """
    if not any(ord(c) in range(0x80, 0x100) for c in text):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


# -- format detection -------------------------------------------------------

def detect(soup: BeautifulSoup) -> str | None:
    if soup.find("div", class_="message_part") and soup.find(
            "span", class_=lambda c: c and "bubble" in _class_list(c)):
        return "imessage"
    if soup.find(class_=lambda c: c and "from_name" in " ".join(
            c if isinstance(c, list) else [c])):
        return "telegram"
    if soup.find(class_="history") and soup.find(class_="message"):
        return "telegram"
    if soup.find(class_=lambda c: c and "chatlog__" in " ".join(
            c if isinstance(c, list) else [c])):
        return "discord"
    if soup.find(class_=lambda c: c and "_a6-g" in " ".join(
            c if isinstance(c, list) else [c])):
        return "meta"
    if soup.find(class_="uiBoxWhite") or soup.find(class_="pam"):
        return "meta"
    return None


# -- per format parsers -----------------------------------------------------

def parse_telegram(soup: BeautifulSoup) -> list[Message]:
    messages: list[Message] = []
    last_sender = None

    for block in soup.find_all("div", class_=lambda c: c and "message" in (
            c if isinstance(c, list) else [c])):
        if _has(block, "service"):
            last_sender = None      # a service line breaks the joined run
            continue

        name_tag = block.find("div", class_="from_name")
        if name_tag is not None:
            last_sender = _text_of(name_tag)
        elif not _has(block, "joined"):
            continue                # no sender and not a continuation
        if not last_sender:
            continue

        text = _text_of(block.find("div", class_="text"))
        if not text:
            continue                # media only

        at = None
        date_tag = block.find("div", class_=lambda c: c and "date" in (
            c if isinstance(c, list) else [c]))
        if date_tag is not None:
            found = TELEGRAM_DATE_RE.search(date_tag.get("title", "") or "")
            if found:
                try:
                    at = datetime.strptime(found.group(1), "%d.%m.%Y %H:%M:%S")
                except ValueError:
                    at = None

        messages.append(Message(last_sender, text, at))

    return messages


def parse_discord(soup: BeautifulSoup) -> list[Message]:
    """DiscordChatExporter. Only the first message in a group names its author."""
    messages: list[Message] = []

    for group in soup.find_all(class_=lambda c: c and "chatlog__message-group" in (
            c if isinstance(c, list) else [c])):
        author_tag = group.find(class_="chatlog__author")
        sender = (author_tag.get("title") if author_tag else None) or \
                 _text_of(author_tag)
        sender = (sender or "").split("#")[0].strip()
        if not sender:
            continue
        for content in group.find_all(class_="chatlog__content"):
            text = _text_of(content)
            if text:
                messages.append(Message(sender, text, None))

    if messages:
        return messages

    # Newer exports drop the group wrapper and repeat the author per message.
    last_sender = None
    for block in soup.find_all(class_=lambda c: c and "chatlog__message" in (
            c if isinstance(c, list) else [c])):
        author_tag = block.find(class_="chatlog__author")
        if author_tag is not None:
            last_sender = ((author_tag.get("title") or _text_of(author_tag))
                           .split("#")[0].strip())
        if not last_sender:
            continue
        text = _text_of(block.find(class_="chatlog__content"))
        if text:
            messages.append(Message(last_sender, text, None))
    return messages


def parse_meta(soup: BeautifulSoup) -> list[Message]:
    """Instagram and Facebook Messenger.

    Meta's class names are obfuscated and change between exports, so the class
    hints are tried first and a structural rule second: a message block is a div
    holding a short sender line, a body, and a date line, in that order.
    """
    messages: list[Message] = []

    blocks = soup.find_all("div", class_=lambda c: c and "_a6-g" in (
        c if isinstance(c, list) else [c]))
    if not blocks:
        blocks = soup.find_all("div", class_=lambda c: c and "uiBoxWhite" in (
            c if isinstance(c, list) else [c]))

    for block in blocks:
        sender_tag = block.find("div", class_=lambda c: c and "_a6-h" in (
            c if isinstance(c, list) else [c]))
        body_tag = block.find("div", class_=lambda c: c and "_a6-p" in (
            c if isinstance(c, list) else [c]))

        if sender_tag is None or body_tag is None:
            children = [d for d in block.find_all("div", recursive=False)]
            if len(children) < 2:
                continue
            sender_tag = sender_tag or children[0]
            body_tag = body_tag or children[1]

        sender = _fix_mojibake(_text_of(sender_tag))
        text = _fix_mojibake(_text_of(body_tag))
        if not sender or not text or sender == text:
            continue
        messages.append(Message(sender, text, None))

    return messages


def _imessage_nested(tag) -> bool:
    """True if this message is quoted inside another one.

    Threaded replies are emitted twice: once in place, and once nested inside
    the message they answer. In this export 1,095 of the 1,097 nested copies
    are byte for byte duplicates of a top level message, so counting them would
    train the model on the same sentences twice and misattribute the rest.
    """
    for parent in tag.parents:
        if any(n in _class_list(parent.get("class")) for n in IMESSAGE_NESTED):
            return True
    return False


def _imessage_bubbles(part) -> list[str]:
    """Text of one message part, taking only the final version of an edit.

    An edited message keeps its whole history in the export: the original
    bubble, an "Edited 6 seconds later" note, then the corrected bubble. Only
    the last one is what the person actually said.
    """
    edited = part.find("div", class_="edited")
    if edited is not None:
        # The history is a table rather than bubbles: one row per version, with
        # the original in the body and each correction in the foot. The last
        # row is what the person actually left standing.
        rows = edited.find_all("tr")
        if rows:
            cells = rows[-1].find_all("td")
            if cells:
                final = _text_of(cells[-1])
                return [final] if final else []

    bubbles = [b for b in part.find_all("span")
               if "bubble" in _class_list(b.get("class"))]
    return [t for t in (_text_of(b) for b in bubbles) if t]


def parse_imessage(soup: BeautifulSoup) -> list[Message]:
    """imessage-exporter HTML output."""
    messages: list[Message] = []

    for block in soup.find_all("div", class_="message"):
        if _imessage_nested(block):
            continue

        holder = None
        for child in block.find_all("div", recursive=False):
            classes = _class_list(child.get("class"))
            if "received" in classes or "sent" in classes:
                holder = child
                break
        if holder is None:
            continue

        header = holder.find("p", recursive=False)
        if header is None:
            continue

        sender_tag = header.find("span", class_="sender")
        sender = _text_of(sender_tag)
        if not sender:
            continue

        at = None
        stamp = header.find("span", class_="timestamp")
        if stamp is not None:
            raw = IMESSAGE_RECEIPT_RE.sub(" ", stamp.get_text(" ", strip=True)).strip()
            for fmt in IMESSAGE_DATE_FORMATS:
                try:
                    at = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue

        lines: list[str] = []
        for part in holder.find_all("div", class_="message_part", recursive=False):
            lines.extend(_imessage_bubbles(part))
        text = "\n".join(lines).strip()
        if not text:
            continue        # attachment only, or a sticker

        messages.append(Message(sender, text, at))

    return messages


# -- entry point ------------------------------------------------------------

def parse_html(raw: str) -> tuple[list[Message], str]:
    """Return the messages and the name of the format they were read as."""
    soup = BeautifulSoup(raw, "html.parser")

    kind = detect(soup)
    parsers = {
        "imessage": parse_imessage,
        "telegram": parse_telegram,
        "discord": parse_discord,
        "meta": parse_meta,
    }

    if kind in parsers:
        messages = parsers[kind](soup)
        if messages:
            return messages, kind

    # Last resort: a plain text export that was saved as a web page.
    for junk in soup.find_all(["script", "style"]):
        junk.decompose()
    messages = parse_text(soup.get_text("\n"))
    return messages, ("text in html" if messages else "unrecognised")


if __name__ == "__main__":
    import sys
    from parser import sender_counts

    with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
        msgs, kind = parse_html(f.read())
    print(f"format: {kind}, {len(msgs)} messages")
    for name, n in sender_counts(msgs)[:10]:
        print(f"  {n:>5}  {name}")
    for m in msgs[:4]:
        print(f"    [{m.at}] {m.sender}: {m.text[:60]!r}")
