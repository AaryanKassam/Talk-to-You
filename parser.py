"""WhatsApp .txt export parser.

Handles both export dialects:
  iOS      [2026-06-16, 9:11:25 PM] Aaryan Kassam: 0-0
  Android  16/06/2026, 21:11 - Aaryan Kassam: 0-0

The invisible characters below are not paranoia. They are all present in real
exports and each one breaks a naive parser:
  U+202F  narrow no-break space, sits between the seconds and AM/PM
  U+200E  left-to-right mark, prefixes system lines AND sometimes the "[" itself
"""

import re
from collections import Counter
from datetime import datetime
from typing import NamedTuple, Optional

NNBSP = " "   # narrow no-break space
LRM = "‎"     # left-to-right mark
ISOLATES = re.compile(r"[\u2066-\u2069]")

# A line that starts a new message, iOS dialect. The sender cannot contain a
# colon, so [^:]+ stops at the first one, which is the sender/body separator.
IOS_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s*(?P<sender>[^:]+?):\s?(?P<body>.*)$")

# Android dialect. The timestamp is matched strictly so that an ordinary
# continuation line containing " - " is not mistaken for a message header.
ANDROID_RE = re.compile(
    r"^(?P<ts>\d{1,2}[/.]\d{1,2}[/.]\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s*[APap]\.?[Mm]\.?)?)\s-\s(?P<sender>[^:]+?):\s(?P<body>.*)$"
)

# Android system lines have a timestamp but no "Sender: " part. Without this
# they would fail the header test and be glued onto the previous message.
# A line that opens with a timestamp but carries no "Sender:" is a system
# line, not the continuation of whatever came before it.
IOS_SYSTEM_RE = re.compile(r"^\[[^\]]+\]\s")

ANDROID_SYSTEM_RE = re.compile(
    r"^\d{1,2}[/.]\d{1,2}[/.]\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s*[APap]\.?[Mm]\.?)?\s-\s"
)

# Placeholder bodies that carry no writing style and must not train the model.
# WhatsApp appends this to the body of any message that was later edited.
EDITED_RE = re.compile(r"\s*\u200e?<this message was edited>\s*$", re.I)

NOISE_BODIES = {
    "<media omitted>", "image omitted", "video omitted", "gif omitted",
    "sticker omitted", "audio omitted", "document omitted",
    "contact card omitted", "this message was deleted",
    "you deleted this message", "missed voice call", "missed video call",
    "null", "<attached>",
}


# Tried in order. The first that parses is cached for the rest of the file,
# so the day/month ambiguity in the Android dialect is at least resolved
# consistently rather than line by line.
TS_FORMATS = (
    "%Y-%m-%d, %I:%M:%S %p", "%Y-%m-%d, %H:%M:%S", "%Y-%m-%d, %H:%M",
    "%d/%m/%Y, %I:%M:%S %p", "%d/%m/%Y, %I:%M %p", "%d/%m/%Y, %H:%M:%S",
    "%d/%m/%Y, %H:%M", "%d/%m/%y, %I:%M:%S %p", "%d/%m/%y, %I:%M %p",
    "%d/%m/%y, %H:%M:%S", "%d/%m/%y, %H:%M",
    "%m/%d/%Y, %I:%M:%S %p", "%m/%d/%Y, %I:%M %p",
    "%m/%d/%y, %I:%M:%S %p", "%m/%d/%y, %I:%M %p",
    # ANDROID_RE accepts a dot as the date separator, so these have to exist
    # or a "16.06.2026" export parses every message with no timestamp at all.
    "%d.%m.%Y, %I:%M:%S %p", "%d.%m.%Y, %I:%M %p", "%d.%m.%Y, %H:%M:%S",
    "%d.%m.%Y, %H:%M", "%d.%m.%y, %I:%M:%S %p", "%d.%m.%y, %I:%M %p",
    "%d.%m.%y, %H:%M:%S", "%d.%m.%y, %H:%M",
)


def parse_timestamp(raw: str, cache: list) -> Optional[datetime]:
    """Best effort timestamp. Returns None rather than guessing wildly.

    Day-first and month-first exports cannot be told apart from a single line,
    so a whole file is read with whichever format matched first. Getting that
    wrong shifts dates but leaves the ordering of messages within a day intact,
    which is all the reply pairing depends on.
    """
    text = raw.replace(NNBSP, " ").replace(LRM, "").strip()

    # Once a format has matched, it is the only one used. Falling back through
    # the whole list per line let one file parse some dates as day-first and
    # others as month-first, which is exactly what this is meant to prevent.
    if cache:
        try:
            return datetime.strptime(text, cache[0])
        except ValueError:
            return None

    for fmt in TS_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        cache.append(fmt)
        return parsed
    return None


class Message(NamedTuple):
    sender: str
    text: str
    at: Optional[datetime] = None


def clean_sender(raw: str) -> str:
    """Strip WhatsApp's display markers from a sender label.

    A leading "~" means "this person is not in your contacts". It is rendering
    chrome, not part of the name, so removing it is safe normalisation. This is
    deliberately NOT the same thing as merging "Aaryan" with "Aaryan Kassam",
    which is fuzzy matching and stays out of scope.
    """
    return raw.replace(LRM, "").replace(NNBSP, " ").lstrip("~").strip()


def _match_header(line: str):
    """Return (sender, body, timestamp) if the line starts a message."""
    line = line.lstrip(LRM)
    for pattern in (IOS_RE, ANDROID_RE):
        m = pattern.match(line)
        if m:
            return (clean_sender(m.group("sender")), m.group("body"),
                    m.group("ts"))
    return None


def _is_system_line(line: str, body: str) -> bool:
    """System and attachment lines, which describe events rather than speech."""
    if body.startswith(LRM):
        return True
    if body.strip().strip("<>").lower() in NOISE_BODIES:
        return True
    return False


def parse_text(raw: str) -> list[Message]:
    """Turn the full text of an export into a list of (sender, text) messages."""
    messages: list[Message] = []
    skip_continuation = False
    ts_cache: list[str] = []

    for line in raw.splitlines():
        header = _match_header(line)

        if header is None:
            # Not a new message. Either a continuation of a multi-line message
            # or the tail of a system line we already discarded.
            stripped = line.lstrip(LRM)
            if ANDROID_SYSTEM_RE.match(stripped) or IOS_SYSTEM_RE.match(stripped):
                skip_continuation = True
                continue
            if messages and not skip_continuation and line.strip():
                prev = messages[-1]
                messages[-1] = Message(
                    prev.sender, prev.text + "\n" + line.rstrip(), prev.at
                )
            continue

        sender, body, raw_ts = header
        if _is_system_line(line, body):
            skip_continuation = True
            continue

        skip_continuation = False
        messages.append(Message(sender, body, parse_timestamp(raw_ts, ts_cache)))

    # Cleanup runs once per message, after every continuation line has been
    # attached. Doing it as each header line arrived meant an edit marker or an
    # isolate mark on the second physical line of a message survived.
    cleaned = []
    for m in messages:
        text = ISOLATES.sub("", EDITED_RE.sub("", m.text)).strip()
        if text:
            cleaned.append(Message(m.sender, text, m.at))
    return cleaned


def parse_file(path: str) -> list[Message]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_text(f.read())


def sender_counts(messages: list[Message]) -> list[tuple[str, int]]:
    """Senders and their message counts, most talkative first."""
    return Counter(m.sender for m in messages).most_common()


def messages_for(messages: list[Message], sender: str) -> list[str]:
    return [m.text for m in messages if m.sender == sender]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit(f"usage: python {sys.argv[0]} <export file>"
                 f"{' <sender name>' if False else ''}")
    path = sys.argv[1]
    msgs = parse_file(path)
    print(f"parsed {len(msgs)} messages\n")
    print("senders by message count:")
    for name, n in sender_counts(msgs):
        print(f"  {n:>6}  {name}")
    dated = sum(1 for m in msgs if m.at)
    print(f"\ntimestamps parsed on {dated} of {len(msgs)} messages")
    print("first 5 messages:")
    for m in msgs[:5]:
        print(f"  [{m.at}] {m.sender}: {m.text[:50]!r}")
