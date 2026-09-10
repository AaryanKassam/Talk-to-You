"""Retrieval over the reply pairs already sitting in a chat export.

The export is not just a pile of sentences. It is a record of what people said
to you and what you actually said back. Pairing those up gives a corpus of real
replies, which is a far better grounding for a persona than anything a model
would invent.

Matching is BM25 over word unigrams and bigrams, hand rolled because the corpus
is a few thousand short documents and pulling in scikit-learn for that would be
heavier than the whole rest of the app. TF-IDF cosine was tried first and is
wrong for this corpus; ReplyIndex explains why.
"""

import math
import re
from collections import Counter, defaultdict
from typing import NamedTuple, Optional

WORD_RE = re.compile(r"[a-z0-9']+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

MAX_GAP_MINUTES = 30      # beyond this it is a new conversation, not a reply
MAX_TURN_CHARS = 400
MAX_REPLY_CHARS = 300
MIN_SCORE = 0.21          # tuned across both corpora; see README

# Dropped from queries only. The index still holds them, but a question made
# entirely of these carries no topic, and an unseen bigram of two of them, say
# "when_they", was being scored as maximally rare and so maximally informative,
# which drowned out the one real word the conversation had to offer.
QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "is", "are", "was", "were", "be", "been", "am", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "them", "my", "your", "his", "our", "their", "as",
    "so", "do", "did", "does", "done", "not", "no", "up", "out", "then",
    "what", "when", "who", "why", "how", "where", "which", "about", "just",
    "u", "ur", "im", "ive", "dont", "cant", "yeah", "ya", "ok", "okay", "lol",
    "like", "got", "get", "gonna", "wanna", "can", "will", "would", "should",
    "have", "has", "had", "there", "here", "one", "all", "some", "any", "too",
}
K1 = 1.5                  # BM25 term frequency saturation
B = 0.75                  # BM25 length normalisation
BIGRAM_WEIGHT = 0.5       # word order is a hint, not half the evidence
CONTEXT_DECAY = 0.7       # how fast earlier turns stop mattering
CONTEXT_TURNS = 4         # how far back the conversation is carried


class Pair(NamedTuple):
    prompt: str     # what someone said to you
    reply: str      # what you actually said back
    who: str        # who said the prompt


def query_tokens(text: str) -> list[str]:
    """Tokens worth searching on: content words, and bigrams that contain one."""
    words = WORD_RE.findall(URL_RE.sub(" ", text.lower()))
    out = [w for w in words if w not in QUERY_STOPWORDS]
    out += [
        f"{a}_{b}" for a, b in zip(words, words[1:])
        if a not in QUERY_STOPWORDS or b not in QUERY_STOPWORDS
    ]
    return out


def tokens(text: str) -> list[str]:
    words = WORD_RE.findall(URL_RE.sub(" ", text.lower()))
    # Bigrams give the match a little word-order sensitivity, which matters for
    # short chat messages where unigrams alone are nearly interchangeable.
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def _turns(messages) -> list[tuple[str, str, Optional[object]]]:
    """Collapse consecutive messages from the same sender into single turns."""
    out = []
    for m in messages:
        if out and out[-1][0] == m.sender:
            sender, text, at = out[-1]
            if len(text) < MAX_TURN_CHARS:
                out[-1] = (sender, (text + " " + m.text).strip(), at)
            continue
        out.append((m.sender, m.text.strip(), m.at))
    return out


def build_pairs(messages, me: set[str]) -> list[Pair]:
    """Every time someone spoke and you answered soon after."""
    pairs: list[Pair] = []
    turns = _turns(messages)

    for prev, curr in zip(turns, turns[1:]):
        prev_sender, prompt, prev_at = prev
        curr_sender, reply, curr_at = curr

        if curr_sender not in me or prev_sender in me:
            continue
        if not prompt or not reply or len(reply) > MAX_REPLY_CHARS:
            continue
        if len(WORD_RE.findall(prompt.lower())) < 2:
            continue  # too little to match against
        if prev_at and curr_at:
            gap = (curr_at - prev_at).total_seconds() / 60.0
            if gap < 0 or gap > MAX_GAP_MINUTES:
                continue

        pairs.append(Pair(prompt[:MAX_TURN_CHARS], reply, prev_sender))

    return pairs


class ReplyIndex:
    """BM25 over the prompts, scored as a fraction of the query's own weight.

    Plain TF-IDF cosine was tried first and is wrong for this corpus. Chat
    messages are tiny, and cosine lets a three word document win on a single
    shared term: "im so tired today" matched "Us today" at 0.31 purely because
    "today" was most of that document. BM25 scores a sum over the query terms
    instead, and dividing by the total weight of the query means matching one
    term out of five scores low no matter how short the document is.
    """

    def __init__(self, pairs: list[Pair]):
        self.pairs = pairs
        docs = [tokens(p.prompt) for p in pairs]

        self.doclen = [len(d) for d in docs]
        self.avgdl = (sum(self.doclen) / len(docs)) if docs else 1.0

        df: Counter = Counter()
        for doc in docs:
            df.update(set(doc))

        n = len(docs) or 1
        self.idf = {
            t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }
        # A word this person has never been asked before is maximally rare, and
        # must still count against the match. Leaving unknown words out of the
        # ceiling is what made "explain quantum tunnelling to me" score 0.62.
        self.max_idf = math.log(1.0 + (n + 0.5) / 0.5)

        self.index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            for token, tf in Counter(doc).items():
                self.index[token].append((doc_id, tf))

    def _weight(self, term: str) -> float:
        idf = self.idf.get(term, self.max_idf)
        return idf * (BIGRAM_WEIGHT if "_" in term else 1.0)

    def search(self, query: str, k: int = 5,
               context: list[str] | None = None) -> list[tuple[float, Pair]]:
        """Rank reply pairs against the message, carrying the conversation.

        Searching on the last message alone is what made the chat forget who it
        was talking about: "but what about when they did that" shares no content
        words with the question it follows, so it retrieved on "what about"
        alone. Earlier turns are folded into the query at a decaying weight, so
        the subject of the conversation keeps pulling.

        The ceiling is computed over the weighted terms, not the current message
        alone. An earlier attempt held the ceiling to the current message so the
        threshold would keep its old meaning, but that capped a context term's
        contribution at about 0.03 against a typical 0.21 match, which is a
        rounding error. A follow-up carries almost none of its own meaning, so
        the score has to be a fraction of the weighted conversation instead.
        """
        terms = set(query_tokens(query))
        weights: dict[str, float] = {t: 1.0 for t in terms}
        if context:
            decay = CONTEXT_DECAY
            for text in reversed(context[-CONTEXT_TURNS:]):
                for t in query_tokens(text):
                    if t not in terms:
                        weights[t] = max(weights.get(t, 0.0), decay)
                decay *= CONTEXT_DECAY

        if not weights:
            return []

        ceiling = sum(self._weight(t) * w for t, w in weights.items()) or 1.0

        scores: dict[int, float] = defaultdict(float)
        for term, term_weight in weights.items():
            postings = self.index.get(term)
            if not postings:
                continue
            weight = self._weight(term) * term_weight
            for doc_id, tf in postings:
                length_norm = 1.0 - B + B * self.doclen[doc_id] / self.avgdl
                saturated = (tf * (K1 + 1.0)) / (tf + K1 * length_norm)
                # Clamped, so one very short document cannot score more than
                # the term is worth.
                scores[doc_id] += weight * min(saturated, 1.0)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(score / ceiling, self.pairs[doc_id]) for doc_id, score in ranked]

    def best(self, query: str) -> Optional[tuple[float, Pair]]:
        hits = self.search(query, k=1)
        if hits and hits[0][0] >= MIN_SCORE:
            return hits[0]
        return None

    def stats(self) -> dict:
        return {"pairs": len(self.pairs), "terms": len(self.idf)}


if __name__ == "__main__":
    import sys
    from parser import parse_file, sender_counts

    if len(sys.argv) < 2:
        sys.exit(f"usage: python {sys.argv[0]} <export file>"
                 f"{' <sender name>' if True else ''}")
    path = sys.argv[1]
    who = sys.argv[2] if len(sys.argv) > 2 else None

    msgs = parse_file(path)
    if who is None:
        who = sender_counts(msgs)[0][0]
        print(f"no sender given, using the most active: {who}\n")
    pairs = build_pairs(msgs, {who})
    index = ReplyIndex(pairs)
    print(f"{index.stats()}\n")

    for q in ["who do you think wins the world cup",
              "did you see that goal",
              "you coming out tonight",
              "that ref is terrible"]:
        print(f"  {q!r}")
        for score, pair in index.search(q, 3):
            print(f"     {score:.2f}  {pair.who}: {pair.prompt[:44]!r}")
            print(f"            you: {pair.reply[:60]!r}")
        print()
