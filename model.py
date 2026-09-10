"""A style model of one person's writing.

Two things it does that a general purpose Markov library does not:

1. Scores the next word against the *whole* recent context, not just the last
   word. Contexts of length 1 to 4 are blended by Witten-Bell interpolation, so
   a four word context that has been seen dominates, and one that has not
   degrades smoothly toward shorter contexts instead of falling off a cliff.

2. Filters that distribution by the prefix of a half-typed word, which is what
   the editor needs while you are mid-word.

Emoji are stripped before training. They carry no next-word information, they
break the n-gram chain wherever they appear mid-sentence, and suggesting one is
never useful.
"""

import re
import time
from collections import Counter, defaultdict

START = "<s>"
END = "</s>"
ORDER = 4          # longest context, in tokens
DISCOUNT = 2.0     # Witten-Bell discount; higher trusts rare contexts less
MIN_MESSAGES = 300

TOKEN_RE = re.compile(r"[A-Za-z0-9_'’]+|[^\sA-Za-z0-9_]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

CLOSING = set(".,!?;:)]}%’'\"…")
OPENING = set("([{$#@")

# Blocks that hold emoji, pictographs, dingbats, variation selectors, flags
# and the zero width joiner that stitches compound emoji together.
EMOJI_RANGES = (
    (0x1F000, 0x1FAFF), (0x1F1E6, 0x1F1FF), (0x2600, 0x27BF),
    (0x2B00, 0x2BFF), (0x2190, 0x21FF), (0x2300, 0x23FF),
    (0x25A0, 0x25FF), (0xFE00, 0xFE0F), (0x200D, 0x200D),
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "them", "my", "your", "his", "our", "their", "as",
    "so", "do", "did", "does", "not", "no", "yes", "up", "out", "then",
}

LAUGHS = ("lol", "lmao", "lmfao", "haha", "hahaha", "ahaha", "rofl", "hehe",
          "lmaoo", "loll")


def is_emoji(token: str) -> bool:
    return any(
        any(lo <= ord(ch) <= hi for lo, hi in EMOJI_RANGES) for ch in token
    )


def tokenize(text: str) -> list[str]:
    """All tokens, emoji included. Used for measuring style."""
    return TOKEN_RE.findall(URL_RE.sub(" ", text))


def train_tokens(text: str) -> list[str]:
    """Tokens used to train the predictor, with emoji removed."""
    return [t for t in tokenize(text) if not is_emoji(t)]


def detokenize(tokens: list[str]) -> str:
    out = ""
    for i, tok in enumerate(tokens):
        if i == 0:
            out = tok
        elif tok in CLOSING or (len(tok) == 1 and tok in ".,!?;:"):
            out += tok
        elif out and out[-1] in OPENING:
            out += tok
        else:
            out += " " + tok
    return out


class StyleModel:
    def __init__(self, messages: list[str], others: list[str] | None = None):
        t0 = time.perf_counter()

        # ctx[n] maps an n-token context to the distribution of what follows.
        self.ctx: dict[int, dict[tuple, Counter]] = {
            n: defaultdict(Counter) for n in range(1, ORDER + 1)
        }
        self.unigram: Counter = Counter()
        surface: dict[str, Counter] = defaultdict(Counter)

        self.message_count = len(messages)
        self.token_count = 0
        train_lists: list[list[str]] = []

        for msg in messages:
            raw = train_tokens(msg)
            if not raw:
                continue
            for tok in raw:
                surface[tok.lower()][tok] += 1
            lowered = [t.lower() for t in raw]
            train_lists.append(lowered)
            self.token_count += len(raw)

            padded = [START] * ORDER + lowered + [END]
            for i in range(ORDER, len(padded)):
                nxt = padded[i]
                self.unigram[nxt] += 1
                for n in range(1, ORDER + 1):
                    self.ctx[n][tuple(padded[i - n:i])][nxt] += 1

        self.surface = {k: c.most_common(1)[0][0] for k, c in surface.items()}
        self.vocab_size = len(self.unigram)

        total = sum(self.unigram.values()) or 1
        self.uni_p = {w: c / total for w, c in self.unigram.items()}
        # Always-considered fallback candidates, so a cold context still ranks.
        self.top_unigrams = [w for w, _ in self.unigram.most_common(60)]

        self.profile = self._build_profile(messages, train_lists, others or [])
        self.build_seconds = time.perf_counter() - t0

    # -- distribution ----------------------------------------------------

    def _dist(self, context: list[str], prefix: str = "") -> dict[str, float]:
        """P(next word) given the recent context, blended across orders.

        Starts from the unigram distribution and folds in each longer context
        in turn. The weight given to a context is N / (N + d * U), where N is
        how often it was seen and U how many different words followed it, so a
        context seen once with one continuation is trusted far less than one
        seen thirty times.
        """
        # Pad with START so the beginning of a message is a real context and
        # not an empty one that collapses to bare word frequencies.
        padded = [START] * ORDER + context
        keys = [(n, tuple(padded[-n:])) for n in range(1, ORDER + 1)]

        candidates = set(self.top_unigrams)
        if prefix:
            # The candidate set is otherwise capped at the commonest words, so
            # a rarer word that completes what is being typed was invisible:
            # "tues" found nothing and was judged a finished word.
            candidates.update(w for w in self.unigram if w.startswith(prefix))
        matched = []
        for n, key in keys:
            counter = self.ctx[n].get(key)
            if counter:
                matched.append(counter)
                candidates.update(counter)

        probs = {w: self.uni_p.get(w, 0.0) for w in candidates}
        for counter in matched:  # ascending context length
            n_total = sum(counter.values())
            n_distinct = len(counter)
            lam = n_total / (n_total + DISCOUNT * n_distinct)
            for w in probs:
                probs[w] *= 1.0 - lam
            for w, c in counter.items():
                probs[w] += lam * c / n_total
        return probs

    def _display(self, token: str) -> str:
        return self.surface.get(token, token)

    def _split(self, text: str):
        # train_tokens drops emoji, so "that was mad 😂" tokenises to
        # ["that","was","mad"] and the last character is not whitespace. Without
        # the emoji test the finished word "mad" is treated as half typed and
        # the editor offers to extend it.
        trailing = text[-1] if text else ""
        tokens = [t.lower() for t in train_tokens(text)]
        if trailing and not trailing.isspace() and not is_emoji(trailing) and tokens:
            return tokens[:-1], tokens[-1]
        return tokens, ""

    def _resolve(self, text: str):
        """Split text into (context, prefix), deciding if the last word is done.

        "wh" is clearly half typed. "a" in "such a" is clearly finished, even
        though "and" would extend it. The two cases are told apart by asking
        which the model finds more likely in this context: the word standing on
        its own, or the best word that extends it.
        """
        context, prefix = self._split(text)
        if not prefix:
            return context, ""

        dist = self._dist(context, prefix)
        best_extension = 0.0
        for word, p in dist.items():
            if word in (START, END) or is_emoji(word):
                continue
            if word.startswith(prefix) and word != prefix:
                best_extension = max(best_extension, p)

        if dist.get(prefix, 0.0) >= best_extension:
            return context + [prefix], ""
        return context, prefix

    def _ranked(self, context: list[str], prefix: str, k: int) -> list[str]:
        scored = sorted(self._dist(context, prefix).items(), key=lambda kv: -kv[1])
        out = []
        for tok, _ in scored:
            if tok in (END, START) or is_emoji(tok):
                continue
            # Punctuation is a real next token but a useless thing to offer as
            # a word to click, and these are now the editor's only suggestions.
            if not tok[0].isalnum():
                continue
            if prefix and (not tok.startswith(prefix) or tok == prefix):
                continue
            out.append(tok)
            if len(out) == k:
                break
        return out

    # -- public API ------------------------------------------------------

    def next_words(self, text: str, k: int = 3) -> list[str]:
        return self.suggest(text, k)["words"]

    def suggest(self, text: str, k: int = 3) -> dict:
        """Suggestions plus how the caller should apply them.

        mode is "extend" when the words complete the half-typed word at the
        caret, and "append" when they are the next word after a finished one.
        """
        context, prefix = self._resolve(text)
        return {
            "words": [self._display(t) for t in self._ranked(context, prefix, k)],
            "mode": "extend" if prefix else "append",
        }

    def complete(self, text: str, max_words: int = 8) -> str:
        """Ghost-text continuation, greedy so it does not flicker."""
        context, prefix = self._resolve(text)

        generated: list[str] = []
        ctx = list(context)
        used = Counter()

        for step in range(max_words):
            if step == 0 and prefix:
                picks = self._ranked(ctx, prefix, 1)
                choice = picks[0] if picks else None
            else:
                choice = None
                ordered = sorted(self._dist(ctx).items(), key=lambda kv: -kv[1])
                for tok, _ in ordered[:20]:
                    if tok == START or is_emoji(tok):
                        continue
                    if tok == END:
                        if step == 0:
                            continue  # never return an empty ghost
                        choice = END
                        break
                    if generated and tok == generated[-1]:
                        continue
                    if len(generated) >= 2 and tok == generated[-2]:
                        continue
                    if used[tok] >= 2:
                        continue
                    choice = tok
                    break

            if choice is None or choice == END:
                break
            generated.append(choice)
            used[choice] += 1
            ctx.append(choice)

        if not generated:
            return ""

        if prefix:
            head = generated[0][len(prefix):]
            rest = detokenize([self._display(t) for t in generated[1:]])
            return head + ((" " + rest) if rest else "")

        out = detokenize([self._display(t) for t in generated])
        # Appended verbatim at the caret, so carry the separating space.
        if text and not text[-1].isspace() and out[:1] not in CLOSING:
            out = " " + out
        return out

    # -- style profile ---------------------------------------------------

    def _build_profile(self, messages, train_lists, others) -> dict:
        # Computed first because the sample chooser scores messages by how many
        # of these they contain.
        self.profile_words_cache = self._favourite_words(train_lists, others)
        return {
            "sayings": self._sayings(train_lists),
            "words": self.profile_words_cache,
            "traits": self._traits(messages),
            "length": self._lengths(messages),
            "samples": self._samples(messages, train_lists),
        }

    def _samples(self, messages, train_lists) -> list[str]:
        """A fixed set of their most characteristic messages.

        Style examples for the editor used to be retrieved by similarity to
        whatever was typed. That fails exactly when it matters: once a sentence
        has drifted into neutral English, the nearest real messages are things
        like "Thanks boss", which score 0.09 and push back on nothing. These are
        chosen once from the corpus and always sent, so there is a floor on how
        much voice the prompt carries no matter what is on screen.
        """
        distinctive = {w["word"].lower() for w in self.profile_words_cache}
        scored = []
        for text in messages:
            words = [t for t in tokenize(text) if t[0].isalnum()]
            if not (4 <= len(words) <= 18):
                continue
            if any(is_emoji(t) for t in tokenize(text)[:1]):
                continue
            hits = sum(1 for w in words if w.lower() in distinctive)
            scored.append((hits, len(words), text))

        scored.sort(key=lambda s: (-s[0], -s[1]))
        out, seen = [], set()
        for _, _, text in scored:
            key = text.lower()[:40]
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) == 14:
                break
        return out

    @staticmethod
    def _lengths(messages) -> dict:
        """Typical and stretched message length.

        The mean alone is a bad instruction to give a model. Chat is mostly
        one word acknowledgements, so a mean of five tells it to answer every
        question in five words. The upper percentile is what says "they do
        write properly when they have something to say".
        """
        counts = sorted(
            len([t for t in tokenize(m) if t[0].isalnum()]) for m in messages
        )
        counts = [c for c in counts if c]
        if not counts:
            return {"median": 5, "p90": 12, "max": 20}
        pick = lambda q: counts[min(len(counts) - 1, int(len(counts) * q))]
        return {"median": pick(0.5), "p90": pick(0.9), "max": pick(0.99)}

    def _sayings(self, train_lists) -> list[dict]:
        """Repeated multi-word phrases, longest and most frequent first."""
        counts: Counter = Counter()
        for toks in train_lists:
            words = [t for t in toks if t[0].isalnum()]
            for n in range(2, 6):
                for i in range(len(words) - n + 1):
                    counts[tuple(words[i:i + n])] += 1

        ranked = sorted(
            (p for p, c in counts.items() if c >= 3),
            key=lambda p: (-counts[p] * (len(p) - 1) ** 2, -len(p)),
        )

        chosen: list[tuple] = []
        for phrase in ranked:
            if all(w in STOPWORDS for w in phrase):
                continue
            # A phrase is a window slid over a repeated sentence, so it can end
            # mid-thought: "first half but not the". Requiring both ends to be
            # real words keeps only the ones that read as something a person
            # would actually say.
            if phrase[0] in STOPWORDS or phrase[-1] in STOPWORDS:
                continue
            if not (phrase[0][0].isalnum() and phrase[-1][0].isalnum()):
                continue
            # Containment is not enough. Sliding a window over one repeated
            # sentence yields phrases that merely overlap, so "who booked the
            # pitch this" and "booked the pitch this week" both survive a
            # containment test and read as two separate catch phrases.
            if any(_overlaps(kept, phrase) for kept in chosen):
                continue
            chosen.append(phrase)
            if len(chosen) == 6:
                break

        return [
            {"text": detokenize([self._display(w) for w in p]),
             "count": counts[p]}
            for p in chosen
        ]

    def _favourite_words(self, train_lists, others) -> list[dict]:
        """Words used far more by this person than by everyone else in the chat.

        The rest of the chat is the baseline, which is better than a generic
        word frequency list: it surfaces what is distinctive about this person
        among these specific people, not just what is rare in English.
        """
        mine: Counter = Counter()
        for toks in train_lists:
            for t in toks:
                if t[0].isalnum() and len(t) > 1:
                    mine[t] += 1

        theirs: Counter = Counter()
        for msg in others:
            for t in train_tokens(msg):
                t = t.lower()
                if t[0].isalnum() and len(t) > 1:
                    theirs[t] += 1

        mine_total = sum(mine.values()) or 1
        their_total = sum(theirs.values()) or 1

        scored = []
        for word, count in mine.items():
            if count < 4:
                continue
            mine_rate = count / mine_total
            their_rate = (theirs.get(word, 0) + 0.5) / their_total
            scored.append((mine_rate / their_rate, word, count))

        scored.sort(reverse=True)
        return [
            {"word": self._display(w), "count": c, "ratio": round(r, 1)}
            for r, w, c in scored[:8]
        ]

    def _traits(self, messages) -> list[dict]:
        """Measurable habits, each with the number that backs it."""
        if not messages:
            return []

        lengths, questions, shouted, emoji_msgs = [], 0, 0, 0
        emoji_used: Counter = Counter()
        laughs: Counter = Counter()
        openers: Counter = Counter()
        ends_punctuated = 0

        for msg in messages:
            toks = tokenize(msg)
            words = [t for t in toks if t[0].isalnum()]
            if not words:
                continue
            lengths.append(len(words))
            if msg.rstrip().endswith("?"):
                questions += 1
            if msg.rstrip()[-1:] in ".!?":
                ends_punctuated += 1
            shouted += sum(1 for w in words if len(w) > 1 and w.isupper())
            found = [t for t in toks if is_emoji(t)]
            if found:
                emoji_msgs += 1
                emoji_used.update(found)
            for w in words:
                lw = w.lower()
                if lw in LAUGHS or re.fullmatch(r"(?:ha){2,}|l+o+l+", lw):
                    laughs[lw] += 1
            openers[words[0].lower()] += 1

        n = len(lengths) or 1
        total_words = sum(lengths) or 1

        traits = [
            {"label": "words per message", "value": f"{total_words / n:.1f}"},
            {"label": "questions", "value": f"{100 * questions / n:.0f}%"},
            {"label": "ends with punctuation", "value": f"{100 * ends_punctuated / n:.0f}%"},
            {"label": "words in caps", "value": f"{100 * shouted / total_words:.1f}%"},
            {"label": "messages with emoji", "value": f"{100 * emoji_msgs / n:.0f}%"},
        ]
        if emoji_used:
            traits.append({"label": "most used emoji",
                           "value": emoji_used.most_common(1)[0][0]})
        if laughs and laughs.most_common(1)[0][1] >= 3:
            top = laughs.most_common(1)[0]
            traits.append({"label": "laughs as", "value": f"{top[0]} ({top[1]}x)"})
        if openers:
            top = openers.most_common(1)[0]
            traits.append({"label": "usually opens with", "value": top[0]})
        return traits

    def stats(self) -> dict:
        return {
            "messages": self.message_count,
            "tokens": self.token_count,
            "vocab": self.vocab_size,
            "build_seconds": round(self.build_seconds, 3),
            "enough_data": self.message_count >= MIN_MESSAGES,
            "min_messages": MIN_MESSAGES,
        }


def _contains(haystack: tuple, needle: tuple) -> bool:
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def _overlaps(a: tuple, b: tuple) -> bool:
    """True if two phrases are substantially the same run of words."""
    if _contains(a, b) or _contains(b, a):
        return True
    shortest = min(len(a), len(b))
    shared = len(set(a) & set(b))
    return shared * 2 > shortest


if __name__ == "__main__":
    import sys
    from parser import messages_for, parse_file, sender_counts

    if len(sys.argv) < 2:
        sys.exit(f"usage: python {sys.argv[0]} <export file>"
                 f"{' <sender name>' if True else ''}")
    path = sys.argv[1]
    who = sys.argv[2] if len(sys.argv) > 2 else None

    msgs = parse_file(path)
    if who is None:
        who = sender_counts(msgs)[0][0]
        print(f"no sender given, using the most active: {who}\n")
    mine = messages_for(msgs, who)
    others = [m.text for m in msgs if m.sender != who]
    model = StyleModel(mine, others)

    print(f"{who}: {model.stats()}\n")
    print("sayings :", [s["text"] + f" ({s['count']}x)" for s in model.profile["sayings"]])
    print("words   :", [w["word"] for w in model.profile["words"]])
    print("traits  :", [f"{t['label']}={t['value']}" for t in model.profile["traits"]])
    print()
    for seed in ["i", "wh", "im gonna", "that was", "do you think we",
                 "bro that was such a", "i cant believe he"]:
        print(f"  {seed!r:28} -> {(seed + model.complete(seed))!r}")
