"""Persona replies: retrieval first, then a local model on top if one is there.

Retrieval alone answers with something the person genuinely said, which is
authentic but only works when the incoming message resembles something they
were actually asked before. A local model alone answers anything, but in its own
generic voice.

Combining them fixes both: the retrieved pairs go into the prompt as few-shot
turns, so the model adapts real replies to the current question instead of
inventing a personality. With no model installed the retrieved reply is returned
verbatim, so the feature still works with nothing to download.
"""

import json
import re
import urllib.error
import urllib.request

import retrieval

OLLAMA = "http://127.0.0.1:11434"

# Small, fast, and comfortable on 16 GB. First one present wins.
# Best first. Chat quality matters more than latency here: a reply takes a
# second either way, and the smaller models invent details and lose the voice
# as soon as a question needs more than a sentence.
PREFERRED = (
    "qwen2.5:7b", "llama3.1:8b", "mistral:7b", "qwen2.5:3b",
    "llama3.2:3b", "phi3:mini", "gemma2:2b", "llama3.2",
)

# Smallest first. The editor asks for a completion on every pause, so it wants
# the fastest model installed, not the best one.
PREFERRED_FAST = ("gemma2:2b", "qwen2.5:3b", "llama3.2:3b", "phi3:mini")

EXAMPLE_COUNT = 5
HISTORY_TURNS = 12        # six exchanges, so pronouns still have an antecedent
CONTEXT_TURNS = 4         # how much conversation steers retrieval


def _get(path: str, timeout: float):
    req = urllib.request.Request(OLLAMA + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def status(timeout: float = 0.8) -> dict:
    """Is Ollama running, and which of its models should we use?"""
    try:
        data = _get("/api/tags", timeout)
    except Exception:
        return {"available": False, "models": [], "model": None}

    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    return {
        "available": True,
        "models": models,
        "model": pick_model(models),
        "fast_model": pick_fast_model(models),
    }


def pick_fast_model(models: list[str]) -> str | None:
    """The quickest installed model, for the inline editor."""
    for name in PREFERRED_FAST:
        for installed in models:
            if installed == name or installed.startswith(name + ":"):
                return installed
    return pick_model(models)


def pick_model(models: list[str]) -> str | None:
    for name in PREFERRED:
        for installed in models:
            if installed == name or installed.startswith(name + ":"):
                return installed
    return models[0] if models else None


def build_system_prompt(identity: str, profile: dict, stats: dict,
                        examples=()) -> str:
    """The system prompt carries the style AND the examples.

    Two earlier versions were wrong in instructive ways.

    The first listed the person's catchphrases. A 3B model reads a list in a
    prompt as a list of things to say, and parroted one back in 5 of 8 replies.
    Those lists are gone; the examples carry the voice instead.

    The second put the examples in the messages array as user and assistant
    turns, ahead of the real conversation. That made the model treat five
    unrelated exchanges as the most recent thing that happened, so a follow-up
    like "but what about when they did that" had no antecedent it trusted. The
    examples now sit here as a labelled transcript, and the messages array holds
    nothing but the actual conversation.
    """
    traits = "\n".join(f"- {t['label']}: {t['value']}" for t in profile["traits"])
    length = profile.get("length") or {"median": 5, "p90": 12}
    # Their longest normal message, with room to answer a real question, but
    # not so much room that the model writes an essay and loses the voice.
    ceiling = max(25, length["p90"] * 3)

    transcript = ""
    if examples:
        lines = []
        for _, pair in examples:
            lines.append(f"  {pair.who}: {pair.prompt}")
            lines.append(f"  {identity}: {pair.reply}")
            lines.append("")
        transcript = (
            "\nExamples of how they write, taken from other conversations. They "
            "are here to show voice and register only. Never reuse their wording "
            "or their subject matter:\n\n" + "\n".join(lines)
        )

    return f"""You are replying as {identity} in a casual group chat with friends.

Measured across {stats['messages']} of their real messages:
{traits}
{transcript}
Rules:
- You are in an ongoing conversation. Read all of it before you answer. Work out
  who "he", "she", "they" and "it" refer to from the earlier turns. The person
  talking to you will not repeat a name they have already used, so if the last
  message has no subject of its own, it is still about whatever you were both
  just discussing.
- Answer at the length the message deserves. Most of their messages run about
  {length['median']} words. A yes or no question gets a one or two word answer.
  A real question gets a real answer, usually {length['p90']} words or so, and
  at the very most {ceiling} words. Do not pad, and do not cut an answer short
  just to sound blunt.
- A longer answer is not a more formal one. It uses exactly the same words,
  slang and misspellings they use in a two word reply. If a sentence sounds
  tidy, polished or like an essay, it is wrong.
- Only talk about things that have actually come up in this conversation. Do not
  invent events, places or details to fill an answer out.
- Match their capitalisation and punctuation, and do not tidy up a spelling they
  actually used. But never invent textspeak. Do not write "wot", "da", "nite" or
  "some1" unless that exact spelling appears in the examples above. When in
  doubt, spell the word normally. Caricaturing how they type is worse than
  spelling it out.
- Always reply in English. Never switch language, even for a single word. The
  model behind this is multilingual and will occasionally drift; do not.
- Use emoji about as often as the figure above says, and only ones they use.
- Never explain yourself, never break character, never mention being a model.
- Output the reply text only. No quotation marks, no name prefix, no preamble."""


def build_messages(system: str, history, message: str) -> list[dict]:
    """System prompt, then the real conversation, and nothing else."""
    out = [{"role": "system", "content": system}]
    # reply() guards this for the retrieval context but did not here, so a
    # caller passing None reached a TypeError that the model path's except
    # clause does not catch, and the retrieval fallback was never tried.
    for turn in (history or [])[-HISTORY_TURNS:]:
        role = "assistant" if turn.get("role") == "assistant" else "user"
        text = turn.get("text", "")
        if text:
            out.append({"role": role, "content": text})
    out.append({"role": "user", "content": message})
    return out


def clean_reply(text: str, identity: str) -> str:
    """Strip the wrappers small models like to add.

    The name prefix is matched against the whole identity and each part of it,
    because a model told it is "Aaryan Kassam" will happily answer "Aaryan:".
    """
    text = text.strip()

    names = {identity, *identity.replace(" and ", " ").split()}
    alternatives = "|".join(
        re.escape(n) for n in sorted(names, key=len, reverse=True) if n
    )
    prefix = re.compile(rf"^(?:{alternatives}|assistant)\s*:\s*", re.I)

    for _ in range(2):
        stripped = prefix.sub("", text).strip()
        if len(stripped) > 1 and stripped[0] in "\"'\u201c\u2018":
            closing = {"\u201c": "\u201d", "\u2018": "\u2019"}.get(stripped[0], stripped[0])
            if stripped.endswith(closing):
                stripped = stripped[1:-1].strip()
        if stripped == text:
            break
        text = stripped

    text = re.sub(r"^\((?:as|in the style of)[^)]*\)\s*", "", text, flags=re.I)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Six rather than two: the old cap silently truncated any reply that ran to
    # more than a couple of lines, which is part of why answers looked clipped.
    return "\n".join(lines[:6]).strip()


def generate(model: str, messages: list[dict], timeout: float = 90.0) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.65, "top_p": 0.9, "num_predict": 150},
    }).encode()

    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return (data.get("message") or {}).get("content", "")


def reply(message, index, identity, profile, stats, history, use_model, model):
    """One persona reply, plus an honest account of where it came from."""
    # Retrieval sees the conversation too, so a follow-up that carries no topic
    # of its own still pulls examples about whatever is being discussed.
    recent = [t.get("text", "") for t in (history or [])[-CONTEXT_TURNS:]
              if t.get("text")]
    examples = index.search(message, EXAMPLE_COUNT, context=recent) if index else []
    shown = [
        {"prompt": p.prompt[:160], "reply": p.reply[:160],
         "who": p.who, "score": round(s, 2)}
        for s, p in examples
    ]

    if use_model and model:
        system = build_system_prompt(identity, profile, stats, examples)
        try:
            text = clean_reply(generate(model, build_messages(
                system, history, message)), identity)
            if text:
                return {"reply": text, "source": "model", "model": model,
                        "examples": shown}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            pass  # fall through to retrieval

    if examples and examples[0][0] >= retrieval.MIN_SCORE:
        score, pair = examples[0]
        return {"reply": pair.reply, "source": "retrieval",
                "match": round(score, 2), "matched_prompt": pair.prompt[:160],
                "examples": shown}

    return {"reply": "", "source": "none", "examples": shown}


# -- sentence completion for the editor -------------------------------------

COMPLETION_WORDS = 12
COMPLETION_EXAMPLES = 8


def build_completion_prompt(identity: str, profile: dict) -> str:
    """Prompt for finishing a half-typed sentence.

    The n-gram model cannot do this. Trained on ten thousand tokens, a four word
    context almost never matches, so it falls back to bigrams and emits whatever
    word usually follows the last one. That is why its completions read as
    unrelated to the sentence: nothing in the model ever saw the sentence.

    The style examples are a fixed anchor sample, not retrieved by similarity to
    what is on screen. Retrieval was tried and is worse on every axis. It is
    weakest exactly when it is needed most: once a sentence has drifted into
    neutral English the nearest real messages score around 0.1 and carry no
    voice at all, so the model carries on drifting. Measured over five seeds
    with four completions each:

        anchor 14 + nearest 4   1103 ms   81% of new words in their vocabulary
        anchor 14 alone          481 ms   88%

    A fixed sample is also the same prompt every time, so the prefix cache hits
    and the whole thing runs twice as fast. The last rule below exists for the
    same drift reason.
    """
    length = profile.get("length") or {"median": 5, "p90": 12}

    anchor = "\n".join(f"  {s}" for s in (profile.get("samples") or [])[:14])
    blocks = ""
    if anchor:
        blocks += ("\nThis is how they actually write. Every line is a real "
                   "message they sent:\n\n" + anchor + "\n")

    return f"""You finish sentences the way {identity} types them.
{blocks}
You will be given the start of a message they are typing. Reply with only the
words that come next.

Rules:
- Never repeat any of the text you were given, and never restate it.
- The continuation must be correct, natural English when joined directly onto
  the text you were given. Read the whole sentence, not just the last word, and
  make the grammar work across the join.
- Stay on the subject of the sentence. If it names a person or a thing, the
  continuation is about that person or thing.
- Do not repeat any word or phrase that already appears anywhere in the text you
  were given. If the sentence already says something, say the next thing.
- The example messages show voice, never content. Never copy a line from them,
  and never borrow their subject. If you find yourself reusing one, you have
  stopped completing the sentence on screen.
- Never write neutral, corporate or customer service English. If your
  continuation would fit in a work email it is wrong. Never write "looking
  forward to", "excited to", "dive in", "make it happen", "touch base", "reach
  out", "great to hear", "a good chat" or anything of that kind.
- If the text on screen already reads as formal or polished, do not carry on in
  that register. Pull it back towards how they write in the messages above.
- At most {COMPLETION_WORDS} words, and stop at the end of the thought. Their
  messages run about {length['median']} words.
- Match their spelling and capitalisation. Never invent textspeak.
- Output the continuation only. No quotes, no explanation, no name prefix."""


def _strip_echo(typed: str, out: str) -> str:
    """Remove any part of the typed text the model repeated back."""
    out = out.strip()
    low_typed, low_out = typed.lower(), out.lower()
    if low_out.startswith(low_typed):
        return out[len(typed):].lstrip()
    # Longest suffix of what was typed that the model restated as a prefix.
    for n in range(min(len(typed), 80), 3, -1):
        if low_out.startswith(low_typed[-n:]):
            return out[n:].lstrip()
    return out


def complete_sentence(model_name, identity, profile, text,
                      mid_word: bool, timeout: float = 20.0) -> str:
    """A continuation that can be appended to `text` verbatim."""
    system = build_completion_prompt(identity, profile)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.75, "top_p": 0.9, "num_predict": 48,
            # Each completion is a fresh generation that sees the sentence so
            # far as its prompt, and without this it will happily echo the
            # phrase it just wrote: "thats so mod ya thats mod thats so mod".
            "repeat_penalty": 1.2, "repeat_last_n": 128,
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)

    out = (data.get("message") or {}).get("content", "")
    out = clean_reply(out, identity)
    out = _strip_echo(text, out)
    out = out.split("\n")[0].strip()
    if not out:
        return ""

    words = out.split()
    if len(words) > COMPLETION_WORDS:
        out = " ".join(words[:COMPLETION_WORDS])

    # Spacing is decided here, not by the model, so the editor can append the
    # string exactly as it arrives.
    if mid_word:
        return out.lstrip()
    if text and not text[-1].isspace():
        return " " + out.lstrip()
    return out.lstrip()
