"""TTYS - Talk To Yourself System. Phase 1: the predictive editor.

Single Flask process, no database, no external API. Uploaded exports are parsed
in memory and never written to disk.
"""

from flask import Flask, jsonify, render_template, request

import html_parser
import persona
from model import MIN_MESSAGES, StyleModel
from parser import parse_text, sender_counts
from retrieval import ReplyIndex, build_pairs

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB across all files

MAX_FILES = 5
TEXT_EXTS = (".txt",)
HTML_EXTS = (".html", ".htm")

# Local single-user app, so process memory is the whole session store.
# Nothing here is persisted; restarting the server clears it.
STATE: dict = {
    "files": [], "model": None, "identity": None, "index": None,
}


def all_messages():
    return [m for f in STATE["files"] for m in f["messages"]]


def sender_payload():
    """Senders aggregated across every uploaded file, most talkative first."""
    messages = all_messages()
    per_file = {
        f["name"]: {m.sender for m in f["messages"]} for f in STATE["files"]
    }
    formats = {f["name"]: f.get("format", "whatsapp") for f in STATE["files"]}
    out = []
    for name, count in sender_counts(messages):
        out.append({
            "name": name,
            "count": count,
            "enough": count >= MIN_MESSAGES,
            "files": [fn for fn, senders in per_file.items() if name in senders],
        })
    return {
        "files": [{"name": f["name"], "count": len(f["messages"]),
                   "format": f.get("format", "whatsapp")}
                  for f in STATE["files"]],
        "total": len(messages),
        "max_files": MAX_FILES,
        "min_messages": MIN_MESSAGES,
        "senders": out,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/privacy")
def privacy():
    return render_template("privacy.html")


@app.get("/terms")
def terms():
    return render_template("terms.html")


@app.post("/api/upload")
def upload():
    incoming = request.files.getlist("files")
    if not incoming:
        return jsonify(error="No file was included in the request."), 400

    room = MAX_FILES - len(STATE["files"])
    if room <= 0:
        return jsonify(error=f"You can add at most {MAX_FILES} exports."), 400
    if len(incoming) > room:
        return jsonify(
            error=f"Only {room} more export{'s' if room > 1 else ''} can be "
                  f"added, for a maximum of {MAX_FILES}."
        ), 400

    existing = {f["name"] for f in STATE["files"]}
    added = []
    # Parsed files are held here until every file in the batch has been
    # accepted. Appending to STATE inside the loop meant a batch whose third
    # file was rejected still left the first two loaded, while the client saw
    # only the error and never refreshed its view.
    parsed = []
    for file in incoming:
        if not file.filename:
            continue
        lower = file.filename.lower()
        if not lower.endswith(TEXT_EXTS + HTML_EXTS):
            return jsonify(
                error=f"{file.filename} is not a .txt or .html file."
            ), 400
        if file.filename in existing:
            return jsonify(error=f"{file.filename} has already been added."), 400

        raw = file.read().decode("utf-8", errors="replace")
        if lower.endswith(HTML_EXTS):
            messages, kind = html_parser.parse_html(raw)
        else:
            messages, kind = parse_text(raw), "whatsapp"

        if not messages:
            return jsonify(
                error=f"No messages were found in {file.filename}. Supported "
                      "exports are WhatsApp .txt, and .html from iMessage, "
                      "Telegram Desktop, Instagram, Facebook Messenger or "
                      "Discord."
            ), 400
        parsed.append(
            {"name": file.filename, "messages": messages, "format": kind}
        )
        existing.add(file.filename)
        added.append(file.filename)

    if not added:
        return jsonify(error="No usable file was included."), 400

    STATE["files"].extend(parsed)

    STATE["model"] = None
    STATE["identity"] = None
    return jsonify(added=added, **sender_payload())


@app.post("/api/remove")
def remove():
    name = (request.json or {}).get("name", "")
    STATE["files"] = [f for f in STATE["files"] if f["name"] != name]
    STATE["model"] = None
    STATE["identity"] = None
    return jsonify(**sender_payload())


@app.post("/api/identity")
def identity():
    if not STATE["files"]:
        return jsonify(error="Add an export first."), 409

    names = (request.json or {}).get("names") or []
    if not names:
        return jsonify(error="Select at least one name."), 400

    chosen = set(names)
    messages = all_messages()
    mine = [m.text for m in messages if m.sender in chosen]
    others = [m.text for m in messages if m.sender not in chosen]
    if not mine:
        return jsonify(error="No messages found for that selection."), 400

    model = StyleModel(mine, others)

    # Reply pairs are built per file so a conversation never runs across an
    # export boundary and pairs a stranger's message with an unrelated reply.
    pairs = []
    for f in STATE["files"]:
        pairs.extend(build_pairs(f["messages"], chosen))
    index = ReplyIndex(pairs)

    STATE["model"] = model
    STATE["identity"] = sorted(chosen)
    STATE["index"] = index

    return jsonify(
        identity=sorted(chosen),
        stats=model.stats(),
        profile=model.profile,
        retrieval=index.stats(),
        ollama=persona.status(),
    )


@app.post("/api/predict")
def predict():
    model: StyleModel = STATE["model"]
    if model is None:
        return jsonify(error="No model has been built yet."), 409

    payload = request.json or {}
    text = payload.get("text", "")
    at_end = bool(payload.get("at_end", True))
    suggestion = model.suggest(text, k=3)
    return jsonify(
        words=suggestion["words"],
        mode=suggestion["mode"],
        completion=model.complete(text, max_words=8) if at_end else "",
    )


@app.get("/api/ollama")
def ollama_status():
    """Checked live, so installing Ollama mid-session is picked up on refresh."""
    return jsonify(**persona.status())


@app.post("/api/chat")
def chat():
    if STATE["model"] is None or STATE["index"] is None:
        return jsonify(error="Build a model first."), 409

    payload = request.json or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify(error="Empty message."), 400

    use_model = bool(payload.get("use_model", True))
    model_name = payload.get("model") or None
    if use_model and not model_name:
        model_name = persona.status().get("model")

    identity = " and ".join(STATE["identity"])
    result = persona.reply(
        message=message,
        index=STATE["index"],
        identity=identity,
        profile=STATE["model"].profile,
        stats=STATE["model"].stats(),
        history=payload.get("history") or [],
        use_model=use_model,
        model=model_name,
    )
    return jsonify(**result)


@app.post("/api/complete")
def complete():
    """Ghost text for the editor, written by the local model when there is one.

    Kept separate from /api/predict so the next-word chips stay instant. The
    chips come from the n-gram in under a millisecond; this runs only when
    typing pauses.
    """
    style: StyleModel = STATE["model"]
    if style is None:
        return jsonify(error="No model has been built yet."), 409

    payload = request.json or {}
    text = payload.get("text", "")
    if not text.strip():
        return jsonify(completion="", source="none")

    model_name = payload.get("model") or persona.status().get("fast_model")
    if not model_name:
        return jsonify(completion="", source="none")

    # Mid-word is the n-gram's strong suit and the model's weakest: asked to
    # finish "bro sam is so mo" it ignores the partial word and starts a new
    # phrase, giving "mojust chill and watch the game". The n-gram extends a
    # prefix correctly from its count tables, so it keeps that case.
    _, prefix = style._resolve(text)
    if prefix:
        return jsonify(completion="", source="midword")

    try:
        completion = persona.complete_sentence(
            model_name=model_name,
            identity=" and ".join(STATE["identity"]),
            profile=style.profile,
            text=text,
            mid_word=bool(prefix),
        )
    except Exception:
        return jsonify(completion="", source="none")

    return jsonify(completion=completion, source="model", model=model_name)


@app.post("/api/reset")
def reset():
    STATE.update(files=[], model=None, identity=None, index=None)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
