# TTYS

Talk To Yourself System.

A predictive keyboard and chat partner built from your own message history.
Add a chat export, pick which sender is you, and TTYS models how you write, then
suggests your next word as you type and answers as you would.

Everything runs on your machine. No API key, no account, no upload.

```
you type   ill book the            you type   tues
suggests   pitch  whole  dark      suggests   tuesday

you say    who's sorting the pitch then
replies    ill book the pitch, same place 7pm
```

## Privacy

This reads private conversations, most of which involve other people, so the
design assumes that data must never leave the machine it was exported on.

- **Nothing is written to disk.** Uploads are read straight into memory, parsed,
  and dropped when the process exits or you press Start over. There is no
  database, no cache and no session file.
- **Nothing is logged.** Message text never reaches a log line.
- **One network destination exists in the whole codebase**, `127.0.0.1:11434`,
  which is the local model running on your own machine. There is no cloud
  fallback, deliberately, so there is no setting to leave switched on by
  mistake.
- **No chat data is in this repository.** Exports are gitignored, and every
  example in this README and in the app comes from `samples/sample_chat.txt`, a
  fabricated conversation between four invented people.

Cloning this repository gives you the code and the fictional sample. It gives
you nothing about anyone's real conversations.

## Try it

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python app.py
```

Open http://127.0.0.1:5001 and add `samples/sample_chat.txt`, then pick
Sam Okafor. To use your own history, export a chat and add that instead.

Next-word suggestions need nothing but Python. Chat replies read better with a
local model, but without one the app still answers using your own real replies,
retrieved verbatim.

```bash
curl -Lo ollama.tgz https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.tgz
mkdir -p ~/.local/ollama && tar -xzf ollama.tgz -C ~/.local/ollama
~/.local/ollama/ollama serve &
~/.local/ollama/ollama pull llama3.2:3b
```

## Supported exports

| App | File | Source |
|---|---|---|
| WhatsApp | `.txt` | Export Chat, Without Media. iOS and Android dialects |
| iMessage | `.html` | imessage-exporter |
| Telegram Desktop | `.html` | Export chat history, format HTML |
| Instagram, Facebook Messenger | `.html` | Download your information, format HTML |
| Discord | `.html` | DiscordChatExporter |

Up to five at a time, and they can be mixed. Senders aggregate across all of
them, so one person appearing under different labels in different exports can be
ticked and merged. Nothing is merged automatically by fuzzy name matching.

## How it works

**Parsing** is the unglamorous half and the half that quietly corrupts a corpus.
Real exports are full of traps, and each of these is handled explicitly because
each one was found in a real file:

- WhatsApp puts a narrow no-break space before AM/PM, and a left-to-right mark
  in front of system lines and sometimes in front of the opening bracket, so
  testing whether a line starts with `[` is not a valid way to find messages.
- Multi-line messages continue on lines that carry no timestamp at all.
- iMessage emits threaded replies twice, once in place and once quoted inside
  the message they answer. In one 9,600 message export, 1,095 of the 1,097
  nested copies were exact duplicates.
- iMessage tapbacks are markup siblings of the message text. Counted naively,
  5,000 reactions enter the corpus as sentences.
- An edited iMessage keeps its entire history in a table with no bubble element,
  so reading only bubbles drops those messages and reading every cell trains on
  both the typo and the correction. Only the final version is what was said.
- Meta writes UTF-8 bytes reinterpreted as Latin-1, so every emoji arrives as a
  run of accented characters and has to be round tripped back.

**Style modelling** blends contexts of one to four words by Witten-Bell
interpolation. A four word context that has been seen dominates; one that has
not degrades toward shorter contexts instead of falling off a cliff. Whether a
trailing word is half typed or finished is decided by comparing its own
probability against the best word that would extend it, which is what stops
`such a` completing to `such and`. Emoji are stripped before training, since
they carry no next-word information and break the chain wherever they appear.

**Retrieval** turns the export back into what it really is: a record of what
people said to you and what you said back. Those pairs are indexed with BM25
over unigrams and bigrams. Plain TF-IDF cosine was tried first and is wrong for
this corpus, because chat messages are tiny and cosine lets a three word
document win on one shared term. Queries drop stopwords before scoring, since an
unseen bigram of two of them was otherwise treated as maximally informative,
drowning out the one real word a follow-up had to offer.

**Chat** puts the five closest real exchanges into the prompt as a labelled
transcript, under a system prompt built from measured style, and a local model
adapts them to the question. With no model installed the closest real reply is
returned verbatim. Conversation history is carried into both the prompt and the
retrieval query, so a follow-up like "who's sorting it then" still knows what
"it" refers to.

**The editor** offers the three likeliest next words above the cursor, from the
n-gram, in under a millisecond on every keystroke. Half typed words are extended
instead: `tues` becomes `tuesday`. Whole sentence ghost text is implemented and
turned off, behind `GHOST_TEXT` in `app.py`. The n-gram alone writes completions
that ignore the sentence it is completing, and the version that reads properly
needs a local model, so shipping it on by default means most people meet the bad
one first.

One finding worth recording, because it was counterintuitive: **listing
someone's catchphrases in a prompt makes a small model recite them.** An early
version of the chat prompt included "phrases they repeat", and it parroted one
back in 5 of 8 replies regardless of the question. The examples carry the voice
on their own; the lists are gone.

## Numbers

Measured on a real 972 KB WhatsApp export and a 8 MB iMessage export.

| | |
|---|---|
| Messages parsed | 13,875 and 9,639 |
| Duplicate threaded replies removed | 1,097 |
| Timestamps parsed | 100% of both |
| Style model build | 0.15 s |
| Next-word suggestion | under 1 ms |
| Chat reply | 0.4 s |
| Retrieval relevant vs unrelated | 18/18 on a hand built probe set |

## Limitations

- The model reproduces how you actually write, including profanity, slang and
  anything else in the source. It has no filter.
- A model trained on one group chat answers everything in that group's register.
  Adding varied exports widens it.
- Catch phrase extraction slides a window over repeated sentences, so on a
  corpus with many identical messages it can surface fragments rather than whole
  phrases.
- Whole sentence ghost text is off. It needs a local model to be worth using,
  and the n-gram fallback is not good enough to ship in its place.
- Chat quality depends on the local model. `qwen2.5:7b` holds a voice across a
  long answer where `llama3.2:3b` invents details.
