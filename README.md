# photo-ai-toolkit

[![CI](https://github.com/karasov-co/photo-ai-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/karasov-co/photo-ai-toolkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

**You came back from a trip with 2,000 frames and you have opened the folder
four times without editing anything.** This sorts them into five piles, ranks
every pile by what each photograph can *become* after a normal edit, says in one
sentence why, and writes a Lightroom starting point for the ones worth an
evening.

![The report: five piles, a score per photograph, and the edit that earns it](docs/images/report.png)

It never moves, changes or deletes an original file.

---

## Start without paying anything

```bash
pip install -r requirements.txt
python3 cli.py analyze --input ./photos --output ./run
open ./run/report/index.html
```

No API key needed. That run gives you every technical measurement, the duplicate
groups, the edit recipes and the full report. What it cannot do is look at the
picture — no content check, no artistic read, and nothing can be ranked as a top
photograph, because nothing looked at it.

For that, add a key and run the same command:

```bash
cp .env.example .env       # then put XAI_API_KEY=... in it, once
python3 cli.py analyze --input ./photos --output ./run
```

It prints the estimated cost and waits for you to type `yes`. About **$1.90 per
100 photographs** on grok-4.6 — measured from a real run's token usage,
reasoning tokens included. Every call is logged to `run/.internal/usage.jsonl`
and the real total is printed at the end, so the estimate is checkable rather
than trusted. `--reasoning low|medium|high` is the dial; low is the default.

Do not `source .env` or `export` anything — the application loads the file
itself, from the project root, whichever directory you run from. `.env` is
gitignored and never committed.

---

## What you get

`run/report/index.html` is a folder you can copy, zip or email. It has no
external requests — no CDN, no web fonts, no JavaScript libraries — so it opens
on a plane, behind a firewall, and in five years. `--standalone` produces a
single `report_standalone.html` with every thumbnail inlined, for sending to one
person.

Five piles, and the count of each at the top of the page:

| Pile | What it means |
|---|---|
| **Top** | Scores 85+. The photographs that carry the shoot. Often empty, which is normal. |
| **Good — stock** | Good, and the frame also has a market. |
| **Good — personal** | Exactly as good. No market, which is not a fault. |
| **Needs decision** | Something is genuinely unclear. You decide, not the tool. |
| **Weak** | A shelf, not a bin. Nothing is deleted. |

Each card carries three numbers, and they measure different things:

- **final score after editing** — the big one, what the photograph becomes
- **technical quality** — how clean the file is right now
- **content** — the moment, the composition, the subject (present only when the
  content check ran)

A pristine picture of nothing scores high on the second and low on the third.
That pair disagreeing is the point of showing both.

---

## Nothing moves unless you say so

The output is a farm of symlinks pointing back at wherever your files already
live, plus a CSV of the same decisions as data. Deletion is a two-step ritual:
the tool writes a list, a contact sheet of the frames on that list, and a script
that moves them to the Trash. It never removes anything itself, and `--apply` is
the only thing that carries out a plan.

---

## Read next

- **[How the score is built](docs/how-it-works.md)** — potential versus current
  quality, the five piles, the artistic read, what a card is telling you
- **[Commands, output and configuration](docs/reference.md)** — every flag, the
  output tree, calibration profiles, marketplaces, metadata
- **[Internals, safety and limits](docs/internals.md)** — architecture, the
  model and why there is no fallback, quarantine, development, known limits

---

## Install

Python 3.12+. FFmpeg is optional and only needed for video.

```bash
pip install -r requirements.txt
brew install ffmpeg          # macOS; apt-get install ffmpeg on Debian/Ubuntu
```

The key is looked for in this order: a variable already set in your shell, then
`.env` in the project root. `XAI_API_KEY` first, `OPENAI_API_KEY` as a fallback,
because most people already have one of those sitting in a file somewhere.

## Disclaimer

A score is one opinion about a photograph, produced by measurements and a
language model, and both are wrong sometimes. Nothing here is a verdict. The
tool is built so that every suggestion is reversible and every original file is
untouched — look at the pictures before you act on any of it.

## Contributing

Issues and pull requests welcome. `ruff check . && pytest -q` must pass.

## License

MIT. See [LICENSE](LICENSE).
