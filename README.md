# next-read

Find books endorsed by people whose taste you trust.

## Two modes

**By Book** — Input a book title and author. We find the blurbers on the back cover, then find every other book those blurbers have endorsed.

**By Person** — Input an endorser name (e.g., "Michael Bloomberg"). We find every book that person has blurbed.

## Architecture

Agentic pipeline with four specialized agents:

- **Blurber Finder Agent** — exhaustive multi-search to identify every real human endorser on a book
- **Endorsement Search Agent** — finds all books a person has blurbed
- **Verifier Agent** — confirms each claim with direct evidence (URL plus quote)
- **Ranker Agent** — aggregates and ranks by overlapping endorsers

Search and verification run in parallel for speed.

## Setup

1. Clone the repo

2. Install dependencies:

```
   pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your API key:

```
   cp .env.example .env
```

   Then edit `.env` and replace the placeholder with your real Anthropic API key.

## Run the web app

```
python app.py
```

Open http://localhost:7860 in your browser.

## Run from the command line

By book:

```
python next_read.py "Streetwise: Getting to and Through Goldman Sachs" "Lloyd Blankfein"
```

By person:

```
python next_read.py "Michael Bloomberg"
```

## Example

Input: Streetwise (Lloyd Blankfein)

Blurbers found: Warren Buffett, Michael Bloomberg, Jim Cramer, Barry Diller, Niall Ferguson, Ken Griffin, David Rubenstein, Liaquat Ahamed

Output: ranked list of other books endorsed by the same people, with overlapping endorsers surfaced first and a sourced quote and URL stored for each verified endorsement.