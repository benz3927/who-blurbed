# who-blurbed

Given a book, find other books endorsed by the same people who blurbed it.

The premise: book endorsements are a curation signal. If a few people you trust
all endorsed Book A, the other books they each endorsed are probably worth a look.

## How it works

1. Input: book title and author
2. Web search to identify the blurbers on the back cover
3. For each blurber, search the web for other books they have endorsed
4. Aggregate and rank by overlapping endorsers (books endorsed by 2+ of the same
   people surface to the top)

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

4. Run it:

```
   python blurb_recommender.py
```

## Custom input

```
python blurb_recommender.py "Book Title" "Author Name"
```

## Example

Input: Streetwise: Getting to and Through Goldman Sachs (Lloyd Blankfein)

Blurbers: Michael Bloomberg, Ken Griffin, Niall Ferguson

Output: ranked list of other books endorsed by the same people, with overlapping
endorsers surfaced first.

## Status

Prototype.