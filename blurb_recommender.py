"""
who-blurbed: Given a book, find other books endorsed by the same people who blurbed it.

Usage:
    1. Create a .env file with: ANTHROPIC_API_KEY=your_key
    2. pip install -r requirements.txt
    3. python blurb_recommender.py
"""

import json
import os
import sys
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
MODEL = "claude-sonnet-4-5"


def find_blurbers(book_title, book_author):
    """Use Claude with web search to identify blurbers on a book's back cover."""
    prompt = f"""I need to find the people who wrote endorsements (blurbs) on the back cover of this book:

Book: "{book_title}" by {book_author}

Search the web. Check Amazon, the publisher's website, Barnes & Noble, Goodreads. Find the people who provided blurbs/endorsements that appear on the book's back cover or in the marketing materials (often labeled "Praise for...", "Editorial Reviews", or similar).

Return ONLY valid JSON (no other text, no markdown code blocks) with this structure:
{{
  "book_title": "...",
  "book_author": "...",
  "blurbers": [
    {{"name": "Full Name", "affiliation": "title or organization", "quote_snippet": "first few words of their blurb"}}
  ]
}}

Include only real, verified blurbers with web evidence. If you cannot find any, return blurbers as an empty list."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    return _extract_json(text)


def find_other_endorsements(blurber_name, exclude_book):
    """For a given person, find other books they have endorsed."""
    prompt = f"""Search the web to find books that {blurber_name} has written endorsements/blurbs for.

Exclude this book: "{exclude_book}".

Focus on books from the last 10 years where there is verifiable web evidence the person provided an endorsement (their name appears on the back cover, in "Praise for" sections, in publisher marketing, etc.).

Return ONLY valid JSON (no other text, no markdown code blocks):
{{
  "endorser": "{blurber_name}",
  "books": [
    {{"title": "Book Title", "author": "Author Name", "year": 2023, "one_line": "brief description"}}
  ]
}}

Only include books where you have web evidence the person actually endorsed them. Do not include books they merely wrote, mentioned, or recommended in interviews. If you can't find any verified endorsements, return books as an empty list."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    return _extract_json(text)


def _extract_json(text):
    """Pull JSON object out of a potentially noisy LLM response."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        print(f"  [warning] JSON parse error: {e}")
        return {}


def recommend_books(book_title, book_author):
    print(f"\n{'=' * 60}")
    print(f"who-blurbed: '{book_title}' by {book_author}")
    print(f"{'=' * 60}")

    print(f"\n[1/3] Finding blurbers...")
    blurber_data = find_blurbers(book_title, book_author)
    blurbers = blurber_data.get("blurbers", [])

    if not blurbers:
        print("  No blurbers found. Try a different book or check the title spelling.")
        return

    print(f"  Found {len(blurbers)} blurber(s):")
    for b in blurbers:
        name = b.get("name", "?")
        aff = b.get("affiliation", "")
        print(f"    - {name}" + (f" ({aff})" if aff else ""))

    print(f"\n[2/3] Searching for each blurber's other endorsements...")
    book_to_endorsers = {}

    for b in blurbers:
        name = b.get("name")
        if not name:
            continue
        print(f"  Searching: {name}")
        result = find_other_endorsements(name, book_title)
        for book in result.get("books", []):
            title = book.get("title", "").strip()
            if not title:
                continue
            if title not in book_to_endorsers:
                book_to_endorsers[title] = {
                    "title": title,
                    "author": book.get("author", "unknown"),
                    "year": book.get("year"),
                    "one_line": book.get("one_line", ""),
                    "endorsers": [],
                }
            if name not in book_to_endorsers[title]["endorsers"]:
                book_to_endorsers[title]["endorsers"].append(name)

    print(f"\n[3/3] Ranked recommendations:")
    print(f"{'-' * 60}")

    if not book_to_endorsers:
        print("  No cross-referenced books found.")
        return

    ranked = sorted(
        book_to_endorsers.values(),
        key=lambda x: (-len(x["endorsers"]), -(x.get("year") or 0)),
    )

    for i, rec in enumerate(ranked, 1):
        endorsers = ", ".join(rec["endorsers"])
        year = rec.get("year") or "n/a"
        print(f"\n  {i}. {rec['title']} ({year})")
        print(f"     by {rec['author']}")
        print(f"     endorsed by: {endorsers}")
        if rec["one_line"]:
            print(f"     {rec['one_line']}")

    out_path = "recommendations.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "input_book": {"title": book_title, "author": book_author},
                "blurbers": blurbers,
                "recommendations": ranked,
            },
            f,
            indent=2,
        )
    print(f"\n  Full output saved to {out_path}")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found.")
        print("Create a .env file in this directory with:")
        print("  ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    if len(sys.argv) >= 3:
        title = sys.argv[1]
        author = sys.argv[2]
    else:
        title = "Streetwise: Getting to and Through Goldman Sachs"
        author = "Lloyd Blankfein"

    recommend_books(title, author)