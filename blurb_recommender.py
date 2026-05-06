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

Include only real, verified blurbers with web evidence. The blurber should be a specific named individual (not an organization or publication). If you cannot find any, return blurbers as an empty list."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    return _extract_json(text)


def find_other_endorsements(blurber_name, exclude_book):
    """For a given person, find candidate books they have endorsed."""
    prompt = f"""Search the web to find books that {blurber_name} has personally written endorsements/blurbs for.

Exclude this book: "{exclude_book}".

CRITICAL DISAMBIGUATION:
- Only include books where {blurber_name} (the specific individual person) wrote a personal endorsement
- Do NOT include books simply because the person works at a related company or publication
- Do NOT include books published BY a company associated with this person
- Do NOT include books reviewed by their employer's publication
- The endorsement must be a personal blurb attributed to this individual by name

Focus on books from the last 10 years where there is verifiable web evidence.

Return ONLY valid JSON (no other text, no markdown code blocks):
{{
  "endorser": "{blurber_name}",
  "books": [
    {{"title": "Book Title", "author": "Author Name", "year": 2023, "one_line": "brief description"}}
  ]
}}

If you can't find verified personal endorsements, return books as an empty list."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    return _extract_json(text)


def verify_endorsement(blurber_name, book_title, book_author):
    """Double-check whether a person actually endorsed a specific book.
    
    Returns dict with keys: verified (bool), evidence_url (str), quote (str), reason (str).
    """
    prompt = f"""I need to verify whether {blurber_name} (the specific individual) personally wrote a blurb or endorsement for this book:

Book: "{book_title}" by {book_author}

Search the web carefully. Look for direct evidence such as:
- The blurb appearing on the book's back cover, Amazon page, or publisher's website
- The exact quote attributed by name to {blurber_name}
- News coverage or interviews confirming the endorsement

CRITICAL — REJECT these false positives:
- The book is published by a company associated with {blurber_name} (e.g., Bloomberg published it, but Michael Bloomberg the person didn't blurb it)
- The book's author works for a company associated with {blurber_name} (e.g., the author works at Bloomberg News, so the LLM assumed Michael Bloomberg endorsed it)
- A publication associated with {blurber_name} reviewed it (a Bloomberg News review is not a Michael Bloomberg endorsement)
- {blurber_name} merely mentioned the book in passing or recommended it in an interview (we want formal blurbs only)

Return ONLY valid JSON:
{{
  "verified": true or false,
  "evidence_url": "URL where you found the blurb, or empty string",
  "quote": "the exact blurb text if you found it, or empty string",
  "reason": "brief explanation of decision"
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
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

    # Step 1: find blurbers on input book
    print(f"\n[1/4] Finding blurbers...")
    blurber_data = find_blurbers(book_title, book_author)
    blurbers = blurber_data.get("blurbers", [])

    if not blurbers:
        print("  No blurbers found.")
        return

    print(f"  Found {len(blurbers)} blurber(s):")
    for b in blurbers:
        name = b.get("name", "?")
        aff = b.get("affiliation", "")
        print(f"    - {name}" + (f" ({aff})" if aff else ""))

    # Step 2: find candidate books each blurber has endorsed
    print(f"\n[2/4] Searching for each blurber's other endorsements...")
    candidates = []  # list of (blurber_name, book_dict)

    for b in blurbers:
        name = b.get("name")
        if not name:
            continue
        print(f"  Searching: {name}")
        result = find_other_endorsements(name, book_title)
        for book in result.get("books", []):
            if book.get("title", "").strip():
                candidates.append((name, book))

    print(f"  Found {len(candidates)} candidate endorsement(s) to verify")

    # Step 3: verify each candidate
    print(f"\n[3/4] Verifying each candidate endorsement...")
    verified = []  # list of (blurber_name, book_dict, verification_dict)

    for blurber_name, book in candidates:
        title = book["title"]
        author = book.get("author", "unknown")
        print(f"  Verifying: {blurber_name} -> '{title}'")
        v = verify_endorsement(blurber_name, title, author)
        if v.get("verified"):
            print(f"    ✓ verified")
            verified.append((blurber_name, book, v))
        else:
            reason = v.get("reason", "no reason given")
            print(f"    ✗ rejected: {reason}")

    # Step 4: aggregate verified endorsements
    print(f"\n[4/4] Ranked recommendations:")
    print(f"{'-' * 60}")

    if not verified:
        print("  No verified cross-referenced books found.")
        return

    book_to_endorsers = {}
    for blurber_name, book, v in verified:
        title = book["title"]
        if title not in book_to_endorsers:
            book_to_endorsers[title] = {
                "title": title,
                "author": book.get("author", "unknown"),
                "year": book.get("year"),
                "one_line": book.get("one_line", ""),
                "endorsers": [],
                "evidence": [],
            }
        if blurber_name not in book_to_endorsers[title]["endorsers"]:
            book_to_endorsers[title]["endorsers"].append(blurber_name)
            book_to_endorsers[title]["evidence"].append({
                "endorser": blurber_name,
                "url": v.get("evidence_url", ""),
                "quote": v.get("quote", ""),
            })

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
    print(f"\n  Full output (with evidence URLs) saved to {out_path}")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found.")
        print("Create a .env file with: ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    if len(sys.argv) >= 3:
        title = sys.argv[1]
        author = sys.argv[2]
    else:
        title = "Streetwise: Getting to and Through Goldman Sachs"
        author = "Lloyd Blankfein"

    recommend_books(title, author)