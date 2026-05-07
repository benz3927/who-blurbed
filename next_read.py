"""
next-read: blurb-based book recommender (OpenAI version)

Two modes:
  Mode A (book): given a book title + author, find blurbers, then find what
    those blurbers endorsed elsewhere
  Mode B (name): given an endorser name, find all books they have blurbed

Architecture: agentic pipeline with separate Search, Verify, and Rank agents.

Improvements over v3:
- LLM-based name variant expansion (handles "Mike" / "Michael" automatically)
- Multi-pass search per blurber (merges results from name variants)
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
MODEL = "gpt-4.1"


# ============================================================
# OpenAI call wrapper with retry
# ============================================================

def _call_with_search(prompt, max_output_tokens=3000, max_retries=2):
    """Call OpenAI Responses API with web search enabled."""
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                tools=[{"type": "web_search"}],
                max_output_tokens=max_output_tokens,
            )
            text = response.output_text or ""
            if not text:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                print(f"  [debug] empty response after {max_retries + 1} attempts")
            return text
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            print(f"  [error] OpenAI API call failed: {e}")
            return ""
    return ""


# ============================================================
# NAME VARIANTS (LLM-driven, no hardcoded map)
# ============================================================

def name_variants(full_name):
    """Use the LLM to generate plausible variants of a person's name for searching."""
    prompt = f"""Generate plausible name variants for this person, to be used in a web search for their book endorsements:

"{full_name}"

Return common variations someone might use to refer to the same individual: formal name, nicknames, common shortenings, middle initial variations, etc.

Examples:
- "Michael Bloomberg" -> ["Michael Bloomberg", "Mike Bloomberg", "Michael R. Bloomberg"]
- "William Buffett" -> ["William Buffett", "Bill Buffett"]
- "Robert Iger" -> ["Robert Iger", "Bob Iger"]

Include only real, commonly-used variants for this same individual. Always include the original input as the first item.

Return ONLY valid JSON (no other text, no markdown):
{{"variants": ["Original Name", "Variant 2", "Variant 3"]}}"""

    try:
        response = client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=400,
        )
        text = response.output_text or ""
        data = _extract_json(text)
        variants = data.get("variants", [])
        if variants:
            if full_name not in variants:
                variants = [full_name] + variants
            seen = set()
            deduped = []
            for v in variants:
                key = v.lower().strip()
                if key not in seen:
                    seen.add(key)
                    deduped.append(v)
            return deduped
        return [full_name]
    except Exception as e:
        print(f"  [warning] name_variants failed: {e}")
        return [full_name]


# ============================================================
# AGENT 1: BLURBER FINDER
# ============================================================

def blurber_finder_agent(book_title, book_author):
    """Find every real human blurber on the back cover."""
    prompt = f"""You are the BLURBER FINDER agent for the next-read app.

Find EVERY real human individual who wrote an endorsement/blurb for this book:

Book: "{book_title}" by {book_author}

BE EXHAUSTIVE. Books often have 6-12+ blurbers.

Search strategy (run multiple searches):
1. Amazon's "Editorial Reviews" / "Praise for" section
2. Publisher's product page
3. Barnes & Noble
4. Google Books preview
5. "praise for {book_title}"
6. "{book_title} blurbs" or "{book_title} endorsements"

Include:
- Named individual humans only (e.g., "Warren Buffett", "Niall Ferguson")
- Each person's affiliation if available
- A snippet of their actual blurb text
- Authors of forewords or introductions count

EXCLUDE:
- Publications or organizations (e.g., "The New York Times")
- Anonymous or generic praise
- Publisher's own marketing copy
- The book's own author or co-authors

Return ONLY valid JSON (no other text, no markdown, no trailing commas):
{{
  "book_title": "...",
  "book_author": "...",
  "blurbers": [
    {{"name": "Full Name", "affiliation": "title or org", "quote_snippet": "first 10-15 words"}}
  ]
}}

Try to find AT LEAST 6-8 blurbers if the book has them."""
    text = _call_with_search(prompt)
    return _extract_json(text)


# ============================================================
# AGENT 2: ENDORSEMENT SEARCH (single pass for one name variant)
# ============================================================

def _endorsement_search_single(blurber_name, exclude_book=None):
    """One pass of endorsement search for a specific name string."""
    exclude_clause = f'\nAlso exclude this specific book: "{exclude_book}".' if exclude_book else ""
    
    prompt = f"""You are the ENDORSEMENT SEARCH agent for the next-read app.

Find books that {blurber_name} (the specific individual person) has personally endorsed for SOMEONE ELSE'S book — meaning {blurber_name} wrote a blurb, foreword, or introduction for a book authored by another person.{exclude_clause}

CRITICAL — DO NOT INCLUDE:
- Books AUTHORED or CO-AUTHORED by {blurber_name}
- Books where {blurber_name} is the subject (biography, memoir target, quote collection)
- Books titled with {blurber_name}'s name
- Books just because the person works at a related company
- Books published by an associated company
- Reviews by their employer's publication
- Casual mentions in interviews

Example for Michael Bloomberg:
- INCLUDE: "Principles" by Ray Dalio (Bloomberg blurbed it)
- EXCLUDE: "Bloomberg by Bloomberg" (he wrote it)

BE EXHAUSTIVE. Active blurbers like Bloomberg, Rubenstein, or Cramer have endorsed many books. Do not stop after finding 1-2.

Search strategy (run multiple searches):
1. "praise for [book]" attributed to {blurber_name}
2. "{blurber_name}" wrote foreword for
3. "{blurber_name}" wrote introduction for
4. Books with cover blurb by {blurber_name}
5. Publisher pages quoting {blurber_name} about another author's book

WHAT COUNTS AS AN ENDORSEMENT:
- Back-cover blurbs / praise quotes attributed by name
- Forewords or introductions written for someone else's book
- "Praise for" sections quoting the person
- Publisher marketing pages quoting their endorsement

Look across the last 15 years.

Return ONLY valid JSON (no other text, no markdown, no trailing commas):
{{
  "endorser": "{blurber_name}",
  "books": [
    {{"title": "Book Title", "author": "Author Name (must NOT be {blurber_name})", "year": 2023, "one_line": "brief description"}}
  ]
}}

Try hard to find at least 3-5 if the person is a known endorser."""
    text = _call_with_search(prompt)
    return _extract_json(text)


def endorsement_search_agent(blurber_name, exclude_book=None):
    """Multi-pass endorsement search across name variants. Returns merged candidates."""
    variants = name_variants(blurber_name)
    
    all_books = {}
    
    for variant in variants:
        try:
            result = _endorsement_search_single(variant, exclude_book)
            for book in result.get("books", []):
                title = book.get("title", "").strip()
                if not title:
                    continue
                key = title.lower()
                if key not in all_books:
                    all_books[key] = book
        except Exception as e:
            print(f"  [warning] search failed for variant '{variant}': {e}")
    
    return {
        "endorser": blurber_name,
        "variants_searched": variants,
        "books": list(all_books.values()),
    }


# ============================================================
# AGENT 3: VERIFIER
# ============================================================

def verifier_agent(blurber_name, book_title, book_author):
    """Verify a single endorsement claim."""
    prompt = f"""You are the VERIFIER agent for the next-read app.

Confirm whether {blurber_name} (the specific individual person, including any common name variants like nicknames) personally endorsed this book, which should be AUTHORED BY SOMEONE ELSE:

Book: "{book_title}" by {book_author}

FIRST CHECK: is {blurber_name} the author or co-author of this book?
- If YES, set verified=false with reason "{blurber_name} is the author/co-author".
- If NO, continue.

Search for direct evidence of an endorsement:
- The blurb on the back cover, Amazon page, or publisher's site
- Foreword or introduction written by {blurber_name}
- Exact quote attributed by name to {blurber_name}
- Publisher announcements or marketing copy quoting the endorsement

ACCEPT (verified=true):
- Back-cover blurbs / praise quotes attributed by name
- Forewords or introductions written by the person
- "Praise for" sections quoting the person
- Publisher marketing pages quoting the endorsement

REJECT (verified=false):
- {blurber_name} is the author or co-author
- Book is about {blurber_name} (biography, memoir, quote collection)
- Book published by a company associated with the person (Bloomberg LP != Michael Bloomberg the person)
- Book's author works for an associated company without separate personal endorsement
- A publication associated with the person reviewed it
- Person merely mentioned the book in passing

Return ONLY valid JSON (no other text, no markdown, no trailing commas):
{{
  "verified": true,
  "evidence_url": "URL where you found the blurb, or empty",
  "quote": "exact blurb text if found, or empty",
  "reason": "brief explanation"
}}"""
    text = _call_with_search(prompt, max_output_tokens=1500)
    return _extract_json(text)


# ============================================================
# AGENT 4: RANKER (deterministic, no LLM)
# ============================================================

def ranker_agent(verified_endorsements):
    """Aggregate and rank verified endorsements."""
    book_to_endorsers = {}
    for blurber_name, book, v in verified_endorsements:
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
    return sorted(
        book_to_endorsers.values(),
        key=lambda x: (-len(x["endorsers"]), -(x.get("year") or 0)),
    )


# ============================================================
# ORCHESTRATORS
# ============================================================

def recommend_from_book(book_title, book_author, log=print):
    """Mode A: book -> recommendations."""
    log(f"\n[1/4] Blurber Finder: scanning '{book_title}'...")
    blurber_data = blurber_finder_agent(book_title, book_author)
    blurbers = blurber_data.get("blurbers", [])
    log(f"  Found {len(blurbers)} blurber(s):")
    for b in blurbers:
        log(f"    - {b.get('name')} ({b.get('affiliation', '')})")

    if not blurbers:
        return {"blurbers": [], "recommendations": []}

    log(f"\n[2/4] Endorsement Search: parallel search across blurbers (with name variants)...")
    candidates = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_name = {
            executor.submit(endorsement_search_agent, b["name"], book_title): b["name"]
            for b in blurbers if b.get("name")
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                variants_used = result.get("variants_searched", [name])
                count = 0
                for book in result.get("books", []):
                    if book.get("title", "").strip():
                        candidates.append((name, book))
                        count += 1
                log(f"  {name} (variants: {', '.join(variants_used)}): {count} candidate(s)")
            except Exception as e:
                log(f"  {name}: error - {e}")

    log(f"  Total candidates: {len(candidates)}")

    log(f"\n[3/4] Verifier: parallel verification...")
    verified = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_pair = {
            executor.submit(verifier_agent, name, book["title"], book.get("author", "")): (name, book)
            for name, book in candidates
        }
        for future in as_completed(future_to_pair):
            name, book = future_to_pair[future]
            try:
                v = future.result()
                if v.get("verified"):
                    log(f"  + {name} -> '{book['title']}'")
                    verified.append((name, book, v))
                else:
                    log(f"  - {name} -> '{book['title']}' ({v.get('reason', '')[:60]})")
            except Exception as e:
                log(f"  error verifying {name}: {e}")

    log(f"\n[4/4] Ranker: aggregating {len(verified)} verified endorsements...")
    recommendations = ranker_agent(verified)

    return {
        "input_book": {"title": book_title, "author": book_author},
        "blurbers": blurbers,
        "recommendations": recommendations,
    }


def recommend_from_name(person_name, log=print):
    """Mode B: endorser name -> all books they have blurbed."""
    log(f"\n[1/3] Endorsement Search: finding everything {person_name} has blurbed (with name variants)...")
    result = endorsement_search_agent(person_name, exclude_book=None)
    variants_used = result.get("variants_searched", [person_name])
    log(f"  Variants searched: {', '.join(variants_used)}")
    candidates = []
    for book in result.get("books", []):
        if book.get("title", "").strip():
            candidates.append((person_name, book))
    log(f"  Found {len(candidates)} candidate(s)")

    log(f"\n[2/3] Verifier: parallel verification...")
    verified = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_book = {
            executor.submit(verifier_agent, person_name, book["title"], book.get("author", "")): book
            for _, book in candidates
        }
        for future in as_completed(future_to_book):
            book = future_to_book[future]
            try:
                v = future.result()
                if v.get("verified"):
                    log(f"  + '{book['title']}'")
                    verified.append((person_name, book, v))
                else:
                    log(f"  - '{book['title']}' ({v.get('reason', '')[:60]})")
            except Exception as e:
                log(f"  error: {e}")

    log(f"\n[3/3] Ranker: {len(verified)} verified...")
    recommendations = ranker_agent(verified)

    return {
        "input_name": person_name,
        "recommendations": recommendations,
    }


# ============================================================
# UTILS
# ============================================================

def _extract_json(text):
    """Pull JSON object out of a potentially noisy LLM response."""
    if not text:
        return {}
    
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    
    candidate = text[start:end + 1]
    
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    
    cleaned = candidate
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    for end_pos in range(end, start, -1):
        sub = text[start:end_pos + 1]
        if sub.endswith("}"):
            try:
                return json.loads(sub)
            except json.JSONDecodeError:
                continue
    
    print(f"  [warning] failed to parse JSON; first 200 chars: {text[:200]}")
    return {}


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    if len(sys.argv) == 2:
        result = recommend_from_name(sys.argv[1])
    elif len(sys.argv) >= 3:
        result = recommend_from_book(sys.argv[1], sys.argv[2])
    else:
        result = recommend_from_book(
            "Streetwise: Getting to and Through Goldman Sachs",
            "Lloyd Blankfein",
        )

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATIONS")
    print("=" * 60)
    for i, rec in enumerate(result.get("recommendations", []), 1):
        endorsers = ", ".join(rec["endorsers"])
        year = rec.get("year") or "n/a"
        print(f"\n{i}. {rec['title']} ({year})")
        print(f"   by {rec['author']}")
        print(f"   endorsed by: {endorsers}")
        if rec.get("one_line"):
            print(f"   {rec['one_line']}")

    with open("recommendations.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to recommendations.json")