"""
next-read: blurb-based book recommender (OpenAI Responses + web_search)

Architecture:
- Optional disambiguation step (handles common names like Ken Griffin)
- Run the endorsement search N times in parallel
- Merge unique books across runs
- Verifier catches false positives (wrong person, author confusion) — but
  defaults to ACCEPT candidates the search agent surfaced, instead of
  requiring independent re-proof of every blurb
- Cache everything on disk for instant repeat queries
"""

import json
import os
import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
MODEL = "gpt-4.1"

RUNS_PER_VARIANT = 5

CACHE_FILE = "cache.json"
_cache = None


# ============================================================
# CACHE
# ============================================================

def _load_cache():
    global _cache
    if _cache is None:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    _cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache():
    if _cache is None:
        return
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_cache, f, indent=2)
    except OSError:
        pass


def _cache_key(prefix, *args):
    payload = json.dumps([prefix, args], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _cache_get(prefix, *args):
    return _load_cache().get(_cache_key(prefix, *args))


def _cache_set(prefix, value, *args):
    cache = _load_cache()
    cache[_cache_key(prefix, *args)] = value
    _save_cache()


# ============================================================
# DEDUP KEY
# ============================================================

def _book_dedup_key(book):
    title = (book.get("title") or "").lower().strip()
    title = title.split(":")[0].split(" - ")[0].split(" — ")[0]
    title = re.sub(r"[^a-z0-9 ]+", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = title[:30]

    author = (book.get("author") or "").lower().strip()
    author = re.sub(r"[^a-z0-9 ]+", "", author)
    author = re.sub(r"\s+", " ", author).strip()
    author = author.split(" and ")[0].split(",")[0].strip()

    return f"{author}|{title}"


def _merge_book_into(books_dict, book):
    key = _book_dedup_key(book)
    if not key.replace("|", "").strip():
        return
    existing = books_dict.get(key)
    if existing is None:
        books_dict[key] = book
        return
    if len(book.get("title", "")) > len(existing.get("title", "")):
        merged = dict(existing)
        merged.update({k: v for k, v in book.items() if v})
        books_dict[key] = merged


# ============================================================
# OpenAI call wrappers
# ============================================================

def _call_with_search(prompt, max_output_tokens=5000, max_retries=2, temperature=0):
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                tools=[{"type": "web_search"}],
                max_output_tokens=max_output_tokens,
                temperature=temperature,
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


def _call_plain(prompt, max_output_tokens=600):
    try:
        response = client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=0,
        )
        return response.output_text or ""
    except Exception as e:
        print(f"  [warning] plain LLM call failed: {e}")
        return ""


# ============================================================
# DISAMBIGUATION
# ============================================================

def disambiguate_person(person_name):
    cached = _cache_get("disambig", person_name)
    if cached is not None:
        return cached

    prompt = f"""A user wants to find books that {person_name} has personally endorsed (written a blurb for).

If "{person_name}" is a common name shared by multiple notable people, identify them.

Pick the SINGLE person most likely to be a book blurb-writer. Priority order:
1. Business executives, investors, finance figures (most likely to blurb business books)
2. Authors, journalists, academics (often blurb books in their field)
3. Politicians, public intellectuals
4. Other public figures

Return ONLY valid JSON, no markdown:
{{
  "primary": "Full name with brief disambiguator in parens, e.g. 'Ken Griffin (Citadel founder)'",
  "alternatives": ["Other notable people with this name"],
  "is_ambiguous": true or false
}}

If the name refers to only one likely person, set is_ambiguous=false and leave alternatives empty.

Name: {person_name}"""

    text = _call_plain(prompt, max_output_tokens=400)
    data = _extract_json(text)
    primary = data.get("primary", person_name)
    result = {
        "primary": primary,
        "alternatives": data.get("alternatives", []),
        "is_ambiguous": bool(data.get("is_ambiguous")),
    }
    _cache_set("disambig", result, person_name)
    return result


# ============================================================
# NAME VARIANTS
# ============================================================

def name_variants(full_name):
    cached = _cache_get("variants", full_name)
    if cached is not None:
        return cached

    prompt = f"""Generate plausible name variants for this person, to be used in a web search for their book endorsements:

"{full_name}"

The input may include a disambiguator in parens like "Ken Griffin (Citadel founder)". Preserve the disambiguator across variants so searches stay targeted.

Return common variations someone might use to refer to the same individual: formal name, nicknames, common shortenings, middle initial variations, etc.

Examples:
- "Michael Bloomberg" -> ["Michael Bloomberg", "Mike Bloomberg", "Michael R. Bloomberg"]
- "Ken Griffin (Citadel founder)" -> ["Ken Griffin (Citadel founder)", "Kenneth Griffin (Citadel founder)", "Kenneth C. Griffin (Citadel founder)"]
- "Robert Iger" -> ["Robert Iger", "Bob Iger"]

Include only real, commonly-used variants for this same individual. Always include the original input as the first item.

Return ONLY valid JSON (no other text, no markdown):
{{"variants": ["Original Name", "Variant 2", "Variant 3"]}}"""

    text = _call_plain(prompt, max_output_tokens=400)
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
        _cache_set("variants", deduped, full_name)
        return deduped
    return [full_name]


# ============================================================
# AGENT 1: BLURBER FINDER
# ============================================================

def blurber_finder_agent(book_title, book_author):
    cached = _cache_get("blurbers", book_title, book_author)
    if cached is not None:
        return cached

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
- Named individual humans only
- Each person's affiliation if available
- A snippet of their actual blurb text
- Authors of forewords or introductions count

EXCLUDE:
- Publications or organizations
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
    result = _extract_json(text)
    if result.get("blurbers") is not None:
        _cache_set("blurbers", result, book_title, book_author)
    return result


# ============================================================
# AGENT 2: ENDORSEMENT SEARCH
# ============================================================

def _endorsement_search_one_run(blurber_name, exclude_book, run_idx):
    exclude_clause = f'\nAlso exclude this specific book: "{exclude_book}".' if exclude_book else ""
    prompt = f"""You are the ENDORSEMENT SEARCH agent for the next-read app.

Find books that {blurber_name} (the specific individual person, including any disambiguator in parens) has personally endorsed for SOMEONE ELSE'S book — meaning they wrote a blurb, foreword, or introduction for a book authored by another person.{exclude_clause}

CRITICAL: if "{blurber_name}" includes a disambiguator in parens (e.g. "(Citadel founder)"), only include endorsements that are clearly from THAT specific person, not other people who share the name.

DO NOT INCLUDE:
- Books AUTHORED or CO-AUTHORED by {blurber_name}
- Books where {blurber_name} is the subject
- Books titled with {blurber_name}'s name
- Books just because the person works at a related company
- Books published by an associated company
- Reviews by their employer's publication
- Casual mentions in interviews
- Books endorsed by a different person with the same name

BE EXHAUSTIVE. Active blurbers have endorsed many books. Do not stop after finding 1-2.

Search strategy:
1. "praise for [book]" attributed to {blurber_name}
2. "{blurber_name}" wrote foreword for
3. "{blurber_name}" wrote introduction for
4. Books with cover blurb by {blurber_name}
5. Publisher pages quoting {blurber_name} about another author's book
6. Search for {blurber_name}'s name on Amazon book pages in their area of expertise
7. Recent high-profile nonfiction books in topics {blurber_name} covers (2023-2025)

WHAT COUNTS AS AN ENDORSEMENT:
- Back-cover blurbs / praise quotes attributed by name
- Forewords or introductions written for someone else's book
- "Praise for" sections quoting the person
- Publisher marketing pages quoting their endorsement

Look across the last 15 years.

Use the EXACT FULL title of the book including any subtitle. Do not abbreviate.

Return ONLY valid JSON (no other text, no markdown, no trailing commas):
{{
  "endorser": "{blurber_name}",
  "books": [
    {{"title": "Full Book Title Including Subtitle", "author": "Author Name (must NOT be {blurber_name})", "year": 2023, "one_line": "brief description"}}
  ]
}}

Try hard to find at least 3-5 if the person is a known endorser."""
    text = _call_with_search(prompt, max_output_tokens=5000, temperature=0.7)
    return _extract_json(text)


def _endorsement_search_single_variant(blurber_name, exclude_book):
    cached = _cache_get("endorse_single_v5", blurber_name, exclude_book or "")
    if cached is not None:
        return cached

    all_books = {}
    with ThreadPoolExecutor(max_workers=RUNS_PER_VARIANT) as ex:
        futures = [
            ex.submit(_endorsement_search_one_run, blurber_name, exclude_book, i)
            for i in range(RUNS_PER_VARIANT)
        ]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                for book in result.get("books", []):
                    if not book.get("title", "").strip():
                        continue
                    _merge_book_into(all_books, book)
            except Exception as e:
                print(f"  [warning] one of the parallel runs failed: {e}")

    merged = {"endorser": blurber_name, "books": list(all_books.values())}
    _cache_set("endorse_single_v5", merged, blurber_name, exclude_book or "")
    return merged


def endorsement_search_agent(blurber_name, exclude_book=None):
    cached = _cache_get("endorse_merged_v5", blurber_name, exclude_book or "")
    if cached is not None:
        return cached

    variants = name_variants(blurber_name)
    all_books = {}

    for variant in variants:
        try:
            result = _endorsement_search_single_variant(variant, exclude_book)
            for book in result.get("books", []):
                if not book.get("title", "").strip():
                    continue
                _merge_book_into(all_books, book)
        except Exception as e:
            print(f"  [warning] variant '{variant}' failed: {e}")

    final = {
        "endorser": blurber_name,
        "variants_searched": variants,
        "books": list(all_books.values()),
    }
    _cache_set("endorse_merged_v5", final, blurber_name, exclude_book or "")
    return final


# ============================================================
# AGENT 3: VERIFIER (now defaults to ACCEPT)
# ============================================================

def verifier_agent(blurber_name, book_title, book_author):
    cached = _cache_get("verify_v3", blurber_name, book_title, book_author)
    if cached is not None:
        return cached

    prompt = f"""You are the VERIFIER agent for the next-read app.

A previous search step has already found this as a likely book endorsement by {blurber_name}:

Book: "{book_title}" by {book_author}

Your job is NOT to independently re-prove the endorsement. The previous search step already did that work. Your job is ONLY to catch a few specific kinds of false positives.

DEFAULT BEHAVIOR: verified=true.

Search the web briefly to check for these specific RED FLAGS. Set verified=false ONLY if you find clear evidence of one:

RED FLAG 1: {blurber_name} is the AUTHOR or CO-AUTHOR of this book (not a blurber).
  -> reason: "Author/co-author, not blurber"

RED FLAG 2: The blurb / endorsement is from a DIFFERENT person with the same name (e.g., the user wants Ken Griffin the Citadel founder, but the blurb is from Ken Griffin the country music singer).
  -> reason: "Different person with same name"

RED FLAG 3: The book is BY OR ABOUT {blurber_name} (e.g., biographies, memoirs of the person, books with their name in the title that they didn't blurb).
  -> reason: "Book is by/about the person"

If you can confirm the endorsement is real (back-cover quote, foreword, "praise for" section attributed by name): verified=true. Include the quote and URL.

If you cannot find an explicit blurb but ALSO cannot find any of the red flags above: verified=true. The earlier search step found this, give it the benefit of the doubt. Include whatever quote text or URL you can.

If you DO find a red flag: verified=false with the matching reason.

DO NOT reject a candidate just because:
- You couldn't independently find the blurb (search results vary)
- The blurb is from years ago and hard to verify online now
- You're not 100% certain it's a "formal" blurb vs an informal endorsement

Return ONLY valid JSON (no markdown, no trailing commas):
{{
  "verified": true,
  "evidence_url": "URL where you found the blurb, or empty",
  "quote": "exact blurb text if found, or empty",
  "reason": "one of the red flag phrases above, or brief confirmation note"
}}"""
    text = _call_with_search(prompt, max_output_tokens=2000, temperature=0)
    result = _extract_json(text)
    # Default to verified=True if the model returned something but didn't
    # explicitly say verified=false. This matches the "accept by default" policy.
    if result:
        verified = result.get("verified")
        if verified is None:
            verified = True
        else:
            verified = bool(verified)
    else:
        # Empty response — be cautious and reject
        verified = False
    final = {
        "verified": verified,
        "evidence_url": result.get("evidence_url", "") if result else "",
        "quote": result.get("quote", "") if result else "",
        "reason": result.get("reason", "") if result else "empty verifier response",
    }
    _cache_set("verify_v3", final, blurber_name, book_title, book_author)
    return final


# ============================================================
# AGENT 4: RANKER
# ============================================================

def ranker_agent(verified_endorsements):
    by_key = {}
    for blurber_name, book, v in verified_endorsements:
        key = _book_dedup_key(book)
        if key not in by_key:
            by_key[key] = {
                "title": book["title"],
                "author": book.get("author", "unknown"),
                "year": book.get("year"),
                "one_line": book.get("one_line", ""),
                "endorsers": [],
                "evidence": [],
            }
        if len(book.get("title", "")) > len(by_key[key]["title"]):
            by_key[key]["title"] = book["title"]
        if blurber_name not in by_key[key]["endorsers"]:
            by_key[key]["endorsers"].append(blurber_name)
            by_key[key]["evidence"].append({
                "endorser": blurber_name,
                "url": v.get("evidence_url", ""),
                "quote": v.get("quote", ""),
            })
    return sorted(
        by_key.values(),
        key=lambda x: (-len(x["endorsers"]), -(x.get("year") or 0)),
    )


# ============================================================
# ORCHESTRATORS
# ============================================================

def recommend_from_book(book_title, book_author, log=print):
    log(f"\n[1/4] Blurber Finder: scanning '{book_title}'...")
    blurber_data = blurber_finder_agent(book_title, book_author)
    blurbers = blurber_data.get("blurbers", [])
    log(f"  Found {len(blurbers)} blurber(s):")
    for b in blurbers:
        log(f"    - {b.get('name')} ({b.get('affiliation', '')})")

    if not blurbers:
        return {"blurbers": [], "recommendations": []}

    log(f"\n[2/4] Endorsement Search ({RUNS_PER_VARIANT} parallel runs per variant)...")
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

    log(f"\n[3/4] Verifier: parallel verification (accept by default)...")
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
    log(f"\n[1/4] Disambiguating '{person_name}'...")
    disambig = disambiguate_person(person_name)
    primary = disambig["primary"]
    log(f"  Resolved to: {primary}")
    if disambig.get("alternatives"):
        log(f"  (Other people with this name: {', '.join(disambig['alternatives'])})")

    log(f"\n[2/4] Endorsement Search ({RUNS_PER_VARIANT} parallel runs per variant)...")
    result = endorsement_search_agent(primary, exclude_book=None)
    variants_used = result.get("variants_searched", [primary])
    log(f"  Variants searched: {', '.join(variants_used)}")
    candidates = []
    for book in result.get("books", []):
        if book.get("title", "").strip():
            candidates.append((primary, book))
    log(f"  Found {len(candidates)} candidate(s)")

    log(f"\n[3/4] Verifier: parallel verification (accept by default)...")
    verified = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_book = {
            executor.submit(verifier_agent, primary, book["title"], book.get("author", "")): book
            for _, book in candidates
        }
        for future in as_completed(future_to_book):
            book = future_to_book[future]
            try:
                v = future.result()
                if v.get("verified"):
                    log(f"  + '{book['title']}'")
                    verified.append((primary, book, v))
                else:
                    log(f"  - '{book['title']}' ({v.get('reason', '')[:60]})")
            except Exception as e:
                log(f"  error: {e}")

    log(f"\n[4/4] Ranker: {len(verified)} verified...")
    recommendations = ranker_agent(verified)
    return {
        "input_name": person_name,
        "resolved_to": primary,
        "alternatives": disambig.get("alternatives", []),
        "recommendations": recommendations,
    }


# ============================================================
# UTILS
# ============================================================

def _extract_json(text):
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
        result = recommend_from_name("Bradley Hope")

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATIONS")
    print("=" * 60)
    if result.get("resolved_to"):
        print(f"Searched as: {result['resolved_to']}")
        if result.get("alternatives"):
            print(f"Other people with this name: {', '.join(result['alternatives'])}")
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