"""
next-read: blurb-based book recommender (OpenAI version)

v3.6:
- File-based result caching for consistency, speed, and cost.
- Majority-vote verifier: each (blurber, book) pair is verified N times,
  and accepted only if >= threshold runs agree. Cached after voting.
- Configurable via VERIFIER_RUNS and VERIFIER_THRESHOLD.
- Robust JSON extraction that handles curly quotes and other LLM quirks.
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

SEARCH_RUNS_PER_VARIANT = 2
MAX_VARIANTS = 3

# Majority-vote verifier settings.
# Set VERIFIER_RUNS=1 to disable voting (single run, cheaper).
VERIFIER_RUNS = 3
VERIFIER_THRESHOLD = 2  # need >= this many "verified" votes to accept

CACHE_FILE = "cache.json"
_cache = None  # lazy-loaded


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
    except OSError as e:
        print(f"  [warning] cache save failed: {e}")


def _cache_key(prefix, *args):
    """Stable hash key for the given function name + args."""
    payload = json.dumps([prefix, args], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _cache_get(prefix, *args):
    cache = _load_cache()
    return cache.get(_cache_key(prefix, *args))


def _cache_set(prefix, value, *args):
    cache = _load_cache()
    cache[_cache_key(prefix, *args)] = value
    _save_cache()


# ============================================================
# OpenAI call wrapper with retry
# ============================================================

def _call_with_search(prompt, max_output_tokens=3000, max_retries=2):
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
# NAME VARIANTS (cached)
# ============================================================

def name_variants(full_name):
    cached = _cache_get("name_variants", full_name)
    if cached is not None:
        return cached
    
    prompt = f"""Generate plausible name variants for this person, to be used in a web search for their book endorsements:

"{full_name}"

Return common variations: formal name, nicknames, common shortenings, middle initial variations, etc.

Examples:
- "Michael Bloomberg" -> ["Michael Bloomberg", "Mike Bloomberg", "Michael R. Bloomberg"]
- "William Buffett" -> ["William Buffett", "Bill Buffett"]
- "Robert Iger" -> ["Robert Iger", "Bob Iger"]

Always include the original input first. Return AT MOST 3 variants.

Return ONLY valid JSON (no other text, no markdown). Use STRAIGHT ASCII quotes only inside JSON strings; never use curly/typographic quotes:
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
            result = deduped[:MAX_VARIANTS]
            _cache_set("name_variants", result, full_name)
            return result
        return [full_name]
    except Exception as e:
        print(f"  [warning] name_variants failed: {e}")
        return [full_name]


# ============================================================
# AGENT 1: BLURBER FINDER (cached)
# ============================================================

def blurber_finder_agent(book_title, book_author):
    cached = _cache_get("blurber_finder", book_title, book_author)
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

Return ONLY valid JSON (no other text, no markdown, no trailing commas).
IMPORTANT: Use STRAIGHT ASCII quotes (") only as JSON delimiters and inside string values. Never use curly typographic quotes ("/" or '/'). If you need to quote text inside a string, paraphrase or escape with backslash:
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
    if result:
        _cache_set("blurber_finder", result, book_title, book_author)
    return result


# ============================================================
# AGENT 2: ENDORSEMENT SEARCH (cached at the merged-result level)
# ============================================================

def _endorsement_search_single(blurber_name, exclude_book=None):
    exclude_clause = f'\nAlso exclude this specific book: "{exclude_book}".' if exclude_book else ""
    
    prompt = f"""You are the ENDORSEMENT SEARCH agent for the next-read app.

Find books that {blurber_name} (the specific individual person) has personally endorsed for SOMEONE ELSE'S book \u2014 meaning {blurber_name} wrote a blurb, foreword, or introduction for a book authored by another person.{exclude_clause}

CRITICAL \u2014 DO NOT INCLUDE:
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

BE EXHAUSTIVE.

Search strategy:
1. "praise for [book]" attributed to {blurber_name}
2. "{blurber_name}" wrote foreword for
3. "{blurber_name}" wrote introduction for
4. Books with cover blurb by {blurber_name}
5. Publisher pages quoting {blurber_name}

Look across the last 15 years.

Return ONLY valid JSON (no other text, no markdown, no trailing commas).
IMPORTANT: Use STRAIGHT ASCII quotes (") only as JSON delimiters and inside string values. Never use curly typographic quotes ("/" or '/'). If a blurb you want to include contains quotes, paraphrase it or remove the inner quotes:
{{
  "endorser": "{blurber_name}",
  "books": [
    {{"title": "Book Title", "author": "Author Name (must NOT be {blurber_name})", "year": 2023, "one_line": "brief description without inner quotes"}}
  ]
}}"""
    text = _call_with_search(prompt)
    return _extract_json(text)


def endorsement_search_agent(blurber_name, exclude_book=None):
    """Cached at the merged-result level."""
    cached = _cache_get("endorsement_search", blurber_name, exclude_book)
    if cached is not None:
        return cached
    
    variants = name_variants(blurber_name)
    
    tasks = []
    for v in variants:
        for run_idx in range(SEARCH_RUNS_PER_VARIANT):
            tasks.append((v, run_idx))
    
    all_books = {}
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_task = {
            executor.submit(_endorsement_search_single, variant, exclude_book): (variant, run_idx)
            for variant, run_idx in tasks
        }
        for future in as_completed(future_to_task):
            variant, run_idx = future_to_task[future]
            try:
                result = future.result()
                for book in result.get("books", []):
                    title = book.get("title", "").strip()
                    if not title:
                        continue
                    key = title.lower()
                    if key not in all_books:
                        all_books[key] = book
            except Exception as e:
                print(f"  [warning] search failed for variant '{variant}' run {run_idx}: {e}")
    
    result = {
        "endorser": blurber_name,
        "variants_searched": variants,
        "runs_per_variant": SEARCH_RUNS_PER_VARIANT,
        "books": list(all_books.values()),
    }
    if all_books:
        _cache_set("endorsement_search", result, blurber_name, exclude_book)
    return result


# ============================================================
# AGENT 3: VERIFIER (majority-vote, cached)
# ============================================================

def _verifier_single(blurber_name, book_title, book_author):
    """One verifier call. No caching here \u2014 caching happens at the voting layer."""
    prompt = f"""You are the VERIFIER agent for the next-read app.

Confirm whether {blurber_name} (the specific individual person) personally endorsed this book, which should be AUTHORED BY SOMEONE ELSE:

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
- Book published by a company associated with the person
- A publication associated with the person reviewed it
- Person merely mentioned the book in passing

Return ONLY valid JSON (no other text, no markdown, no trailing commas).
IMPORTANT: Use STRAIGHT ASCII quotes (") only as JSON delimiters and inside string values. Never use curly typographic quotes ("/" or '/'). If the blurb text you want to include contains quotes, paraphrase or remove the inner quotes:
{{
  "verified": true,
  "evidence_url": "URL where you found the blurb, or empty",
  "quote": "exact blurb text if found, or empty",
  "reason": "brief explanation"
}}"""
    text = _call_with_search(prompt, max_output_tokens=1500)
    return _extract_json(text)


def verifier_agent(blurber_name, book_title, book_author):
    """
    Majority-vote verifier. Runs _verifier_single up to VERIFIER_RUNS times in parallel,
    accepts only if >= VERIFIER_THRESHOLD runs return verified=true.
    Caches the final aggregated result.
    """
    cached = _cache_get("verifier", blurber_name, book_title, book_author)
    if cached is not None:
        return cached
    
    # Fast path: if voting is disabled, behave like the old single-run verifier.
    if VERIFIER_RUNS <= 1:
        result = _verifier_single(blurber_name, book_title, book_author)
        if result:
            _cache_set("verifier", result, blurber_name, book_title, book_author)
        return result
    
    results = []
    with ThreadPoolExecutor(max_workers=VERIFIER_RUNS) as executor:
        futures = [
            executor.submit(_verifier_single, blurber_name, book_title, book_author)
            for _ in range(VERIFIER_RUNS)
        ]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  [warning] verifier run failed: {e}")
                results.append({})
    
    votes_yes = sum(1 for r in results if r.get("verified") is True)
    votes_no = sum(1 for r in results if r.get("verified") is False)
    verified = votes_yes >= VERIFIER_THRESHOLD
    
    # Pick the best evidence: prefer a "verified" run with a real URL + quote.
    best = None
    if verified:
        candidates = [r for r in results if r.get("verified")]
        # rank: has URL AND quote > has URL > has quote > anything else
        candidates.sort(
            key=lambda r: (
                bool(r.get("evidence_url")) and bool(r.get("quote")),
                bool(r.get("evidence_url")),
                bool(r.get("quote")),
            ),
            reverse=True,
        )
        best = candidates[0] if candidates else {}
    else:
        # If rejected, surface the most informative reason.
        rejections = [r for r in results if r.get("verified") is False and r.get("reason")]
        best = rejections[0] if rejections else (results[0] if results else {})
    
    final = {
        "verified": verified,
        "evidence_url": best.get("evidence_url", "") if best else "",
        "quote": best.get("quote", "") if best else "",
        "reason": best.get("reason", "") if best else "",
        "votes": f"{votes_yes}/{VERIFIER_RUNS} verified, {votes_no}/{VERIFIER_RUNS} rejected",
    }
    _cache_set("verifier", final, blurber_name, book_title, book_author)
    return final


# ============================================================
# AGENT 4: RANKER (deterministic, no LLM)
# ============================================================

def ranker_agent(verified_endorsements):
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
                "votes": v.get("votes", ""),
            })
    return sorted(
        book_to_endorsers.values(),
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

    log(f"\n[2/4] Endorsement Search...")
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
                count = 0
                for book in result.get("books", []):
                    if book.get("title", "").strip():
                        candidates.append((name, book))
                        count += 1
                log(f"  {name}: {count} candidate(s)")
            except Exception as e:
                log(f"  {name}: error - {e}")

    log(f"  Total candidates: {len(candidates)}")

    log(f"\n[3/4] Verifier (majority vote: {VERIFIER_THRESHOLD}/{VERIFIER_RUNS})...")
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
                votes = v.get("votes", "")
                if v.get("verified"):
                    log(f"  + {name} -> '{book['title']}' [{votes}]")
                    verified.append((name, book, v))
                else:
                    log(f"  - {name} -> '{book['title']}' [{votes}] ({v.get('reason', '')[:60]})")
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
    log(f"\n[1/3] Endorsement Search: finding everything {person_name} has blurbed...")
    result = endorsement_search_agent(person_name, exclude_book=None)
    variants_used = result.get("variants_searched", [person_name])
    runs = result.get("runs_per_variant", 1)
    log(f"  Variants searched: {', '.join(variants_used)} (x{runs} runs each)")
    candidates = []
    for book in result.get("books", []):
        if book.get("title", "").strip():
            candidates.append((person_name, book))
    log(f"  Found {len(candidates)} candidate(s)")

    log(f"\n[2/3] Verifier (majority vote: {VERIFIER_THRESHOLD}/{VERIFIER_RUNS})...")
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
                votes = v.get("votes", "")
                if v.get("verified"):
                    log(f"  + '{book['title']}' [{votes}]")
                    verified.append((person_name, book, v))
                else:
                    log(f"  - '{book['title']}' [{votes}] ({v.get('reason', '')[:60]})")
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

def _normalize_quotes(s):
    """Normalize curly typographic quotes to safe straight equivalents.
    LLMs often emit curly quotes like 'He said "gripping" about it', which
    breaks JSON parsing because the inner curly quotes look like string
    delimiters to a naive parser after replacement. We escape them.
    """
    # Curly double quotes -> escaped straight double quotes
    s = s.replace("\u201c", "\\\"").replace("\u201d", "\\\"")
    # Other double-quote-like characters
    s = s.replace("\u201f", "\\\"").replace("\u201e", "\\\"")
    # Curly single quotes -> straight apostrophe
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201a", "'").replace("\u201b", "'")
    return s


def _extract_json(text):
    if not text:
        return {}
    
    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    
    candidate = text[start:end + 1]
    
    # Attempt 1: parse as-is
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: strip trailing commas + control chars
    cleaned = re.sub(r',\s*([}\]])', r'\1', candidate)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Attempt 3: normalize curly quotes (most common LLM JSON failure)
    requoted = _normalize_quotes(cleaned)
    try:
        return json.loads(requoted)
    except json.JSONDecodeError:
        pass
    
    # Attempt 4: progressively shorter substrings, each tried both raw and requoted
    for end_pos in range(end, start, -1):
        sub = text[start:end_pos + 1]
        if not sub.endswith("}"):
            continue
        try:
            return json.loads(sub)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_normalize_quotes(sub))
        except json.JSONDecodeError:
            continue
    
    print(f"  [warning] failed to parse JSON; first 500 chars: {text[:500]}")
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