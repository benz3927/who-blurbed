"""
next-read v10: blurb-first, with optional casual shares.

David's spec:
  - PRIMARY: Blurb mode. Physical-book endorsements only — blurb, foreword,
    introduction, jacket quote, praise page. This is THE product.
    Optimized for RECALL: don't miss a real blurb. Zero results is fine;
    we tell the user "this person has not written blurbs for any books, yet."
  - OPTIONAL: A toggle to ALSO show casual personal shares — tweets, blog
    posts, Substack, "I just read this" podcast moments. Acts as backup
    if blurb mode returns empty. NOT shareholder letters, reading lists,
    biographer appendices, or aggregator listicles.

Two gather modes, two pure signal sets:

  BLURB_SIGNALS = {blurb, foreword, introduction, jacket_quote, praise_page}
  CASUAL_SIGNALS = {tweet, blog_post, substack, instagram, podcast_moment,
                    interview_moment}

That's the entire taxonomy. No tiers, no scoring, no "high-confidence" filter.
The gather prompts are explicit about what counts and what doesn't.
"""

import json
import os
import re
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
MODEL = "gpt-4.1"

# Recall-first parameters. David's hard constraint: "don't miss a real blurb."
BLURB_RUNS = 6      # heavy on blurb gather — this is the headline feature
CASUAL_RUNS = 3
MAX_WORKERS = 10

MIN_SAME_PERSON_CONFIDENCE = 0.5

BLURB_SIGNALS = {"blurb", "foreword", "introduction", "jacket_quote", "praise_page"}
CASUAL_SIGNALS = {"tweet", "blog_post", "substack", "instagram",
                  "podcast_moment", "interview_moment", "social_post"}

# Known tags from anywhere in the system; rogue inventions get dropped.
KNOWN_SIGNALS = BLURB_SIGNALS | CASUAL_SIGNALS | {"not_a_blurb", "not_casual", "unknown"}

CACHE_FILE = "cache.json"
_cache = None
_cache_lock = threading.Lock()


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


def _save_cache_locked():
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
    with _cache_lock:
        return _load_cache().get(_cache_key(prefix, *args))


def _cache_set(prefix, value, *args):
    with _cache_lock:
        cache = _load_cache()
        cache[_cache_key(prefix, *args)] = value
        _save_cache_locked()


# ============================================================
# DEDUP
# ============================================================

def _normalize_title(title):
    title = (title or "").lower().strip()
    title = re.sub(r"\s*\([^)]*\)\s*", " ", title)
    title = title.split(":")[0].split(" - ")[0].split(" — ")[0]
    title = re.sub(r"[^a-z0-9 ]+", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:50]


def _book_dedup_key(book):
    return _normalize_title(book.get("title", ""))


def _author_score(author):
    if not author:
        return 0
    a = author.lower()
    if "unknown" in a:
        return 0
    if "?" in a:
        return 1
    return len(author)


def _merge_book_into(books_dict, book):
    key = _book_dedup_key(book)
    if not key.strip():
        return
    existing = books_dict.get(key)
    if existing is None:
        if "signal_types" not in book:
            book["signal_types"] = [book.get("signal_type", "unknown")]
        books_dict[key] = book
        return

    incoming = book.get("signal_types") or [book.get("signal_type", "unknown")]
    existing_signals = existing.get("signal_types") or [existing.get("signal_type", "unknown")]
    existing["signal_types"] = list(dict.fromkeys(existing_signals + incoming))

    if len(book.get("title", "")) > len(existing.get("title", "")):
        existing["title"] = book["title"]

    if _author_score(book.get("author")) > _author_score(existing.get("author")):
        existing["author"] = book["author"]

    for k, v in book.items():
        if k in ("signal_type", "signal_types", "title", "author"):
            continue
        if v and not existing.get(k):
            existing[k] = v


def _clean_url(url):
    if not url:
        return ""
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return ""
    return url


# ============================================================
# OpenAI wrappers
# ============================================================

def _call_with_search(prompt, max_output_tokens=6000, max_retries=2, temperature=0.7):
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
    cached = _cache_get("disambig_v3", person_name)
    if cached is not None:
        return cached

    prompt = f"""A user wants to find books that {person_name} has personally endorsed (blurbs, forewords, etc).

If "{person_name}" is a common name shared by multiple notable people, identify them.

Pick the SINGLE person most likely to publicly endorse or recommend books. Priority order:
1. Authors, journalists, academics
2. Business executives, investors, public intellectuals
3. Athletes, artists, musicians, chefs, public figures
4. Anyone else notable

In the disambiguator (the part in parens after their name), pack in their most-famous work, role, or area of expertise — anything that would help someone reading "Ben Smith (X)" instantly know which person we mean AND what kind of books they'd be most likely to blurb. Examples:
- Ken Griffin (Citadel founder)
- Bradley Hope (journalist, co-author of 'Billion Dollar Whale')
- John Carreyrou (journalist, author of 'Bad Blood')
- Stephen King (horror novelist, author of 'The Shining')
- Yo-Yo Ma (cellist)
- Ina Garten (cookbook author and TV host)
- LeBron James (NBA player and producer)
- Anthony Bourdain (chef, author of 'Kitchen Confidential')

The disambiguator should be 3–8 words. Include their best-known work in single quotes if relevant. Cover ANY beat — sports, art, fiction, food, science, etc. — not just business/finance.

Return ONLY valid JSON, no markdown:
{{
  "primary": "Full name (rich disambiguator with beat or famous work)",
  "alternatives": ["Other notable people with this name (brief disambig each)"],
  "is_ambiguous": true or false
}}

Even if unambiguous, ALWAYS add a rich disambiguator.

Name: {person_name}"""

    text = _call_plain(prompt, max_output_tokens=500)
    data = _extract_json(text)
    primary = data.get("primary")
    if not primary:
        return {"primary": person_name, "alternatives": [], "is_ambiguous": False}

    result = {
        "primary": primary,
        "alternatives": data.get("alternatives", []),
        "is_ambiguous": bool(data.get("is_ambiguous")),
    }
    _cache_set("disambig_v3", result, person_name)
    return result


# ============================================================
# BLURB GATHER — the headline feature, optimized for recall
# ============================================================

def _gather_blurbs(name, run_idx):
    disambig_match = re.search(r"\(([^)]+)\)", name)
    disambig = disambig_match.group(1) if disambig_match else ""
    base_name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    disambig_hint = (
        f"\n\nThis person is specifically {disambig}. When searching, pair the "
        f"name with terms from their profession/company so search results aren't "
        f"flooded with the wrong person."
        if disambig else ""
    )

    prompt = f"""You are searching for BACK-COVER BLURBS, FOREWORDS, INTRODUCTIONS, JACKET QUOTES, and PRAISE PAGE quotes by {name}. Endorsements that appear physically ON or IN a book.{disambig_hint}

WHAT COUNTS — and ONLY these things count:
- Back-cover blurb attributed to {name} by name
- Dust-jacket quote attributed to {name}
- "Praise for [book]" page quote attributed to {name}
- {name} wrote the foreword for a book
- {name} wrote the introduction for a book
- Amazon "Editorial Reviews" / publisher product page quoting {name} as endorser

WHAT DOES NOT COUNT (skip these always):
- {name}'s tweets, blog posts, Substack, podcast mentions about books
- {name}'s shareholder letters, annual letters, commencement speeches
- Books on {name}'s "recommended reading list" or "favorite books" (aggregator lists)
- Books {name} reviewed in a publication
- Books that mention or profile {name} (no quote FROM them)
- Books authored or co-authored by {name}
- Biographies of {name}

THE TEST: {name}'s name and quoted endorsement text appears physically ON or IN the book — back cover, front cover, dust jacket, opening "praise for" page, retailer page's "Editorial Reviews" or "Praise" section, or foreword/introduction byline. ALL of these positions count equally. Do NOT downgrade or reject an endorsement just because you can't verify it's literally on the back cover specifically — if a publisher, retailer, or "praise for" page attributes a quote to {name} as an endorser of the book, that IS a blurb for our purposes. "Professional reviews" listed on a book's retail page (BetterWorldBooks, Amazon Editorial Reviews, Apple Books, etc.) attributed to {name} also count.

YOU MUST BE EXHAUSTIVE. The user cares MOST about not missing a real blurb. Cast a wide net. Search aggressively across many angles:
- "praise for [book]" + {name}
- "blurbed by {name}" / "endorsed by {name}"
- "foreword by {name}" / "introduction by {name}"
- Amazon editorial reviews mentioning {name}
- Publisher product pages quoting {name}
- Barnes & Noble book pages
- Google Books preview pages (back-matter section)
- Goodreads book pages quoting {name}
- BetterWorldBooks, Bookshop.org, Apple Books retailer pages
- Books in {name}'s topic area — check their "praise" pages for {name}'s name
- {name} + book titles related to their expertise

CRITICAL: also try TOPIC-SPECIFIC queries combining {name}'s name with major subjects in their beat. For example, "{name} Musk book" or "{name} crypto book" or "{name} hedge fund book" — these surface specific blurb pages that pure "{name} blurb" queries miss. Think of the major subjects {name} covers and run a separate search for each.

If {name} has a disambiguator in parens, only include endorsements from THAT specific person.

Look back 20+ years. Aim for 10+ if they exist — but only include real blurbs.

Return ONLY valid JSON (no markdown, no trailing commas):
{{
  "endorser": "{name}",
  "books": [
    {{
      "title": "Full Book Title Including Subtitle",
      "author": "Author Name (NOT {name})",
      "year": 2023,
      "one_line": "brief description of the book",
      "signal_type": "blurb | foreword | introduction | jacket_quote | praise_page",
      "quote_snippet": "first 10-20 words of {name}'s actual quote (very important)",
      "source_url": "URL where you saw this, only real https URLs",
      "notes": "any uncertainty"
    }}
  ]
}}

If you find none, return empty books array."""
    return _extract_json(_call_with_search(prompt, temperature=0.7))


# ============================================================
# CASUAL GATHER — the optional fallback
# ============================================================

def _gather_casual_shares(name, run_idx):
    disambig_match = re.search(r"\(([^)]+)\)", name)
    disambig = disambig_match.group(1) if disambig_match else ""
    disambig_hint = (
        f"\n\nThis person is specifically {disambig}. Include this context in "
        f"your searches to avoid the wrong person."
        if disambig else ""
    )

    prompt = f"""You are searching for CASUAL PERSONAL SHARES by {name} where they recommended a book on a casual public channel — the kind of post where someone finishes a book and impulsively tells the world about it.{disambig_hint}

WHAT COUNTS:
- {name}'s tweet / X post saying they read or loved a book
- {name}'s Instagram post about a book
- {name}'s personal blog post about a book they read
- {name}'s Substack or newsletter post recommending a book
- A podcast moment where {name} spontaneously said "I just read [book], it's great"
- An on-the-record interview where {name} mentioned a book they personally loved
- {name}'s LinkedIn post about a book
- {name}'s Threads / Bluesky post about a book

THE TEST: Was this a casual, in-the-moment, personal recommendation in {name}'s own voice on an informal channel? Did they sound like they finished reading and wanted to share?

WHAT DOES NOT COUNT (these are formal / curated documents, NOT casual shares):
- {name}'s shareholder letters, annual letters, founder letters
- {name}'s commencement speeches
- {name}'s formal published reading lists
- Biographer's appendices listing what {name} reads
- Aggregator listicles ("10 books CEOs love")
- "Reading lists" compiled by third parties
- Back-cover blurbs / forewords (these are blurbs, a different category)
- Books that just mention or profile {name}
- Books authored by {name}

Be EXHAUSTIVE. Famous people who tweet about books (Bezos, Naval, Patrick Collison, etc.) often have many such posts. Aim for 10+ if they exist.

Search angles:
- "{name}" tweet book
- "{name}" Twitter / X book recommendation
- "{name}" Instagram book
- "{name}" Substack
- "{name}" blog book
- "{name}" podcast "just read" OR "loved"
- "{name}" interview favorite book

If {name} has a disambiguator in parens, only include the specific person.

Return ONLY valid JSON (no markdown, no trailing commas):
{{
  "endorser": "{name}",
  "books": [
    {{
      "title": "Full Book Title Including Subtitle",
      "author": "Author Name (NOT {name})",
      "year": 2023,
      "one_line": "brief description of the book",
      "signal_type": "tweet | blog_post | substack | instagram | podcast_moment | interview_moment | social_post",
      "quote_snippet": "first 10-20 words of what {name} said",
      "source_url": "URL of the tweet/post/transcript — required for casual shares",
      "notes": "any uncertainty"
    }}
  ]
}}

If you find none, return empty books array."""
    return _extract_json(_call_with_search(prompt, temperature=0.7))


# ============================================================
# GATHER ORCHESTRATION
# ============================================================

def gather_for_mode(name, mode, log=print):
    """mode is 'blurb' or 'casual'. Caches separately."""
    cache_prefix = f"gather_v10_5_{mode}"
    cached = _cache_get(cache_prefix, name)
    if cached is not None:
        log(f"  [cache hit] {cache_prefix} for {name}")
        return cached

    if mode == "blurb":
        fn = _gather_blurbs
        runs = BLURB_RUNS
    else:
        fn = _gather_casual_shares
        runs = CASUAL_RUNS

    all_books = {}
    log(f"  Running {runs} parallel {mode} gather calls...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fn, name, i) for i in range(runs)]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                for book in result.get("books", []):
                    if not book.get("title", "").strip():
                        continue
                    _merge_book_into(all_books, book)
            except Exception as e:
                log(f"  [warning] {mode} run failed: {e}")

    final = {"endorser": name, "mode": mode, "books": list(all_books.values())}
    log(f"  Found {len(final['books'])} unique candidates")

    if final["books"]:
        _cache_set(cache_prefix, final, name)
    return final


# ============================================================
# CLASSIFY — confirms each candidate against the mode's bar
# ============================================================

def classify_candidate(name, book_title, book_author, mode, prior_signals, prior_source_url=""):
    cache_prefix = f"classify_v10_{mode}"
    cached = _cache_get(cache_prefix, name, book_title, book_author)
    if cached is not None:
        if not cached.get("source_url") and prior_source_url:
            cached = dict(cached)
            cached["source_url"] = _clean_url(prior_source_url)
        return cached

    if mode == "blurb":
        allowed = "blurb, foreword, introduction, jacket_quote, praise_page"
        rejection = "not_a_blurb"
        bar = (
            "A real BLURB-class endorsement means {name}'s name and quoted words appear "
            "physically ON or IN the book (back cover, dust jacket, praise page, foreword, "
            "or introduction). Tweets, blog posts, reading lists, and shareholder letters "
            "DO NOT count — those are 'not_a_blurb'."
        ).format(name=name)
    else:
        allowed = "tweet, blog_post, substack, instagram, podcast_moment, interview_moment, social_post"
        rejection = "not_casual"
        bar = (
            "A real CASUAL SHARE means {name} personally posted about the book in an "
            "in-the-moment, informal channel — a tweet, Instagram post, blog post, Substack, "
            "or a 'just read this' podcast/interview moment. Shareholder letters, formal "
            "reading lists, biographer compilations, and aggregator listicles DO NOT count "
            "— those are 'not_casual'. Back-cover blurbs and forewords also DO NOT count "
            "here — those are a different category."
        ).format(name=name)

    prior_hint = (
        f"\nThe gather step tagged this with signal type(s): {', '.join(prior_signals)}."
        if prior_signals else ""
    )

    prompt = f"""You are the CLASSIFIER for the next-read app.

A previous gather step found this as a likely match for {name} in {mode.upper()} mode:

Book: "{book_title}" by {book_author}{prior_hint}

{bar}

Determine via quick web search:

1. signal_type — one of: {allowed}, OR "{rejection}" if it doesn't pass the bar above, OR "unknown" if you can't tell.

2. is_author — is {name} the author/co-author? true / false

3. same_person_confidence — float 0.0–1.0. If {name} has a disambiguator, how confident are you this is from THAT specific person? No ambiguity → 1.0.

4. source_url — best real https:// URL backing this up. NEVER write prose here, only URLs or empty string.

5. quote — exact quoted words from {name} (empty if no quote).

DEFAULT TO INCLUSION. The gather step already evaluated this; only mark "{rejection}" if you find positive evidence it shouldn't count.

Return ONLY valid JSON:
{{
  "signal_type": "blurb",
  "is_author": false,
  "same_person_confidence": 1.0,
  "source_url": "",
  "quote": "",
  "notes": ""
}}"""
    text = _call_with_search(prompt, max_output_tokens=1500, temperature=0)
    result = _extract_json(text)

    def _to_bool(v, default=False):
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.lower() in ("true", "yes"):
            return True
        if isinstance(v, str) and v.lower() in ("false", "no"):
            return False
        return default

    if not result:
        final = {
            "signal_type": prior_signals[0] if prior_signals else "unknown",
            "is_author": False,
            "same_person_confidence": 0.7,
            "source_url": _clean_url(prior_source_url),
            "quote": "",
            "notes": "empty classifier — defaulting to include",
        }
    else:
        confidence = result.get("same_person_confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.7

        sig = result.get("signal_type") or (prior_signals[0] if prior_signals else "unknown")
        if sig not in KNOWN_SIGNALS:
            sig = "unknown"

        final = {
            "signal_type": sig,
            "is_author": _to_bool(result.get("is_author"), False),
            "same_person_confidence": confidence,
            "source_url": _clean_url(result.get("source_url", "")) or _clean_url(prior_source_url),
            "quote": result.get("quote", "") or "",
            "notes": result.get("notes", "") or "",
        }

    _cache_set(cache_prefix, final, name, book_title, book_author)
    return final


def _should_include(classification, mode):
    if classification.get("is_author"):
        return False, "author of book"
    if classification.get("same_person_confidence", 1.0) < MIN_SAME_PERSON_CONFIDENCE:
        return False, "different person with same name"

    sig = classification.get("signal_type", "unknown")
    accepted = BLURB_SIGNALS if mode == "blurb" else CASUAL_SIGNALS

    if sig in accepted:
        # Casual shares require a source URL — there's no physical book to point to.
        if mode == "casual" and not classification.get("source_url"):
            return False, "casual share without source URL"
        return True, "ok"

    if sig == "unknown":
        # Unknown only passes if we have a quote AND a URL — strong evidence
        if classification.get("quote") and classification.get("source_url"):
            return True, "unknown but with quote+URL"
        return False, "unknown without quote+URL"

    return False, f"signal_type={sig}"


# ============================================================
# RANK
# ============================================================

def rank(verified):
    by_key = {}
    for name, book, c in verified:
        key = _book_dedup_key(book)
        if key not in by_key:
            by_key[key] = {
                "title": book["title"],
                "author": book.get("author", "unknown"),
                "year": book.get("year"),
                "one_line": book.get("one_line", ""),
                "endorsers": [],
                "evidence": [],
                "signal_types": [],
            }
        if len(book.get("title", "")) > len(by_key[key]["title"]):
            by_key[key]["title"] = book["title"]
        if _author_score(book.get("author")) > _author_score(by_key[key]["author"]):
            by_key[key]["author"] = book["author"]
        if name not in by_key[key]["endorsers"]:
            by_key[key]["endorsers"].append(name)
        sig = c.get("signal_type", "unknown")
        if sig not in by_key[key]["signal_types"]:
            by_key[key]["signal_types"].append(sig)
        by_key[key]["evidence"].append({
            "endorser": name,
            "url": c.get("source_url", "") or book.get("source_url", ""),
            "quote": c.get("quote", "") or book.get("quote_snippet", ""),
            "signal_type": sig,
        })

    return sorted(
        by_key.values(),
        key=lambda x: (-len(x["endorsers"]), -(x.get("year") or 0)),
    )


# ============================================================
# ORCHESTRATOR
# ============================================================

def _run_mode(primary, mode, log):
    log(f"\n--- {mode.upper()} MODE ---")
    result = gather_for_mode(primary, mode, log=log)
    candidates = [(primary, b) for b in result.get("books", []) if b.get("title", "").strip()]
    log(f"  Classifying {len(candidates)} candidates...")
    verified = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_book = {
            executor.submit(
                classify_candidate,
                primary,
                book["title"],
                book.get("author", ""),
                mode,
                [book.get("signal_type", "unknown")],
                book.get("source_url", ""),
            ): book
            for _, book in candidates
        }
        for future in as_completed(future_to_book):
            book = future_to_book[future]
            try:
                c = future.result()
                include, why = _should_include(c, mode)
                if include:
                    log(f"  + '{book['title'][:60]}' [{c.get('signal_type')}]")
                    verified.append((primary, book, c))
                else:
                    log(f"  - '{book['title'][:60]}' ({why})")
            except Exception as e:
                log(f"  error: {e}")
    return rank(verified)


def recommend_from_name(person_name, include_casual=False, log=print):
    """Primary: blurb mode. If include_casual=True, also run casual gather as backup."""
    log(f"\n[1] Disambiguating '{person_name}'...")
    disambig = disambiguate_person(person_name)
    primary = disambig["primary"]
    log(f"  Resolved to: {primary}")
    if disambig.get("alternatives"):
        log(f"  (Other people with this name: {', '.join(disambig['alternatives'])})")

    blurbs = _run_mode(primary, "blurb", log)
    casual = []
    if include_casual:
        casual = _run_mode(primary, "casual", log)

    return {
        "input_name": person_name,
        "resolved_to": primary,
        "alternatives": disambig.get("alternatives", []),
        "include_casual": include_casual,
        "blurbs": blurbs,
        "casual_shares": casual,
    }


# Back-compat shim for api.py
def recommend_from_book(book_title, book_author, log=print):
    log("[info] recommend_from_book disabled in v10 — use /search/endorser")
    return {
        "input_book": {"title": book_title, "author": book_author},
        "blurbers": [],
        "recommendations": [],
    }


# ============================================================
# JSON EXTRACTION
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
    # Replace bare-word non-keyword values like `"year":unknown`
    cleaned = re.sub(
        r':\s*(?!true\b|false\b|null\b)([A-Za-z_][A-Za-z0-9_/.\-]*)\s*([,}\]])',
        r': null\2',
        cleaned,
    )
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Streaming cutoff repair
    try:
        books_match = re.search(r'"books"\s*:\s*\[', cleaned)
        if books_match:
            books_start = books_match.end()
            depth = 0
            last_complete = books_start
            in_string = False
            escape = False
            for i in range(books_start, len(cleaned)):
                ch = cleaned[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        last_complete = i + 1
            repaired = cleaned[:last_complete] + "]}"
            return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
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

    include_casual = "--casual" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--casual"]
    name = args[0] if args else "Bradley Hope"

    result = recommend_from_name(name, include_casual=include_casual)

    print("\n" + "=" * 60)
    print(f"BLURBS by {result['resolved_to']}")
    print("=" * 60)
    blurbs = result.get("blurbs", [])
    if not blurbs:
        print(f"\n{result['resolved_to']} has not written blurbs for any books, yet.")
    else:
        for i, rec in enumerate(blurbs, 1):
            year = rec.get("year") or "n/a"
            sig = ", ".join(rec.get("signal_types", []))
            print(f"\n{i}. {rec['title']} ({year})")
            print(f"   by {rec['author']}")
            print(f"   type: {sig}")
            urls = [ev["url"] for ev in rec.get("evidence", []) if ev.get("url")]
            if urls:
                print(f"   source: {urls[0]}")

    if include_casual:
        print("\n" + "=" * 60)
        print(f"CASUAL SHARES by {result['resolved_to']}")
        print("=" * 60)
        casual = result.get("casual_shares", [])
        if not casual:
            print(f"\nNo casual shares found for {result['resolved_to']}.")
        else:
            for i, rec in enumerate(casual, 1):
                year = rec.get("year") or "n/a"
                sig = ", ".join(rec.get("signal_types", []))
                print(f"\n{i}. {rec['title']} ({year})")
                print(f"   by {rec['author']}")
                print(f"   type: {sig}")
                urls = [ev["url"] for ev in rec.get("evidence", []) if ev.get("url")]
                if urls:
                    print(f"   source: {urls[0]}")

    with open("recommendations.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to recommendations.json")