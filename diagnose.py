"""Probe which queries surface the pophistorydig Hubris Maximus blurb page.

The goal: find a *general-purpose* query pattern that would surface this kind
of page (a page collecting blurbs about a recent notable book), without
hardcoding the book title.
"""
from dotenv import load_dotenv
load_dotenv()

from next_read import tavily_search

TARGET_DOMAIN = "pophistorydig.com"

# Candidate query patterns to test.
# Mix of:
#   (a) name + book-collection-style language ("praise for", "blurbs for")
#   (b) name + journalistic beat keywords (Musk, Trump, finance, fraud)
#   (c) name + site-targeting (pophistorydig is a known blurb-aggregator)
CANDIDATES = [
    # collection-style
    '"Bradley Hope" "praise for"',
    '"Bradley Hope" blurbs',
    '"Bradley Hope" book endorsement list',
    # beat-aware
    '"Bradley Hope" Musk book',
    '"Bradley Hope" Elon book blurb',
    '"Bradley Hope" Trump book',
    '"Bradley Hope" finance book endorsement',
    '"Bradley Hope" fraud book endorsement',
    # site-targeted
    '"Bradley Hope" site:pophistorydig.com',
    'Bradley Hope endorses book pophistorydig',
    # combined
    'Bradley Hope praise for book 2025',
    'Bradley Hope new nonfiction blurb 2025',
]

print(f"Testing {len(CANDIDATES)} candidate queries against Tavily...\n")
for q in CANDIDATES:
    results = tavily_search(q, 10)
    hit_position = None
    hit_url = None
    for i, r in enumerate(results):
        url = r.get("url", "")
        if TARGET_DOMAIN in url.lower() or "hubris-maximus" in url.lower():
            hit_position = i + 1
            hit_url = url
            break
    if hit_position:
        print(f"  HIT  @{hit_position}: {q}")
        print(f"       -> {hit_url}")
    else:
        # Print the top result so we see what Tavily IS returning
        top_url = results[0].get("url", "(no results)") if results else "(no results)"
        print(f"  MISS         {q}")
        print(f"       top -> {top_url}")