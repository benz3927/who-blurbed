"""
NextRead API v5 — wraps the v10 blurb-first pipeline.

Two endpoints:
  POST /search/endorser  body: {"name": "...", "include_casual": false}
  GET  /health

Auth: header x-api-key must match NEXTREAD_API_KEY env var.
Run: uvicorn backend_api:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import urllib.parse

from next_read import recommend_from_name

app = FastAPI(title="NextRead API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("NEXTREAD_API_KEY", "changeme")


def require_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="bad key")
    return True


def _amazon_link(title: str, author: str) -> str:
    if not author or "unknown" in author.lower():
        q = urllib.parse.quote_plus(title)
    else:
        q = urllib.parse.quote_plus(f"{title} {author}")
    return f"https://www.amazon.com/s?k={q}&i=stripbooks"


def _to_swift_rec(rec: dict) -> dict:
    """Map a v10 ranked recommendation dict to the iOS Recommendation model."""
    title = rec.get("title", "")
    author = rec.get("author") or ""

    # Find the longest quote and its source URL
    evidence = rec.get("evidence") or []
    best_quote = ""
    best_url = ""
    for ev in evidence:
        q = ev.get("quote", "") or ""
        u = ev.get("url", "") or ""
        if u and not (u.startswith("http://") or u.startswith("https://")):
            u = ""
        if len(q) > len(best_quote):
            best_quote = q
            best_url = u
        elif not best_url and u:
            best_url = u

    return {
        "book_title": title,
        "book_author": author or "—",
        "year": rec.get("year"),
        "blurb_excerpt": best_quote or rec.get("one_line", ""),
        "amazon_url": _amazon_link(title, author),
        "source_url": best_url,
        "endorsers": rec.get("endorsers", []),
        "signal_types": rec.get("signal_types", []),
    }


class EndorserQuery(BaseModel):
    name: str
    include_casual: Optional[bool] = False


@app.get("/health")
def health():
    return {"status": "ok", "version": "5.0"}


@app.post("/search/endorser", dependencies=[Depends(require_key)])
def search_endorser(q: EndorserQuery):
    try:
        result = recommend_from_name(
            q.name,
            include_casual=bool(q.include_casual),
            log=lambda *a, **kw: None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")

    blurbs: List[dict] = [_to_swift_rec(r) for r in result.get("blurbs", [])]
    casual: List[dict] = [_to_swift_rec(r) for r in result.get("casual_shares", [])]

    return {
        "query": q.name,
        "resolved_to": result.get("resolved_to"),
        "alternatives": result.get("alternatives", []),
        "include_casual": bool(q.include_casual),
        "blurbs": blurbs,
        "casual_shares": casual,
        # For backwards compat with the old iOS app that expects "results":
        "results": blurbs,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)