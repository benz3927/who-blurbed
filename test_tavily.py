import os
from dotenv import load_dotenv
load_dotenv()

key = os.environ.get("TAVILY_API_KEY", "")
print(f"Key loaded: {bool(key)}, starts with: {key[:8]}, length: {len(key)}")

from tavily import TavilyClient
tavily = TavilyClient(api_key=key)
result = tavily.search(query="Bradley Hope book blurb", max_results=5)
print(f"Got {len(result.get('results', []))} results")
for r in result.get("results", [])[:3]:
    print(f"- {r.get('title', '')[:80]}")
    print(f"  {r.get('content', '')[:150]}")