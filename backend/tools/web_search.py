# backend/tools/web_search.py

from __future__ import annotations

from typing import Any, Dict, List

from ddgs import DDGS

from backend import config


def web_search(query: str, top_k: int | None = None) -> Dict[str, Any]:
    """
    Returns:
      {
        "query": str,
        "results": [
          {
            "title": str,
            "url": str,
            "snippet": str,
            "source": "web"
          }, ...
        ],
        "count": int
      }
    """
    k = top_k or config.WEB_TOP_K

    if not config.ENABLE_WEB_SEARCH:
        return {"query": query, "results": [], "count": 0}

    results: List[Dict[str, str]] = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k):
                title = (r.get("title") or "").strip()
                url = (r.get("href") or "").strip()
                snippet = (r.get("body") or "").strip()

                if not title or not url:
                    continue

                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": "web",
                    }
                )

                if len(results) >= k:
                    break
                
# register error in the pipeline insted 

    except Exception as e:
        return {
            "query": query,
            "results": [],
            "count": 0,
            "error": f"web_search_failed: {type(e).__name__}: {e}",
        }

    return {"query": query, "results": results, "count": len(results)}