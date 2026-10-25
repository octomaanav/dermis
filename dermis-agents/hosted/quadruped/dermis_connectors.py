"""Outside-world tools the agents' LLM can call: read a web page, query the live SpacetimeDB.

Pure Python + httpx, so they run in hosted agents. Both are read-only and size-capped.
"""

from __future__ import annotations

import html
import re

import httpx

import dermis_config as config
from dermis_dataset import DatasetError, _guard, _raw_url

PAGE_CHARS = 7000


def read_web_page(url: str) -> dict:
    url = _raw_url((url or "").strip())
    if not re.match(r"https?://", url):
        return {"error": "give a full http(s) URL"}
    try:
        _guard(url)
        r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "dermis-agents/0.3 (+https://github.com/Soham1010/dermis-agents)"})
        r.raise_for_status()
    except (httpx.HTTPError, DatasetError) as e:
        return {"error": f"couldn't fetch {url}: {e}"}
    ctype = r.headers.get("content-type", "")
    text = r.text
    title = ""
    if "html" in ctype or text.lstrip().lower().startswith(("<!doctype", "<html")):
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
        title = html.unescape(m.group(1).strip()) if m else ""
        main = re.search(r"<(main|article)\b[^>]*>(.*)</\1>", text, re.S | re.I)
        if main and len(main.group(2)) > 500:  # skip menus and sidebars when the page marks its content
            text = main.group(2)
        first_p = re.search(r"<p[\s>]", text, re.I)
        if first_p and first_p.start() < len(text) // 2:  # body text, not the toolbars above it
            text = text[first_p.start():]
        text = re.sub(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<br\s*/?>|</(p|div|li|h\d|tr)>", "\n", text, flags=re.I)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        text = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()
    return {"url": str(r.url), "title": title, "content_type": ctype.split(";")[0],
            "text": text[:PAGE_CHARS], "truncated": len(text) > PAGE_CHARS}


def query_live_database(sql: str) -> dict:
    """Read-only SQL against the team's SpacetimeDB (Dermis recordings as they're captured)."""
    if not config.SPACETIME_DB:
        return {"error": "the live SpacetimeDB isn't connected yet (SPACETIME_DB unset); "
                         "use the snapshot tools for the recorded data"}
    sql = (sql or "").strip().rstrip(";")
    if not re.match(r"(?is)^\s*select\b", sql) or re.search(r";|\b(insert|update|delete|drop|alter|create)\b", sql, re.I):
        return {"error": "only single read-only SELECT statements are allowed"}
    if not re.search(r"\blimit\s+\d+", sql, re.I):
        sql += " LIMIT 200"
    headers = {"Content-Type": "text/plain"}
    if config.SPACETIME_TOKEN:
        headers["Authorization"] = f"Bearer {config.SPACETIME_TOKEN}"
    try:
        r = httpx.post(f"{config.SPACETIME_HOST}/v1/database/{config.SPACETIME_DB}/sql",
                       content=sql, headers=headers, timeout=20)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return {"error": f"SpacetimeDB query failed: {e}"}
    rows = []
    for stmt in r.json():
        names = [(el.get("name") or {}).get("some") if isinstance(el.get("name"), dict) else el.get("name")
                 for el in stmt["schema"]["elements"]]
        rows.extend(dict(zip(names, row)) for row in stmt["rows"])
    return {"database": config.SPACETIME_DB, "rows": rows[:200], "row_count": len(rows)}
