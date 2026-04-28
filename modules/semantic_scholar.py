"""
Semantic Scholar Search Module
Complements ArXiv: finds high-IMPACT papers (filtered by citation count).
ArXiv  → recency signal  (2022–2026, cutting edge)
S2     → influence signal (2020–2026, citation count >= MIN_CITATIONS)

API: free, no key required (60 req/min unauthenticated, 100/min with key).
"""

import os
import time
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,year,citationCount,authors,url,openAccessPdf"

YEAR_FROM = 2020
YEAR_TO   = 2026
MIN_CITATIONS = 50        # proven influence threshold
MAX_RESULTS_PER_QUERY = 5
REQUEST_DELAY_S = 1.2     # stay under unauthenticated rate limit


def _headers() -> dict:
    token = os.getenv("S2_API_KEY", "")
    if token:
        return {"x-api-key": token}
    return {}


def _search_one(query: str) -> list[dict]:
    params = {
        "query": query,
        "fields": FIELDS,
        "year": f"{YEAR_FROM}-{YEAR_TO}",
        "minCitationCount": MIN_CITATIONS,
        "limit": MAX_RESULTS_PER_QUERY,
    }
    resp = requests.get(S2_API, params=params, headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    papers = []
    for p in data:
        pdf_url = None
        if p.get("openAccessPdf"):
            pdf_url = p["openAccessPdf"].get("url")

        papers.append({
            "title":        p.get("title", ""),
            "authors":      [a["name"] for a in p.get("authors", [])[:3]],
            "year":         p.get("year"),
            "citations":    p.get("citationCount", 0),
            "abstract":     (p.get("abstract") or "")[:400] + "..." if len(p.get("abstract") or "") > 400 else (p.get("abstract") or ""),
            "url":          p.get("url") or f"https://www.semanticscholar.org/paper/{p['paperId']}",
            "pdf_url":      pdf_url,
            "source":       "semantic_scholar",
        })

    return papers


def search_semantic_scholar(queries: list[str]) -> list[dict]:
    console.print(Panel(
        f"[bold cyan]Step 2c: Searching Semantic Scholar[/bold cyan]\n"
        f"[dim]Date range: {YEAR_FROM}–{YEAR_TO} | Min citations: {MIN_CITATIONS} | Signal: influence[/dim]"
    ))

    seen_titles: set[str] = set()
    all_papers: list[dict] = []

    for i, query in enumerate(queries):
        console.print(f"[yellow]Query:[/yellow] {query}")
        try:
            results = _search_one(query)
            new = 0
            for p in results:
                key = p["title"].lower().strip()
                if key not in seen_titles:
                    seen_titles.add(key)
                    all_papers.append(p)
                    new += 1
            console.print(f"  Found {new} new papers (≥{MIN_CITATIONS} citations)\n")
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}\n")

        if i < len(queries) - 1:
            time.sleep(REQUEST_DELAY_S)

    # Sort by citation count descending — most influential first
    all_papers.sort(key=lambda x: x["citations"], reverse=True)

    _display_papers(all_papers)
    return all_papers


def _display_papers(papers: list[dict]) -> None:
    table = Table(title="Semantic Scholar — High-Impact Papers", show_lines=True)
    table.add_column("Title", style="cyan", max_width=45)
    table.add_column("Year", style="green", width=6)
    table.add_column("Citations", style="bold yellow", width=10)
    table.add_column("URL", style="blue", max_width=40)

    for p in papers:
        table.add_row(
            p["title"],
            str(p["year"] or ""),
            str(p["citations"]),
            p["url"],
        )

    console.print(table)
    console.print(f"[bold green]Total S2 papers:[/bold green] {len(papers)}\n")
