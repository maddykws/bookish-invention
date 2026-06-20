"""
ArXiv Search Module
Finds cutting-edge research papers relevant to the problem.
Date range: 2022–2026 (recent architectures, implementable approaches).
For foundational/high-citation work, see semantic_scholar.py (2020–2026).
"""

import arxiv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

YEAR_FROM = 2022
YEAR_TO   = 2026
MAX_RESULTS_PER_QUERY = 8   # fetch more, date-filter brings it down
SORT_BY = arxiv.SortCriterion.Relevance


def _build_query(query: str) -> str:
    # ArXiv query syntax for date range (YYYYMMDD format)
    return f"({query}) AND submittedDate:[{YEAR_FROM}0101 TO {YEAR_TO}1231]"


def search_arxiv(queries: list[str]) -> list[dict]:
    console.print(Panel(
        f"[bold cyan]Step 2a: Searching ArXiv[/bold cyan]\n"
        f"[dim]Date range: {YEAR_FROM}–{YEAR_TO} | Signal: recency + relevance[/dim]"
    ))

    client = arxiv.Client()
    all_papers: dict[str, dict] = {}

    for query in queries:
        console.print(f"[yellow]Query:[/yellow] {query}")

        search = arxiv.Search(
            query=_build_query(query),
            max_results=MAX_RESULTS_PER_QUERY,
            sort_by=SORT_BY,
        )

        results = list(client.results(search))

        added = 0
        for paper in results:
            paper_id = paper.entry_id
            year = paper.published.year
            if paper_id not in all_papers and YEAR_FROM <= year <= YEAR_TO:
                all_papers[paper_id] = {
                    "title":         paper.title,
                    "authors":       [a.name for a in paper.authors[:3]],
                    "published":     paper.published.strftime("%Y-%m"),
                    "year":          year,
                    "abstract":      paper.summary[:400] + "..." if len(paper.summary) > 400 else paper.summary,
                    "url":           paper.entry_id,
                    "categories":    paper.categories,
                    "matched_query": query,
                    "source":        "arxiv",
                }
                added += 1

        console.print(f"  Found {added} papers ({YEAR_FROM}–{YEAR_TO})\n")

    papers = list(all_papers.values())
    _display_papers(papers)
    return papers


def _display_papers(papers: list[dict]) -> None:
    table = Table(title=f"ArXiv Papers ({YEAR_FROM}–{YEAR_TO})", show_lines=True)
    table.add_column("Title", style="cyan", max_width=48)
    table.add_column("Published", style="green", width=10)
    table.add_column("Categories", style="yellow", max_width=22)
    table.add_column("URL", style="blue", max_width=38)

    for p in papers:
        table.add_row(
            p["title"],
            p["published"],
            ", ".join(p["categories"][:2]),
            p["url"],
        )

    console.print(table)
    console.print(f"[bold green]Total unique ArXiv papers:[/bold green] {len(papers)}\n")
