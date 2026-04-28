"""
ArXiv Search Module
Finds cutting-edge research papers relevant to the problem.
Targets: architecture patterns, SOTA approaches, evaluation methods.
"""

import arxiv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

MAX_RESULTS_PER_QUERY = 5
SORT_BY = arxiv.SortCriterion.Relevance


def search_arxiv(queries: list[str]) -> list[dict]:
    console.print(Panel("[bold cyan]Step 2a: Searching ArXiv for Research[/bold cyan]"))

    client = arxiv.Client()
    all_papers = {}

    for query in queries:
        console.print(f"[yellow]Query:[/yellow] {query}")

        search = arxiv.Search(
            query=query,
            max_results=MAX_RESULTS_PER_QUERY,
            sort_by=SORT_BY,
        )

        results = list(client.results(search))

        for paper in results:
            paper_id = paper.entry_id
            if paper_id not in all_papers:
                all_papers[paper_id] = {
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors[:3]],
                    "published": paper.published.strftime("%Y-%m"),
                    "abstract": paper.summary[:400] + "..." if len(paper.summary) > 400 else paper.summary,
                    "url": paper.entry_id,
                    "categories": paper.categories,
                    "matched_query": query,
                }

        console.print(f"  Found {len(results)} papers\n")

    papers = list(all_papers.values())

    _display_papers(papers)
    return papers


def _display_papers(papers: list[dict]):
    table = Table(title="ArXiv Papers Found", show_lines=True)
    table.add_column("Title", style="cyan", max_width=50)
    table.add_column("Published", style="green", width=10)
    table.add_column("Categories", style="yellow", max_width=25)
    table.add_column("URL", style="blue", max_width=40)

    for p in papers:
        table.add_row(
            p["title"],
            p["published"],
            ", ".join(p["categories"][:2]),
            p["url"],
        )

    console.print(table)
    console.print(f"[bold green]Total unique papers:[/bold green] {len(papers)}\n")
