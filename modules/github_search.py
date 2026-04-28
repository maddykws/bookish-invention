"""
GitHub Search Module
Finds existing open-source implementations relevant to the problem.
Filters: 50+ stars OR 50+ forks. Sorted by stars descending.
"""

import os
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

GITHUB_API = "https://api.github.com/search/repositories"
MIN_STARS = 50
MIN_FORKS = 50
MAX_RESULTS_PER_QUERY = 5


def _headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    return {"Accept": "application/vnd.github+json"}


def _search(query: str, extra_filter: str = "") -> list[dict]:
    full_query = f"{query} {extra_filter}".strip()
    params = {
        "q": full_query,
        "sort": "stars",
        "order": "desc",
        "per_page": MAX_RESULTS_PER_QUERY,
    }
    resp = requests.get(GITHUB_API, params=params, headers=_headers(), timeout=15)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [
        {
            "name": r["full_name"],
            "description": (r["description"] or "")[:120],
            "stars": r["stargazers_count"],
            "forks": r["forks_count"],
            "url": r["html_url"],
            "language": r.get("language", "N/A"),
            "topics": r.get("topics", [])[:4],
        }
        for r in items
        if r["stargazers_count"] >= MIN_STARS or r["forks_count"] >= MIN_FORKS
    ]


def search_github_implementations(queries: list[str]) -> list[dict]:
    console.print(Panel("[bold cyan]Step 2b: Searching GitHub for Implementations[/bold cyan]"))

    all_repos = {}

    for query in queries:
        console.print(f"[yellow]Query:[/yellow] {query}")
        try:
            results = _search(query)
            for r in results:
                if r["name"] not in all_repos:
                    all_repos[r["name"]] = r
            console.print(f"  Found {len(results)} qualifying repos (50+ stars/forks)\n")
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}\n")

    repos = sorted(all_repos.values(), key=lambda x: x["stars"], reverse=True)
    _display_repos(repos, "GitHub Repos — Implementations")
    return repos


def search_github_judge_tools(queries: list[str]) -> list[dict]:
    console.print(Panel("[bold cyan]Step 3: Searching GitHub for AI Judge / Eval Tools[/bold cyan]"))

    all_repos = {}

    for query in queries:
        console.print(f"[yellow]Query:[/yellow] {query}")
        try:
            results = _search(query)
            for r in results:
                if r["name"] not in all_repos:
                    all_repos[r["name"]] = r
            console.print(f"  Found {len(results)} qualifying repos\n")
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}\n")

    repos = sorted(all_repos.values(), key=lambda x: x["stars"], reverse=True)
    _display_repos(repos, "GitHub Repos — AI Judge & Eval Tools")
    return repos


def _display_repos(repos: list[dict], title: str):
    table = Table(title=title, show_lines=True)
    table.add_column("Repo", style="cyan", max_width=35)
    table.add_column("Stars", style="green", width=7)
    table.add_column("Forks", style="yellow", width=7)
    table.add_column("Lang", style="magenta", width=12)
    table.add_column("Description", style="white", max_width=50)

    for r in repos:
        table.add_row(
            r["name"],
            str(r["stars"]),
            str(r["forks"]),
            r["language"] or "N/A",
            r["description"],
        )

    console.print(table)
    console.print(f"[bold green]Total repos:[/bold green] {len(repos)}\n")
