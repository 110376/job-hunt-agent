from __future__ import annotations

from datetime import datetime

from rich.console import Console


_CONSOLE = Console()

_TAG_STYLE = {
    "input": "bold cyan",
    "agent": "bold magenta",
    "tool": "bold blue",
    "tool-result": "bold green",
    "llm": "bold yellow",
    "state": "bold bright_blue",
    "warn": "bold red",
    "done": "bold green",
}


def log(tag: str, message: str) -> None:
    style = _TAG_STYLE.get(tag, "white")
    now = datetime.now().strftime("%H:%M:%S")
    _CONSOLE.print(f"[{tag}] {now} {message}", style=style, markup=False, highlight=False)
