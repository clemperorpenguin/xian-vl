"""Luduan CLI — ``typer``-based command interface.

Commands
--------
translate  : Translate an EPUB file.
robobook   : Full pipeline — translate + narrate + encode.
batch      : Process a directory of EPUBs.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="luduan",
    help="Luduan — EPUB to Robobook pipeline powered by Lemonade",
    no_args_is_help=True,
)
console = Console()


@app.command()
def translate(
    input_file: Path = typer.Argument(..., help="Path to the EPUB file"),
    source_lang: str = typer.Option("Chinese", "--source", "-s"),
    target_lang: str = typer.Option("English", "--target", "-t"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o"),
) -> None:
    """Translate an EPUB file, saving intermediate JSON."""
    console.print(f"[bold]Translating:[/bold] {input_file.name}")
    console.print(f"  {source_lang} → {target_lang}")
    console.print(f"  Output: {output_dir}")
    # TODO: Wire up xian.pipeline + luduan.parser
    console.print("[yellow]Translation pipeline not yet wired — scaffold only.[/yellow]")


@app.command()
def robobook(
    input_file: Path = typer.Argument(..., help="Path to the EPUB file"),
    source_lang: str = typer.Option("Chinese", "--source", "-s"),
    target_lang: str = typer.Option("English", "--target", "-t"),
    voice: str = typer.Option("af_heart", "--voice", "-v"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o"),
) -> None:
    """Full pipeline: translate → narrate → encode to Robobook."""
    console.print(f"[bold]Robobook pipeline:[/bold] {input_file.name}")
    console.print(f"  {source_lang} → {target_lang}, voice={voice}")
    console.print(f"  Output: {output_dir}")
    # TODO: Wire up full pipeline
    console.print("[yellow]Robobook pipeline not yet wired — scaffold only.[/yellow]")


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing EPUB files"),
    source_lang: str = typer.Option("Chinese", "--source", "-s"),
    target_lang: str = typer.Option("English", "--target", "-t"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o"),
) -> None:
    """Batch-process all EPUBs in a directory."""
    epubs = sorted(input_dir.glob("*.epub"))
    console.print(f"[bold]Batch:[/bold] Found {len(epubs)} EPUB(s) in {input_dir}")
    for epub in epubs:
        console.print(f"  • {epub.name}")
    # TODO: Loop translate/robobook over each
    console.print("[yellow]Batch pipeline not yet wired — scaffold only.[/yellow]")


if __name__ == "__main__":
    app()
