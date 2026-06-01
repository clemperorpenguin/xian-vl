"""Luduan CLI — ``typer``-based command interface.

Commands
--------
translate  : Translate an EPUB file.
robobook   : Full pipeline — translate + narrate + encode.
batch      : Process a directory of EPUBs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path

import typer
from rich.console import Console

from shared_types.constants import DEFAULT_API_URL
from xian.omni_router import OmniModelRouter
from luduan.parser import parse_epub, ParsedBook
from luduan.translator import DocumentTranslator
from luduan.audio_engine import AudioEngine, concatenate_wavs, get_wav_duration_ms
from luduan.encoder import AudioSegment, encode_opus, generate_koreader_manifest, write_manifest

app = typer.Typer(
    name="luduan",
    help="Luduan — EPUB to RoboBook pipeline powered by Lemonade",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger("luduan")


def write_translated_markdown(book: ParsedBook, translated_texts: list[str], output_path: Path) -> None:
    lines = [f"# {book.title}"]
    if book.author:
        lines.append(f"**Author:** {book.author}")
    lines.append("")

    current_chapter_idx = -1
    for p, trans_text in zip(book.paragraphs, translated_texts):
        if p.chapter_index != current_chapter_idx:
            current_chapter_idx = p.chapter_index
            lines.append("")
        if p.is_heading:
            lines.append(f"## {trans_text}")
        else:
            lines.append(trans_text)
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def compile_robobook_chapters(
    book: ParsedBook,
    translated_texts: list[str],
    segments: list[AudioSegment] | None = None
) -> list[dict]:
    # Group paragraphs and translations by chapter_index
    chapters_dict = {}

    for idx, (p, trans) in enumerate(zip(book.paragraphs, translated_texts)):
        ch_idx = p.chapter_index
        if ch_idx not in chapters_dict:
            chapters_dict[ch_idx] = {
                "index": ch_idx,
                "title": "",
                "original_paras": [],
                "translated_paras": [],
                "audio_offset_ms": 0,
                "audio_duration_ms": 0,
                "initialized": False
            }

        if p.is_heading:
            chapters_dict[ch_idx]["title"] = trans

        chapters_dict[ch_idx]["original_paras"].append(p.text)
        chapters_dict[ch_idx]["translated_paras"].append(trans)

        if segments:
            seg = segments[idx]
            if not chapters_dict[ch_idx]["initialized"]:
                chapters_dict[ch_idx]["audio_offset_ms"] = seg.offset_ms
                chapters_dict[ch_idx]["initialized"] = True
            chapters_dict[ch_idx]["audio_duration_ms"] += seg.duration_ms

    out_chapters = []
    for ch_idx in sorted(chapters_dict.keys()):
        ch = chapters_dict[ch_idx]
        out_chapters.append({
            "index": ch["index"],
            "title": ch["title"],
            "original_text": "\n\n".join(ch["original_paras"]),
            "translated_text": "\n\n".join(ch["translated_paras"]),
            "audio_offset_ms": ch["audio_offset_ms"],
            "audio_duration_ms": ch["audio_duration_ms"]
        })
    return out_chapters


async def run_translate(input_file: Path, source_lang: str, target_lang: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse book
    book = parse_epub(input_file)
    if not book.paragraphs:
        console.print("[bold red]No text content found in EPUB.[/bold red]")
        return

    # 2. Discover models
    router = OmniModelRouter(DEFAULT_API_URL)
    with console.status("[bold blue]Discovering Lemonade Models...[/bold blue]"):
        await router.discover_async()

    # 3. Translate
    translator = DocumentTranslator(
        source_lang=source_lang,
        target_lang=target_lang,
        router=router
    )

    passages = [p.text for p in book.paragraphs]
    with console.status(f"[bold blue]Translating {len(passages)} paragraphs...[/bold blue]"):
        translated_texts = await translator.translate_batch(passages, concurrency=8)

    # 4. Save Markdown output
    md_path = output_dir / f"{input_file.stem}_translated.md"
    write_translated_markdown(book, translated_texts, md_path)
    console.print(f"[green]Saved translated Markdown to {md_path}[/green]")

    # 5. Save JSON sidecar
    json_path = output_dir / f"{input_file.stem}_translated.json"
    chapters = compile_robobook_chapters(book, translated_texts)
    json_path.write_text(json.dumps(chapters, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved structured JSON to {json_path}[/green]")


async def run_robobook(input_file: Path, source_lang: str, target_lang: str, voice: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse book
    book = parse_epub(input_file)
    if not book.paragraphs:
        console.print("[bold red]No text content found in EPUB.[/bold red]")
        return

    # 2. Discover models
    router = OmniModelRouter(DEFAULT_API_URL)
    with console.status("[bold blue]Discovering Lemonade Models...[/bold blue]"):
        await router.discover_async()

    # 3. Translate
    translator = DocumentTranslator(
        source_lang=source_lang,
        target_lang=target_lang,
        router=router
    )

    passages = [p.text for p in book.paragraphs]
    with console.status(f"[bold blue]Translating {len(passages)} paragraphs...[/bold blue]"):
        translated_texts = await translator.translate_batch(passages, concurrency=8)

    # 4. Narrate (TTS)
    audio_engine = AudioEngine(voice=voice, router=router)
    segments = []
    wav_datas = []
    offset_ms = 0

    try:
        with console.status("[bold blue]Generating audiobook narration...[/bold blue]") as status:
            for idx, (p, trans_text) in enumerate(zip(book.paragraphs, translated_texts)):
                status.update(f"[bold blue]Synthesizing speech: paragraph {idx+1}/{len(passages)}...[/bold blue]")
                try:
                    wav_bytes = await audio_engine.synthesize(trans_text)
                    duration_ms = get_wav_duration_ms(wav_bytes)
                except Exception as e:
                    logger.error("TTS failed for paragraph %d: %s", idx, e)
                    wav_bytes = b""
                    duration_ms = 0.0

                segments.append(AudioSegment(
                    chapter_index=p.chapter_index,
                    paragraph_index=idx,
                    text=trans_text,
                    wav_path=Path(""),
                    offset_ms=int(offset_ms),
                    duration_ms=int(duration_ms)
                ))
                if wav_bytes:
                    wav_datas.append(wav_bytes)
                offset_ms += duration_ms

        # 5. Concatenate and Encode Opus
        if not wav_datas:
            console.print("[bold red]Failed to generate narration audio.[/bold red]")
            return

        success = False
        opus_filename = f"{input_file.stem}_translated.opus"
        opus_path = output_dir / opus_filename
        
        with console.status("[bold blue]Encoding audio to Opus...[/bold blue]"):
            combined_wav = concatenate_wavs(wav_datas)
            # Create a secure temp WAV file to encode
            fd, tmp_wav_name = tempfile.mkstemp(suffix=".wav")
            tmp_wav_path = Path(tmp_wav_name)
            try:
                tmp_wav_path.write_bytes(combined_wav)
                success = encode_opus(tmp_wav_path, opus_path)
            finally:
                import os
                try:
                    os.close(fd)
                    tmp_wav_path.unlink()
                except Exception:
                    pass

        if success:
            console.print(f"[green]Saved Opus audio to {opus_path}[/green]")
        else:
            console.print("[bold red]Opus encoding failed.[/bold red]")

        # 6. Save KOReader manifest
        manifest_filename = f"{input_file.stem}_translated.audio.json"
        manifest_path = output_dir / manifest_filename
        manifest = generate_koreader_manifest(segments, opus_filename)
        write_manifest(manifest, manifest_path)
        console.print(f"[green]Saved KOReader manifest to {manifest_path}[/green]")

        # 7. Save Markdown translation
        md_path = output_dir / f"{input_file.stem}_translated.md"
        write_translated_markdown(book, translated_texts, md_path)
        console.print(f"[green]Saved translated Markdown to {md_path}[/green]")

        # 8. Save structured JSON
        json_path = output_dir / f"{input_file.stem}_translated.json"
        chapters = compile_robobook_chapters(book, translated_texts, segments)
        json_path.write_text(json.dumps(chapters, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]Saved structured RoboBook JSON to {json_path}[/green]")

    finally:
        await audio_engine.close()


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
    asyncio.run(run_translate(input_file, source_lang, target_lang, output_dir))


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
    asyncio.run(run_robobook(input_file, source_lang, target_lang, voice, output_dir))


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing EPUB files"),
    source_lang: str = typer.Option("Chinese", "--source", "-s"),
    target_lang: str = typer.Option("English", "--target", "-t"),
    voice: str = typer.Option("af_heart", "--voice", "-v"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o"),
) -> None:
    """Batch-process all EPUBs in a directory."""
    epubs = sorted(input_dir.glob("*.epub"))
    console.print(f"[bold]Batch:[/bold] Found {len(epubs)} EPUB(s) in {input_dir}")
    for epub in epubs:
        console.print(f"  • {epub.name}")
        asyncio.run(run_robobook(epub, source_lang, target_lang, voice, output_dir))


if __name__ == "__main__":
    app()
