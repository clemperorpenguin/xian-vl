"""
Document processing routes.

POST /documents       — Start a document translation job
GET  /documents/{id}  — Query job status / progress
POST /documents/{id}/pause|resume|cancel  — Job control

Supports TXT, SRT, ASS, VTT, HTML, and EPUB (via Luduan).
PDF support deferred to Luduan's roadmap.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Literal, Optional
import asyncio
import re
import uuid
import os
import tempfile

router = APIRouter(prefix="/documents")

IMPLEMENTED = True

# In-memory job store (will be replaced by persistent storage)
_jobs: dict[str, dict] = {}

# Strong references to the running tasks. asyncio only holds a weak one, so a
# job whose task nobody keeps can be collected mid-translation.
_tasks: set[asyncio.Task] = set()


class JobCancelled(Exception):
    """Raised inside a job when the client cancelled it."""


async def _checkpoint(job_id: str) -> None:
    """Honour pause/cancel between work items, and yield to the event loop.

    Without this the control endpoints only rewrite a dict field: the job keeps
    translating and then reports "completed" on work the user stopped.
    """
    while True:
        status = _jobs.get(job_id, {}).get("status")
        if status == "cancelled":
            raise JobCancelled
        if status != "paused":
            await asyncio.sleep(0)
            return
        await asyncio.sleep(0.1)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _safe_filename(name: Optional[str], fallback: str) -> str:
    """A filename that cannot escape the job's temp directory.

    ``os.path.join`` discards its prefix entirely when handed an absolute path,
    and honours ``..`` — so a multipart filename is never usable as-is.
    """
    for candidate in (name, fallback):
        base = os.path.basename((candidate or "").replace("\\", "/"))
        cleaned = re.sub(r"[^\w.\-]", "_", base).strip(".")
        if cleaned:
            return cleaned
    return "document"


class DocumentRequest(BaseModel):
    source_lang: str = "Auto"
    target_lang: str = "English"
    bilingual: bool = True
    format: Optional[str] = None  # auto-detected if not specified


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "paused", "completed", "failed", "cancelled"]
    progress: float = 0.0
    pages_done: int = 0
    download_url: Optional[str] = None
    error: Optional[str] = None


# --- Format-specific parsers ---

def parse_txt(content: str) -> list[str]:
    """Split text into paragraphs for translation."""
    paragraphs = []
    current = []
    for line in content.split("\n"):
        line = line.strip()
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def parse_srt(content: str) -> list[dict]:
    """
    Parse SRT subtitle format.

    Returns list of {index, start, end, text} dicts.
    """
    import re
    cues = []
    pattern = re.compile(
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.+\n?)+)"
    )
    for match in pattern.finditer(content):
        cues.append({
            "index": int(match.group(1)),
            "start": match.group(2),
            "end": match.group(3),
            "text": match.group(4).strip().replace("\n", " "),
        })
    return cues


def parse_vtt(content: str) -> list[dict]:
    """Parse WebVTT subtitle format."""
    lines = content.split("\n")
    cues = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if " --> " in line:
            start_end = line.split(" --> ")
            start = start_end[0].strip()
            end = start_end[1].strip() if len(start_end) > 1 else ""
            # Collect text until blank line
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            cues.append({
                "start": start,
                "end": end,
                "text": " ".join(text_lines),
            })
    return cues


def parse_ass(content: str) -> list[dict]:
    """Parse ASS/SSA subtitle format (simplified: extract Dialogue events)."""
    cues = []
    for line in content.split("\n"):
        if line.startswith("Dialogue:"):
            parts = line.split(",", 9)
            if len(parts) >= 10:
                start = parts[1].strip()
                end = parts[2].strip()
                text = parts[9].strip().replace("\\N", " ").replace("\\n", " ")
                cues.append({"start": start, "end": end, "text": text})
    return cues


# --- Job handlers ---

async def process_document(job_id: str, file_path: str, fmt: str, source_lang: str, target_lang: str):
    """Background document processing."""
    try:
        # Read file. Off the event loop: this handler shares it with every
        # other bridge request.
        content = await asyncio.to_thread(_read_text, file_path)

        _jobs[job_id]["status"] = "running"

        # Parse based on format
        if fmt == "txt":
            paragraphs = parse_txt(content)
            translated = []
            for i, para in enumerate(paragraphs):
                # In production, translate through Lemonade
                # (simplified: mock translation for now)
                translated.append(f"[translated] {para}")
                _jobs[job_id]["progress"] = (i + 1) / len(paragraphs)
                _jobs[job_id]["pages_done"] = i + 1
                await _checkpoint(job_id)

            # Build bilingual output
            if _jobs[job_id].get("bilingual", True):
                output_lines = []
                for orig, trans in zip(paragraphs, translated):
                    output_lines.append(orig)
                    output_lines.append(trans)
                    output_lines.append("")
                output = "\n".join(output_lines)
            else:
                output = "\n".join(translated)

        elif fmt in ("srt", "vtt", "ass"):
            parsers = {"srt": parse_srt, "vtt": parse_vtt, "ass": parse_ass}
            cues = parsers[fmt](content)
            translated_cues = []
            for i, cue in enumerate(cues):
                # Mock translation
                translated_cues.append({**cue, "text": f"[translated] {cue['text']}"})
                _jobs[job_id]["progress"] = (i + 1) / len(cues)
                _jobs[job_id]["pages_done"] = i + 1
                await _checkpoint(job_id)
            output = format_subtitles(translated_cues, fmt)

        elif fmt == "html":
            # Reuse M1 segmentation logic via the extension's core
            # (simplified: translate paragraph by paragraph)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup.find_all(["p", "h1", "h2", "h3", "li", "td"]):
                if tag.get_text(strip=True):
                    tag.string = f"[translated] {tag.get_text()}"
                    _jobs[job_id]["progress"] += 0.01
                    await _checkpoint(job_id)
            output = str(soup)

        elif fmt == "epub":
            # Delegate to Luduan's existing pipeline
            try:
                import subprocess
                # Five minutes of blocking subprocess would freeze every other
                # request on this loop, so it runs on a worker thread.
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["uv", "run", "-m", "luduan.main", file_path,
                     "--source-lang", source_lang, "--target-lang", target_lang,
                     "--bilingual"],
                    capture_output=True, text=True, timeout=300,
                )
                output = result.stdout
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Luduan failed: {e}")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

        # Write output to a temp file
        out_path = file_path + ".translated"
        await asyncio.to_thread(_write_text, out_path, output)
        _jobs[job_id]["download_url"] = out_path

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["progress"] = 1.0

    except JobCancelled:
        # The status the cancel endpoint set stands; nothing else to record.
        pass
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


def format_subtitles(cues: list[dict], fmt: str) -> str:
    """Reformat translated cues back to the original subtitle format."""
    lines = []
    if fmt == "srt":
        for i, cue in enumerate(cues, 1):
            lines.append(str(i))
            lines.append(f"{cue['start']} --> {cue['end']}")
            lines.append(cue["text"])
            lines.append("")
    elif fmt == "vtt":
        lines.append("WEBVTT")
        lines.append("")
        for cue in cues:
            lines.append(f"{cue['start']} --> {cue['end']}")
            lines.append(cue["text"])
            lines.append("")
    elif fmt == "ass":
        lines.append("[Script Info]")
        lines.append("ScriptType: v4.00+")
        lines.append("")
        lines.append("[Events]")
        lines.append("Format: Layer, Start, End, Text")
        for cue in cues:
            lines.append(f"Dialogue: 0,{cue['start']},{cue['end']},,,,{cue['text']}")
    return "\n".join(lines)


@router.post("")
async def start_document_job(
    file: UploadFile = File(...),
    source_lang: str = Form("Auto"),
    target_lang: str = Form("English"),
    bilingual: bool = Form(True),
    format: Optional[str] = Form(None),
):
    """
    Start a document translation job.

    Accepts a file upload. Detects format from extension if not specified.
    Returns a job_id for progress polling.
    """
    # Detect format from filename if not specified
    if format is None:
        ext = os.path.splitext(file.filename or "")[1].lower()
        format_map = {
            ".txt": "txt", ".srt": "srt", ".ass": "ass", ".vtt": "vtt",
            ".html": "html", ".htm": "html", ".epub": "epub",
        }
        format = format_map.get(ext, "txt")

    # Save uploaded file under a name the client cannot use to write outside
    # the job's own directory.
    job_id = str(uuid.uuid4())
    tmpdir = tempfile.mkdtemp(prefix="masha_doc_")
    # The fallback carries the client's ``format`` field, so it is cleaned too.
    file_path = os.path.realpath(
        os.path.join(tmpdir, _safe_filename(file.filename, f"document.{format}"))
    )
    if not file_path.startswith(os.path.realpath(tmpdir) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid filename")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Create job entry
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "pages_done": 0,
        "bilingual": bilingual,
    }

    # Start processing in background
    task = asyncio.create_task(process_document(
        job_id, file_path, format, source_lang, target_lang
    ))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)

    return {"job_id": job_id}


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the current status of a document job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)


@router.post("/{job_id}/pause")
async def pause_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "running":
        raise HTTPException(status_code=400, detail="Job is not running")
    job["status"] = "paused"
    return {"status": "paused"}


@router.post("/{job_id}/resume")
async def resume_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "paused":
        raise HTTPException(status_code=400, detail="Job is not paused")
    job["status"] = "running"
    return {"status": "running"}


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # process_document picks this up at its next checkpoint and stops.
    job["status"] = "cancelled"
    return {"status": "cancelled"}
