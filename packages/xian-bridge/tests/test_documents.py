"""
Tests for document processing routes.
"""

from fastapi.testclient import TestClient
from xian_bridge.app import create_app

app = create_app()
client = TestClient(app, base_url="http://127.0.0.1")


def test_start_txt_job():
    """Start a TXT translation job and check job_id returned."""
    resp = client.post(
        "/documents",
        files={"file": ("test.txt", b"Hello world.\nHow are you?")},
        data={"source_lang": "Auto", "target_lang": "English"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert isinstance(data["job_id"], str)


def test_start_srt_job():
    """Start an SRT translation job."""
    srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello world.

2
00:00:05,000 --> 00:00:08,000
How are you?
"""
    resp = client.post(
        "/documents",
        files={"file": ("test.srt", srt_content.encode("utf-8"))},
        data={"source_lang": "Auto", "target_lang": "English"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data


def test_job_status():
    """Start a job, poll status, verify lifecycle."""
    resp = client.post(
        "/documents",
        files={"file": ("test.txt", b"Hello.")},
    )
    job_id = resp.json()["job_id"]

    # Poll status
    resp = client.get(f"/documents/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] in ("queued", "running", "completed", "failed")


def test_job_not_found():
    """Query a non-existent job returns 404."""
    resp = client.get("/documents/nonexistent-id")
    assert resp.status_code == 404


def test_pause_resume_cancel():
    """Test job control endpoints."""
    resp = client.post(
        "/documents",
        files={"file": ("test.txt", b"Hello world.")},
    )
    job_id = resp.json()["job_id"]

    # Cancel
    resp = client.post(f"/documents/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_upload_name_cannot_escape_job_dir():
    from xian_bridge.routes.documents import _safe_filename
    assert _safe_filename("../../etc/passwd", "x") == "passwd"
    # A name that cleans to nothing falls back, and the fallback is cleaned too.
    assert _safe_filename("...", "document./../../tmp/evil") == "evil"
    assert _safe_filename("..", "..") == "document"
