from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
from io import BytesIO
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, HttpUrl
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter

APP_TOKEN = os.getenv("DOCLING_WORKER_TOKEN", "").strip()
MAX_BYTES = int(os.getenv("DOCLING_MAX_BYTES", str(20 * 1024 * 1024)))
MAX_PAGES = int(os.getenv("DOCLING_MAX_PAGES", "120"))
TIMEOUT_SECONDS = float(os.getenv("DOCLING_FETCH_TIMEOUT", "25"))

app = FastAPI(title="TarlaPusula Docling Worker", version="1.0.1")
converter = DocumentConverter(allowed_formats=[InputFormat.PDF])


class ConvertRequest(BaseModel):
    url: HttpUrl


def require_token(authorization: str | None) -> None:
    if not APP_TOKEN:
        raise HTTPException(503, "Worker token is not configured")
    if authorization != f"Bearer {APP_TOKEN}":
        raise HTTPException(401, "Unauthorized")


def ensure_public_url(raw_url: str) -> None:
    parsed = urlparse(raw_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise HTTPException(400, "Only public HTTPS document URLs are allowed")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise HTTPException(400, "Document host could not be resolved")
    for entry in addresses:
        if not ipaddress.ip_address(entry[4][0]).is_global:
            raise HTTPException(400, "Private or local document hosts are not allowed")


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "tarlapusula-docling",
        "max_pages": MAX_PAGES,
        "max_bytes": MAX_BYTES,
    }


@app.post("/convert")
async def convert_document(payload: ConvertRequest, authorization: str | None = Header(default=None)):
    require_token(authorization)
    source_url = str(payload.url)
    ensure_public_url(source_url)

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False) as client:
            response = await client.get(source_url, headers={"User-Agent": "TarlaPusula-Docling/1.0"})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Document fetch failed: {type(exc).__name__}")

    if response.status_code != 200:
        raise HTTPException(502, f"Document fetch returned HTTP {response.status_code}")

    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    body = response.content

    if len(body) > MAX_BYTES:
        raise HTTPException(413, "Document is too large")
    if content_type not in {"application/pdf", "application/octet-stream"} and not body.startswith(b"%PDF"):
        raise HTTPException(415, "Only PDF documents are accepted")

    digest = hashlib.sha256(body).hexdigest()

    try:
        stream = DocumentStream(name=f"{digest}.pdf", stream=BytesIO(body))
        result = converter.convert(stream, max_num_pages=MAX_PAGES)
        document = result.document
        markdown = document.export_to_markdown()
        structured = document.export_to_dict()
    except Exception as exc:
        raise HTTPException(422, f"Docling conversion failed: {type(exc).__name__}")

    return {
        "ok": True,
        "sha256": digest,
        "source_url": source_url,
        "content_type": "application/pdf",
        "markdown": markdown,
        "document": structured,
    }
