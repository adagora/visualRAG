"""Bounded PNG/JPEG/WebP reference images, normalized to PNG."""
from __future__ import annotations
import base64
import binascii
import io
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000
MIMES = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}


def _normalize(raw: bytes, expected: str | None = None) -> bytes:
    if len(raw) > MAX_BYTES:
        raise ValueError("The reference image must be at most 5 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in MIMES.values() or (expected and source.format != expected):
                raise ValueError("Use a PNG, JPEG or WebP image matching its declared format.")
            if source.width * source.height > MAX_PIXELS or min(source.size) < 16:
                raise ValueError("Use sides at least 16 px and at most 16 million pixels.")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            while True:
                output = io.BytesIO()
                image.save(output, format="PNG")
                if output.tell() <= MAX_BYTES:
                    return output.getvalue()
                if min(image.size) <= 32:
                    raise ValueError("The normalized image is too large.")
                image = image.resize((max(16, round(image.width * .8)), max(16, round(image.height * .8))), Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The reference image could not be decoded.") from exc


def decode_image(data_url: str | None) -> bytes | None:
    if not data_url:
        return None
    if not isinstance(data_url, str):
        raise ValueError("Use a PNG, JPEG or WebP data URL.")
    header, separator, payload = data_url.partition(",")
    expected = next((fmt for mime, fmt in MIMES.items() if header == f"data:{mime};base64"), None)
    if not separator or expected is None:
        raise ValueError("Use a PNG, JPEG or WebP data URL.")
    if len(payload) > ((MAX_BYTES + 2) // 3) * 4:
        raise ValueError("The reference image must be at most 5 MB.")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The reference image could not be decoded.") from exc
    return _normalize(raw, expected)


def read_image(path: Path | str) -> bytes:
    with Path(path).open("rb") as source:
        return _normalize(source.read(MAX_BYTES + 1))
