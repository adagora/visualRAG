import base64
import io
from pathlib import Path
from PIL import Image
import pytest
import visual_query as vq


def image_bytes(fmt="PNG", size=(40, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize("mime,fmt", [("image/png", "PNG"), ("image/jpeg", "JPEG"), ("image/webp", "WEBP")])
def test_supported_formats_become_png(tmp_path, mime, fmt):
    raw = image_bytes(fmt)
    result = vq.decode_image("data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii"))
    assert result.startswith(b"\x89PNG")
    path = tmp_path / "reference"
    path.write_bytes(raw)
    assert vq.read_image(path) == result


def test_empty_optional_image():
    assert vq.decode_image(None) is None
    assert vq.decode_image("") is None


@pytest.mark.parametrize("value", ["http://example.com/image.png", "data:image/png;base64,!!", "data:image/gif;base64,AAAA", "data:image/png;base64,AAAA"])
def test_bad_data_urls(value):
    with pytest.raises(ValueError):
        vq.decode_image(value)


def test_size_and_type_limits(tmp_path):
    path = tmp_path / "reference"
    for raw in (b"x" * (vq.MAX_BYTES + 1), image_bytes(size=(15, 100)), image_bytes("GIF"), image_bytes(size=(4100, 4100))):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            vq.read_image(path)
    with pytest.raises(ValueError):
        vq.decode_image("data:image/png;base64," + "A" * (7 * 1024 * 1024))


def test_mime_must_match_bytes():
    value = "data:image/jpeg;base64," + base64.b64encode(image_bytes("PNG")).decode("ascii")
    with pytest.raises(ValueError, match="declared format"):
        vq.decode_image(value)
