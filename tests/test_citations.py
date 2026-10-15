"""Real PDF text, crop/rotation and image-only citation locations; no API calls."""
import io

import numpy as np
from PIL import Image
import pymupdf
import pytest

from citations import locate


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("cropped", [False, True])
def test_quote_highlight_covers_rendered_text_after_crop_and_rotation(tmp_path, rotation, cropped):
    path = tmp_path / "warranty.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((100, 150), "Gwarancja trwa 5 lat.", fontsize=18)
        if cropped:
            page.set_cropbox(pymupdf.Rect(40, 60, 540, 700))
        page.set_rotation(rotation)
        document.save(path)
    result = locate(path, 1, "Gwarancja trwa 5 lat.")
    assert result["highlight_status"] == "matched"
    assert len(result["rects"]) == 1
    with pymupdf.open(path) as document:
        image = document[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY)
    pixels = np.frombuffer(image.samples, dtype=np.uint8).reshape(image.height, image.width)
    # The page contains only this text: its visible ink must fall in the highlight.
    ys, xs = np.where(pixels < 128)
    rect = result["rects"][0]
    assert rect["left"] * image.width <= xs.min() + 1
    assert (rect["left"] + rect["width"]) * image.width >= xs.max() - 1
    assert rect["top"] * image.height <= ys.min() + 1
    assert (rect["top"] + rect["height"]) * image.height >= ys.max() - 1
    assert rect["width"] * rect["height"] < .08


def test_wrapped_hyphenated_quote_has_one_highlight_per_line(tmp_path):
    path = tmp_path / "wrapped.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((60, 100), "Warranty cov-\nerage lasts five years.", fontsize=16)
        document.save(path)
    result = locate(path, 1, "Warranty coverage lasts five years.")
    assert result["highlight_status"] == "matched"
    assert len(result["rects"]) == 2
    assert result["rects"][0]["top"] < result["rects"][1]["top"]


def test_image_only_source_uses_approximate_reader_box(tmp_path):
    path = tmp_path / "scan.pdf"
    image = io.BytesIO()
    Image.new("RGB", (400, 500), "white").save(image, format="PNG")
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=500)
        page.insert_image(page.rect, stream=image.getvalue())
        document.save(path)
    result = locate(path, 1, "Gwarancja trwa 5 lat.",
                    {"left": .1, "top": .2, "right": .6, "bottom": .4})
    assert result["highlight_status"] == "approximate"
    assert result["rects"] == [{"left": .1, "top": .2, "width": .5, "height": .2}]


@pytest.mark.parametrize("box", [None, [], {"left": 0},
    {"left": True, "top": 0, "right": 1, "bottom": 1},
    {"left": 0, "top": 0, "right": float("nan"), "bottom": 1},
    {"left": 0, "top": 0, "right": float("inf"), "bottom": 1},
    {"left": .7, "top": 0, "right": .2, "bottom": 1},
    {"left": 0, "top": 0, "right": 1.01, "bottom": 1},
    {"left": 0, "top": 0, "right": 0, "bottom": 1}])
def test_invalid_or_missing_location_keeps_page_without_fabricated_highlight(box):
    result = locate(None, 1, "Unsupported quote", box)
    assert result["highlight_status"] == "unavailable"
    assert result["rects"] == []


def test_unreadable_pdf_keeps_valid_approximate_region(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"broken")
    result = locate(path, 1, "Quote", {"left": 0, "top": .1, "right": .5, "bottom": .3})
    assert result["highlight_status"] == "approximate"
