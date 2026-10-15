"""Validate image queries and encode mixed inputs with the indexed Qwen model."""
from __future__ import annotations

import base64
import binascii
import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000


def decode_image(value: str | None) -> bytes | None:
    """Accept a bounded browser data URL; normalize orientation and format."""
    if not value:
        return None
    if not value.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
        raise ValueError("Use a PNG, JPEG or WebP screenshot.")
    payload = value.split(",", 1)[1]
    if len(payload) > MAX_BYTES * 4 // 3 + 8:
        raise ValueError("The screenshot must be at most 5 MB.")
    try:
        raw = base64.b64decode(payload, validate=True)
        if len(raw) > MAX_BYTES:
            raise ValueError("The screenshot must be at most 5 MB.")
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > MAX_PIXELS or min(image.size) < 16:
                raise ValueError("Use an image with sides at least 16 px and at most 16 million pixels.")
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            image.save(out, format="PNG")
            while out.tell() > MAX_BYTES and min(image.size) > 32:
                image = image.resize((max(16, round(image.width * .8)), max(16, round(image.height * .8))), Image.Resampling.LANCZOS)
                out = io.BytesIO()
                image.save(out, format="PNG")
            if out.tell() > MAX_BYTES:
                raise ValueError("The normalized screenshot is too large.")
            return out.getvalue()
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The screenshot could not be decoded.") from exc


def data_url(image: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def encode(state: dict, text: str, image: bytes, instruction: str) -> list[float]:
    """Run ungraphed: screenshot shapes differ from the captured text shapes."""
    torch, processor, model = state["torch"], state["processor"], state["model"]
    with Image.open(io.BytesIO(image)) as source:
        source = source.convert("RGB")
        content = [{"type": "image", "image": source}]
        if text.strip():
            content.append({"type": "text", "text": text})
        messages = [{"role": "system", "content": [{"type": "text", "text": instruction}]},
                    {"role": "user", "content": content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[source], return_tensors="pt", padding=True)
    inputs = {key: value.to(state["device"]) if hasattr(value, "to") else value
              for key, value in inputs.items()}
    forward = getattr(model, "_ungraphed_model", model.model)
    with torch.inference_mode():
        hidden = forward(**inputs).last_hidden_state
        last = inputs["attention_mask"].sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        vector = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return vector.cpu().float().numpy()[0].tolist()
