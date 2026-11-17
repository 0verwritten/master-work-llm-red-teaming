from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from preliminary_study.protocol import Protocol

FONT_PATH = Path(__file__).parent / "assets" / "DejaVuSans.ttf"
FONT_SHA256 = "7da195a74c55bef988d0d48f9508bd5d849425c1770dba5d7bfc6ce9ed848954"
RENDERER_VERSION = "preliminary-study-plain-typographic-v1"

MIN_INK_FRACTION = 0.002


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderedImage:
    png_bytes: bytes
    width: int
    height: int
    lines: int
    ink_fraction: float

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.png_bytes).hexdigest()


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.exists():
        raise RenderError(f"vendored font missing at {FONT_PATH}")
    actual = hashlib.sha256(FONT_PATH.read_bytes()).hexdigest()
    if actual != FONT_SHA256:
        raise RenderError(f"font sha256 {actual} != pinned {FONT_SHA256}")
    return ImageFont.truetype(str(FONT_PATH), size)


def render_payload(payload: str, protocol: Protocol) -> RenderedImage:

    font = _load_font(protocol.image_font_size)
    wrapped = textwrap.fill(payload.strip(), width=protocol.image_wrap_columns)

    canvas = Image.new(
        "RGB", (protocol.image_width, protocol.image_height), protocol.image_background
    )
    draw = ImageDraw.Draw(canvas)

    left, top, right, bottom = draw.multiline_textbbox((0, 0), wrapped, font=font, align="left")
    text_width = right - left
    text_height = bottom - top
    if text_width > protocol.image_width or text_height > protocol.image_height:
        raise RenderError(
            f"wrapped payload needs {text_width}x{text_height}px but the canvas is "
            f"{protocol.image_width}x{protocol.image_height}px; it would be clipped"
        )

    origin = (
        (protocol.image_width - text_width) // 2 - left,
        (protocol.image_height - text_height) // 2 - top,
    )
    draw.multiline_text(origin, wrapped, fill=protocol.image_foreground, font=font, align="left")

    ink_fraction = _ink_fraction(canvas, protocol.image_background)
    if ink_fraction < MIN_INK_FRACTION:
        raise RenderError(f"render carries almost no visible text (ink {ink_fraction:.5f})")

    buffer = BytesIO()

    canvas.save(buffer, format="PNG", compress_level=6)

    return RenderedImage(
        png_bytes=buffer.getvalue(),
        width=protocol.image_width,
        height=protocol.image_height,
        lines=wrapped.count("\n") + 1,
        ink_fraction=ink_fraction,
    )


def _ink_fraction(canvas: Image.Image, background: str) -> float:
    background_rgb = tuple(int(background[index : index + 2], 16) for index in (1, 3, 5))
    pixels = list(canvas.getdata())
    if not pixels:
        return 0.0
    inked = sum(1 for pixel in pixels if pixel != background_rgb)
    return inked / len(pixels)
