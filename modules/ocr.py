"""
OCR module — uses Ollama with scb10x/typhoon-ocr vision model.

Performance optimisations (vs. previous version):
- Image thumbnail capped at 768 px (was 1024) → fewer vision tokens
- PDF render at 150 DPI (was 200) → smaller images, faster convert_from_path
- num_ctx reduced to 2048 (was 4096) → less VRAM, faster inference
- max_tokens reduced to 1024 (was 2048) → shorter generation
- keep_alive="5m" → keeps model loaded between consecutive pages
"""

import base64
from pathlib import Path

from openai import OpenAI

import config

# ── Tuning knobs ──────────────────────────────────────────
_MAX_IMAGE_PX: int = 768         # thumbnail max px (use 512 for very low-RAM)
_PDF_DPI: int = 150              # DPI for pdf → image conversion
_NUM_CTX: int = 2048             # Ollama context window size
_MAX_TOKENS: int = 1024          # max output tokens per page
_KEEP_ALIVE: str = "5m"         # keep model in VRAM between calls


def _get_client() -> OpenAI:
    """Create OpenAI-compatible client pointing at Ollama."""
    return OpenAI(
        api_key="ollama",  # Ollama ignores the key but the SDK requires one
        base_url=f"{config.OLLAMA_BASE_URL}/v1",
    )


def ocr_image(image_path: str) -> str:
    """
    Run OCR on a single image file via Ollama vision model.

    Args:
        image_path: Path to the image file.

    Returns:
        Extracted text as a string.
    """
    from PIL import Image
    import io

    path = Path(image_path)

    # Resize image to save VRAM on tokenization
    with Image.open(path) as img:
        img.thumbnail((_MAX_IMAGE_PX, _MAX_IMAGE_PX))
        # Handle formats correctly, fallback to PNG
        fmt = img.format if img.format else "PNG"
        if img.mode in ("RGBA", "P") and fmt == "JPEG":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format=fmt)
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Detect MIME type
    suffix = path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
    mime_type = mime_map.get(suffix, "image/png")

    client = _get_client()
    response = client.chat.completions.create(
        model=config.OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "อ่านข้อความทั้งหมดในภาพนี้ให้ครบถ้วน "
                            "ส่งกลับเป็น plain text (ภาษาไทยและอังกฤษ) "
                            "ไม่ต้องอธิบายเพิ่ม"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                        },
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=_MAX_TOKENS,
        extra_body={
            "keep_alive": _KEEP_ALIVE,
            "options": {
                "num_ctx": _NUM_CTX,
            },
        },
    )

    return response.choices[0].message.content.strip()


def ocr_pdf(
    pdf_path: str,
    *,
    progress_callback=None,
) -> list[dict]:
    """
    Run OCR on each page of a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        progress_callback: Optional callable(current_page, total_pages)
            for progress reporting.

    Returns:
        List of {"page": int, "text": str} dicts.
    """
    import tempfile
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path, dpi=_PDF_DPI)
    total = len(images)
    pages = []

    for i, image in enumerate(images, 1):
        if progress_callback:
            progress_callback(i, total)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, "PNG")
            text = ocr_image(tmp.name)
            pages.append({"page": i, "text": text})
            # Clean up temp file
            Path(tmp.name).unlink(missing_ok=True)

    return pages


def ocr_images(image_paths: list[str]) -> str:
    """
    Run OCR on multiple images and concatenate results.

    Args:
        image_paths: List of image file paths.

    Returns:
        Combined extracted text separated by newlines.
    """
    results = []
    for path in image_paths:
        text = ocr_image(path)
        if text:
            results.append(text)
    return "\n\n".join(results)
