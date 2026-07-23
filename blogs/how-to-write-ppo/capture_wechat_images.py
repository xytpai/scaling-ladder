from __future__ import annotations

import math
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


HERE = Path(__file__).resolve().parent
SOURCE_HTML = HERE / "ppo-wechat.html"
OUTPUT_DIR = HERE / "ppo-wechat-images"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

CSS_WIDTH = 390
PIXEL_WIDTH = 1080
DEVICE_SCALE = 3
TARGET_SLICE_HEIGHT = 900
MIN_SLICE_HEIGHT = 680
MAX_SLICE_HEIGHT = 1000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

SCREENSHOT_CSS = """
html, body {
  width: 390px !important;
  min-width: 390px !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow-x: hidden !important;
  background: #ffffff !important;
}
.preview-toolbar, .toast {
  display: none !important;
}
.preview-canvas {
  width: 390px !important;
  max-width: 390px !important;
  margin: 0 !important;
  overflow: hidden !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
.wx-article {
  width: 390px !important;
  max-width: 390px !important;
  margin: 0 !important;
  padding: 34px 20px 46px !important;
}
.wx-article h1 {
  font-size: 27px !important;
}
.wx-article .reading-route {
  white-space: normal !important;
}
.wx-article .math-scroll {
  overflow: hidden !important;
}
.wx-article .codehilite,
.wx-article .codehilite pre {
  overflow: visible !important;
}
.wx-article .codehilite pre,
.wx-article .codehilite code {
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}
"""

FIT_WIDE_CONTENT = """
() => {
  document.querySelectorAll(".math-scroll").forEach((scroller) => {
    const formula = scroller.querySelector(".katex-display");
    if (!formula) return;

    formula.style.minWidth = "max-content";
    formula.style.transform = "none";
    const naturalWidth = Math.max(
      formula.scrollWidth,
      formula.getBoundingClientRect().width
    );
    const naturalHeight = formula.getBoundingClientRect().height;
    const availableWidth = scroller.clientWidth - 2;

    if (naturalWidth > availableWidth) {
      const scale = availableWidth / naturalWidth;
      formula.style.width = naturalWidth + "px";
      formula.style.transformOrigin = "left top";
      formula.style.transform = `scale(${scale})`;
      scroller.style.height = Math.ceil(naturalHeight * scale + 10) + "px";
    }
  });
}
"""

COLLECT_BREAKS = """
() => {
  const article = document.querySelector("#wechat-article");
  const articleRect = article.getBoundingClientRect();
  const offsets = [];

  article.querySelectorAll(":scope > *").forEach((element) => {
    const rect = element.getBoundingClientRect();
    offsets.push(Math.round(rect.bottom - articleRect.top));
  });

  article.querySelectorAll(".code-card pre").forEach((pre) => {
    const rect = pre.getBoundingClientRect();
    const style = getComputedStyle(pre);
    const lineHeight = parseFloat(style.lineHeight) || 20;
    const top = rect.top - articleRect.top;
    for (let y = top + lineHeight; y < rect.bottom - articleRect.top; y += lineHeight) {
      offsets.push(Math.round(y));
    }
  });

  return {
    height: Math.ceil(articleRect.height),
    offsets: [...new Set(offsets)].sort((a, b) => a - b)
  };
}
"""


def choose_slices(total_height: int, candidates: list[int]) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    start = 0

    while total_height - start > MAX_SLICE_HEIGHT:
        eligible = [
            point
            for point in candidates
            if start + MIN_SLICE_HEIGHT <= point <= start + MAX_SLICE_HEIGHT
        ]
        if eligible:
            target = start + TARGET_SLICE_HEIGHT
            end = min(eligible, key=lambda point: abs(point - target))
        else:
            end = start + TARGET_SLICE_HEIGHT

        if total_height - end < MIN_SLICE_HEIGHT // 2:
            end = total_height
        slices.append((start, end))
        start = end

    if start < total_height:
        slices.append((start, total_height))
    return slices


def resize_slice(raw_path: Path, output_path: Path) -> None:
    with Image.open(raw_path) as image:
        target_height = round(image.height * PIXEL_WIDTH / image.width)
        resized = image.resize((PIXEL_WIDTH, target_height), Image.Resampling.LANCZOS)
        resized.save(output_path, format="PNG", optimize=True)


def ensure_upload_size(path: Path) -> Path:
    if path.stat().st_size <= MAX_UPLOAD_BYTES:
        return path

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        jpg_path = path.with_suffix(".jpg")
        for quality in (94, 90, 86, 82):
            rgb.save(
                jpg_path,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            if jpg_path.stat().st_size <= MAX_UPLOAD_BYTES:
                path.unlink()
                return jpg_path
    raise RuntimeError(f"Could not compress {path.name} below 5 MB")


def build_long_image(parts: list[Path]) -> Path:
    opened = [Image.open(path).convert("RGB") for path in parts]
    try:
        total_height = sum(image.height for image in opened)
        canvas = Image.new("RGB", (PIXEL_WIDTH, total_height), "#ffffff")
        y = 0
        for image in opened:
            canvas.paste(image, (0, y))
            y += image.height

        long_path = OUTPUT_DIR / "ppo-wechat-long.jpg"
        for quality in (88, 84, 80, 76, 72, 68):
            canvas.save(
                long_path,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            if long_path.stat().st_size <= MAX_UPLOAD_BYTES:
                return long_path
        raise RuntimeError("Could not compress the long image below 5 MB")
    finally:
        for image in opened:
            image.close()


def main() -> None:
    if not SOURCE_HTML.exists():
        raise FileNotFoundError(f"Missing {SOURCE_HTML}")
    if not CHROME.exists():
        raise FileNotFoundError(f"Missing Chrome at {CHROME}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    for old_file in OUTPUT_DIR.glob("ppo-wechat-*.*"):
        old_file.unlink()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(CHROME),
            headless=True,
            args=["--allow-file-access-from-files", "--disable-gpu"],
        )
        context = browser.new_context(
            viewport={"width": CSS_WIDTH, "height": 844},
            device_scale_factor=DEVICE_SCALE,
            color_scheme="light",
        )
        page = context.new_page()
        page.goto(SOURCE_HTML.resolve().as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.add_style_tag(content=SCREENSHOT_CSS)
        page.evaluate(FIT_WIDE_CONTENT)
        page.wait_for_timeout(300)

        article_box = page.locator("#wechat-article").bounding_box()
        if article_box is None:
            raise RuntimeError("Could not locate the article")
        metrics = page.evaluate(COLLECT_BREAKS)
        slices = choose_slices(int(metrics["height"]), list(metrics["offsets"]))

        output_parts: list[Path] = []
        for index, (start, end) in enumerate(slices, start=1):
            raw_path = OUTPUT_DIR / f".raw-{index:02d}.png"
            png_path = OUTPUT_DIR / f"ppo-wechat-{index:02d}.png"
            page.set_viewport_size(
                {"width": CSS_WIDTH, "height": max(1, end - start)}
            )
            page.evaluate("(offset) => window.scrollTo(0, offset)", start)
            page.wait_for_timeout(50)
            page.screenshot(
                path=str(raw_path),
                animations="disabled",
                caret="hide",
            )
            resize_slice(raw_path, png_path)
            raw_path.unlink()
            output_parts.append(ensure_upload_size(png_path))

        context.close()
        browser.close()

    long_path = build_long_image(output_parts)
    print(f"Long image: {long_path.name} ({long_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Upload-ready slices: {len(output_parts)}")
    for path in output_parts:
        with Image.open(path) as image:
            print(
                f"  {path.name}: {image.width}x{image.height}, "
                f"{path.stat().st_size / 1024 / 1024:.2f} MB"
            )


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None
    main()
