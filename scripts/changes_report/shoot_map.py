#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screenshot the full-page HTML changes map to a single tall PNG.

Pages taller than Chromium's single-capture limit are captured in vertical
slices: the viewport is resized to each slice height first, so
window.scrollTo() is never clamped (max scroll = docHeight - viewportHeight).
Slices are stitched with Pillow into one PNG."""
import argparse
import asyncio
import json

from PIL import Image
from playwright.async_api import async_playwright

WIDTH = 1680
SCALE = 1.25
SLICE = 4000


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", default="dist/changes-report/map.html")
    ap.add_argument("--out", default="dist/changes-report/map.png")
    ap.add_argument("--data", default="dist/changes-report/commits.json",
                    help="used to verify that every commit/file chip rendered")
    return ap


async def capture(html_path, out_png, expect_commits, expect_fchips):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--force-color-profile=srgb", "--font-render-hinting=none",
                  "--disable-lcd-text"])
        page = await browser.new_page(viewport={"width": WIDTH, "height": SLICE},
                                      device_scale_factor=SCALE)
        await page.goto(f"file://{html_path}")
        await page.evaluate("document.fonts.ready")
        await page.wait_for_timeout(900)
        height = await page.evaluate("document.documentElement.scrollHeight")
        n_commits = await page.evaluate("document.querySelectorAll('article.commit').length")
        n_fchips = await page.evaluate("document.querySelectorAll('.fchip').length")
        print(f"content height: {height}px | commits rendered: {n_commits} | file chips: {n_fchips}")
        assert n_commits == expect_commits, f"missing commit cards: {n_commits}/{expect_commits}"
        assert n_fchips == expect_fchips, f"missing file chips: {n_fchips}/{expect_fchips}"

        parts, y, i = [], 0, 0
        while y < height:
            h = min(SLICE, height - y)
            # resize viewport to slice height => max scroll == y (no clamping)
            await page.set_viewport_size({"width": WIDTH, "height": h})
            await page.wait_for_timeout(140)
            await page.evaluate(f"window.scrollTo(0, {y})")
            await page.wait_for_timeout(160)
            got = await page.evaluate("Math.round(window.scrollY)")
            assert got == y, f"scroll clamp! wanted {y}, got {got}"
            path = f"/tmp/map_part_{i:02d}.png"
            await page.screenshot(path=path, animations="disabled")
            img = Image.open(path)
            target_h = int(round(h * SCALE))
            if img.height > target_h:      # crop trailing blank on last slice
                img = img.crop((0, 0, img.width, target_h))
                img.save(path)
            parts.append(path)
            print(f"  slice {i}: y={y} h={h} px={img.width}x{img.height}")
            y += h
            i += 1
        await browser.close()

    imgs = [Image.open(p) for p in parts]
    canvas = Image.new("RGB", (imgs[0].width, sum(im.height for im in imgs)), "white")
    yy = 0
    for im in imgs:
        canvas.paste(im, (0, yy))
        yy += im.height
    canvas.save(out_png, optimize=True)
    print(f"stitched map saved: {out_png} ({canvas.width}x{canvas.height}px)")


def main():
    args = build_argparser().parse_args()
    with open(args.data, encoding="utf-8") as f:
        commits = json.load(f)
    expect_fchips = sum(c["n_files"] for c in commits)
    asyncio.run(capture(args.html, args.out, len(commits), expect_fchips))


if __name__ == "__main__":
    main()
