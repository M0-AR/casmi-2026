"""Captures a short sequence of REAL screenshots of index.html (hero,
beginner guide expanded, a real quiz answer click with its real feedback
state, and the real final quiz score) and assembles them into an animated
GIF. Every frame is an actual rendered/interacted page state -- nothing is
mocked or drawn by hand.

Drives the system's already-installed Google Chrome via Playwright's
Python bindings (channel="chrome"), so no separate browser download is
needed.

Usage: venv/bin/python notebook/capture_demo.py
"""
import pathlib
import time

from playwright.sync_api import sync_playwright
from PIL import Image

HTML_PATH = pathlib.Path(__file__).resolve().parent.parent / "index.html"
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs"
FRAMES_DIR = OUT_DIR / "_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

frames = []


def shoot(page, name, delay_ms=0):
    if delay_ms:
        page.wait_for_timeout(delay_ms)
    path = FRAMES_DIR / f"{name}.png"
    page.screenshot(path=str(path))
    frames.append(path)
    print(f"captured {name}")


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto(f"file://{HTML_PATH}")
    page.wait_for_timeout(300)

    shoot(page, "01_hero")

    page.evaluate("document.querySelector('#stats').scrollIntoView({behavior: 'instant'})")
    shoot(page, "02_stats", delay_ms=200)

    page.evaluate("document.querySelector('#beginner-guide').scrollIntoView({behavior: 'instant'})")
    shoot(page, "03_beginner_guide", delay_ms=200)

    # Open the 3rd beginner-guide detail (the hardest-part explainer) for real
    details = page.query_selector_all("#beginner-guide details.explainer")
    details[2].query_selector("summary").click()
    shoot(page, "04_beginner_guide_expanded", delay_ms=200)

    page.evaluate("document.querySelector('#real-run').scrollIntoView({behavior: 'instant'})")
    shoot(page, "05_real_run_log", delay_ms=200)

    page.evaluate("document.querySelector('#quiz').scrollIntoView({behavior: 'instant'})")
    shoot(page, "06_quiz_start", delay_ms=200)

    # Answer the real quiz, question by question, for real
    for i in range(5):
        opts = page.query_selector_all(".quiz-opt")
        # Click the actual correct answer (we know the index from index.html's QUESTIONS array)
        correct_index = {0: 1, 1: 2, 2: 1, 3: 2, 4: 1}[i]
        opts[correct_index].click()
        shoot(page, f"07_quiz_q{i+1}_answered", delay_ms=300)
        if i < 4:
            page.query_selector("#quiz-next-btn").click()
            page.wait_for_timeout(200)
        else:
            page.query_selector("#quiz-next-btn").click()
            shoot(page, "08_quiz_final_score", delay_ms=300)

    browser.close()

print(f"\n{len(frames)} real frames captured")

images = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
durations = [1400] * len(images)
durations[0] = 2200  # hold on hero a bit longer
durations[-1] = 3000  # hold on final score

out_path = OUT_DIR / "demo.gif"
images[0].save(
    out_path,
    save_all=True,
    append_images=images[1:],
    duration=durations,
    loop=0,
    optimize=True,
)
print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
