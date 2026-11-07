"""Retake the pictures in the README, from the real app in a real browser.

DON'T PANIC. 🐬  You only need this if you changed the app and a picture has gone stale.

    uv run --no-project --python 3.12 --with playwright --with pillow python scripts/take_readme_screenshots.py

Playwright and Pillow come from that command, in a throwaway environment, so nothing is
added to this project. It starts the app and JupyterLab, clicks through them as a reader
would, saves cropped images to docs/images/, and shuts everything down.

It needs the data extracted, and the `slm` extra and model for the chat pictures.
It drives Microsoft Edge, which is on every Windows machine, so there is no browser to
download. On another system, change CHANNEL to "chrome".

🔒 These pictures go on the internet, and tax rolls describe real people's homes:

  - Tab 2 shows real rows of the roll after merging, so it is never photographed.
  - It asks the assistant about the whole roll and never about a parcel.
  - LOOK AT EVERY IMAGE before committing it. This script cannot tell you it caught
    nobody's name, and only a person can.

ASCII only in console output - see scripts/preflight.py.
"""

import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
NOTEBOOK = "notebooks/A_HitchHikers_Guide_To_Building_A_Machine_Learning_Model.ipynb"
CHANNEL = "msedge"
APP_PORT, JUPYTER_PORT, TOKEN = "8597", "8896", "guide"

# Each picture, and the part of the window worth showing: left, top, right, bottom.
CROPS = {
    "jupyter_lab": (0, 0, 1440, 760),
    "load_data": (0, 0, 1440, 1060),
    "build_model": (300, 260, 1440, 900),
    "results": (300, 380, 1440, 1385),
    "head_to_head": (300, 600, 1440, 1040),
    "chat_choice": (300, 260, 1440, 1000),
    "chat_answer": (300, 232, 1440, 985),
    "settings": (300, 260, 1440, 1095),
}


def wait_for(url):
    for _ in range(120):
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("Never came up: " + url)


def settle(page, longest=900_000):
    """Wait for Streamlit to finish its run."""
    page.wait_for_timeout(1200)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=longest)
    page.wait_for_timeout(1200)


def tab(page, label):
    page.get_by_role("tab", name=re.compile(re.escape(label))).click()
    page.wait_for_timeout(700)


def scroll_main(page, where="0"):
    page.evaluate("() => { const m = document.querySelector('[data-testid=\"stMain\"]');"
                  " if (m) m.scrollTo(0, %s); }" % where)
    page.wait_for_timeout(500)


def save(page, name, folder):
    if page.locator('[data-testid="stException"]').count():
        raise SystemExit("An exception is on the page at: " + name)
    page.screenshot(path=str(folder / (name + ".png")))
    print("  took %s" % name)


def photograph(folder):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=CHANNEL, headless=True)

        page = browser.new_context(viewport={"width": 1440, "height": 900}).new_page()
        page.goto("http://localhost:%s/lab/tree/%s?token=%s&reset" % (JUPYTER_PORT, NOTEBOOK, TOKEN))
        page.wait_for_selector(".jp-Notebook", timeout=120_000)
        page.wait_for_timeout(6000)
        news = page.get_by_role("button", name="No", exact=True)
        if news.count():
            news.first.click()
            page.wait_for_timeout(800)
        page.screenshot(path=str(folder / "jupyter_lab.png"))
        print("  took jupyter_lab")
        page.close()

        page = browser.new_context(viewport={"width": 1440, "height": 1100}).new_page()
        page.goto("http://localhost:%s" % APP_PORT)
        page.get_by_role("tab", name=re.compile("Load Data")).wait_for(timeout=120_000)
        settle(page)
        save(page, "load_data", folder)

        page.get_by_role("button", name="Load the Lee County example").click()
        page.get_by_text("Lee County loaded.").wait_for(timeout=300_000)
        settle(page)

        # 🔒 Merging puts real rows of the roll on this tab. Never photographed.
        tab(page, "2. Prepare")
        page.get_by_role("button", name="Merge and Preprocess My Files").click()
        settle(page)

        tab(page, "3. Build Model")
        page.get_by_text("Single Family Residential Improved Model").click()
        settle(page)
        save(page, "build_model", folder)
        page.get_by_role("button", name="Build Model").click()
        page.get_by_text(re.compile("successfully built")).wait_for(timeout=900_000)
        settle(page)

        page.set_viewport_size({"width": 1440, "height": 1500})
        tab(page, "4. Results")
        scroll_main(page)
        save(page, "results", folder)

        page.get_by_text("Display Model Performance Graphs").click()
        settle(page)
        radio = page.locator('[data-testid="stRadio"]').filter(has_text="Model Metric").first
        radio.get_by_text("COD", exact=True).click()
        settle(page)
        page.get_by_text(re.compile("Model vs county roll, on identical sales")).first.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        save(page, "head_to_head", folder)

        page.set_viewport_size({"width": 1440, "height": 1100})
        tab(page, "6. Settings")
        scroll_main(page)
        save(page, "settings", folder)

        tab(page, "5. Ask Questions")
        scroll_main(page)
        page.get_by_text(re.compile("Local  ?-")).first.click()
        settle(page)
        save(page, "chat_choice", folder)

        page.get_by_role("button", name="Start Chat!").click()
        box = page.locator('[data-testid="stChatInputTextArea"]')
        box.wait_for(timeout=300_000)
        settle(page)
        box.click()
        box.fill("How does the model compare with the county roll on COD and PRD?")
        box.press("Enter")
        # A chat answer arrives late, so wait for words from it, not for the page to be idle.
        page.get_by_text(re.compile("11\\.0168")).first.wait_for(timeout=300_000)
        settle(page)
        scroll_main(page, "m.scrollHeight")
        save(page, "chat_answer", folder)

        page.get_by_role("button", name="Shut Down").click()
        page.wait_for_timeout(1500)
        browser.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jupyter = subprocess.Popen(
        ["uv", "run", "--no-sync", "jupyter", "lab", "--no-browser", "--port", JUPYTER_PORT,
         "--ServerApp.token=" + TOKEN],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    app = subprocess.Popen(
        ["uv", "run", "--no-sync", "streamlit", "run", "app/NAL_Navigator.py",
         "--server.headless", "true", "--server.port", APP_PORT],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        wait_for("http://localhost:%s/_stcore/health" % APP_PORT)
        wait_for("http://localhost:%s/api/status?token=%s" % (JUPYTER_PORT, TOKEN))
        with tempfile.TemporaryDirectory() as raw:
            photograph(Path(raw))
            for name, box in CROPS.items():
                image = Image.open(Path(raw) / (name + ".png")).convert("RGB").crop(box)
                image.save(OUT / (name + ".png"), optimize=True)
                print("  saved docs/images/%s.png  %d KB"
                      % (name, (OUT / (name + ".png")).stat().st_size // 1024))
    finally:
        try:
            urllib.request.urlopen(urllib.request.Request(
                "http://localhost:%s/api/shutdown?token=%s" % (JUPYTER_PORT, TOKEN), method="POST"), timeout=5)
        except Exception:
            pass
        for process in (app, jupyter):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()

    print()
    print("Done. Now LOOK AT EVERY IMAGE before committing it: no owners, no parcels.")


if __name__ == "__main__":
    main()
