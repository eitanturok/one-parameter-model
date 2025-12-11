# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "playwright",
# ]
# ///

import os
import subprocess
from playwright.sync_api import sync_playwright

input_file = "docs/docs.html"
output_file = "docs/index.html"

subprocess.run(["playwright", "install", "chromium-headless-shell"], check=True)

with sync_playwright() as p:
    with p.chromium.launch(headless=True) as browser:
        page = browser.new_page()
        page.goto(
            f"file:///{os.path.abspath(input_file)}",
            wait_until="networkidle",
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(page.content())
