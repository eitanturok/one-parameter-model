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

GOOGLE_TAG = """<!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-7QG3BKMM52"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-7QG3BKMM52');
    </script>\n"""

# used for tracking website traffic with Google Analytics
def inject_google_tag(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    if "G-7QG3BKMM52" not in html:
        html = html.replace("<head>", f"<head>\n{GOOGLE_TAG}", 1)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

def main():

    # run marimo to export the OneParameterModel.py file as an HTML file
    subprocess.run(
        ["marimo", "export", "html", "OneParameterModel.py", "-o", input_file, "--force", "--no-include-code"],
        check=True,
    )

    # use playwright to render the HTML file and save the rendered content to a new file
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

    # inject the Google Analytics tag into the output HTML file
    inject_google_tag(output_file)

if __name__ == "__main__":
    main()
