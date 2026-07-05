# -*- coding: utf-8 -*-
"""
Build the deliverable zip: every GIF next to its exact description.

Layout inside the zip (folder: gpu_ai_linkedin_gif_series/):
  00_README.txt
  00_ALL_POSTS.md                         <- all 60 in one scrollable file
  day_01_gpu_memory_coalescing.gif
  day_01_gpu_memory_coalescing.txt        <- concept + ready-to-paste post
  ... (x60)

Run: python3 package_build.py
Outputs: out_package/  and  gpu_ai_linkedin_gif_series.zip
"""
import os
import shutil
import zipfile
from package_posts import POSTS

HERE = os.path.dirname(__file__)
GIF_DIR = os.path.join(HERE, "out")
PKG_NAME = "gpu_ai_linkedin_gif_series"
STAGE = os.path.join(HERE, "out_package", PKG_NAME)
ZIP_PATH = os.path.join(HERE, "out_package", PKG_NAME + ".zip")

README = """GPU / CUDA / AI - 60-Day LinkedIn GIF Series
============================================

60 original animated GIFs (each a looping animation under 6 seconds) with a
ready-to-paste, pain/truth-led LinkedIn caption for every one.

Topics: GPU * CUDA * Artificial Intelligence * LLMs * GEMM * NVIDIA NeMo *
NVIDIA Nemotron * InfiniBand.

WHAT'S IN HERE
--------------
* day_XX_name.gif   -> the animation to attach to the post
* day_XX_name.txt   -> that GIF's concept + the exact LinkedIn caption
* 00_ALL_POSTS.md   -> every day's caption in one scrollable file

HOW TO POST
-----------
1. Pick the day. Open its .txt (or scroll 00_ALL_POSTS.md).
2. Attach the matching .gif to a new LinkedIn post.
3. Paste the caption text. Post one per day.
4. Reply to comments in the first hour - it's the biggest reach lever.

WRITING STYLE
-------------
Each caption leads with the real pain or truth of the scenario - the line that
makes an AI/GPU engineer stop scrolling - then delivers a genuine technical
insight and ends on a question to spark discussion. Keyword-rich for reach,
never job-seeking.

Total: 60 GIFs + 60 captions.
"""


def main():
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    all_lines = ["# 60-Day LinkedIn GIF Series - GPU / CUDA / AI / LLMs\n",
                 "_Each GIF pairs with a pain/truth-led, keyword-rich caption. "
                 "One post per day._\n"]
    missing = []

    for day, fname, concept, post in POSTS:
        src = os.path.join(GIF_DIR, fname)
        if not os.path.exists(src):
            missing.append(fname)
            continue
        shutil.copy2(src, os.path.join(STAGE, fname))

        base = fname[:-4]  # strip .gif
        txt = (
            f"DAY {day}\n"
            f"{'=' * 60}\n\n"
            f"GIF FILE: {fname}\n\n"
            f"GIF CONCEPT:\n{concept}\n\n"
            f"LINKEDIN POST (paste as-is):\n"
            f"{'-' * 60}\n{post}\n"
        )
        with open(os.path.join(STAGE, base + ".txt"), "w", encoding="utf-8") as f:
            f.write(txt)

        all_lines.append(f"\n---\n\n## Day {day}\n")
        all_lines.append(f"**GIF file:** `{fname}`\n")
        all_lines.append(f"**Concept:** {concept}\n")
        all_lines.append(f"\n**LinkedIn post:**\n\n{post}\n")

    with open(os.path.join(STAGE, "00_README.txt"), "w", encoding="utf-8") as f:
        f.write(README)
    with open(os.path.join(STAGE, "00_ALL_POSTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))

    # zip it
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(STAGE):
            for fn in sorted(files):
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, os.path.dirname(STAGE))
                z.write(full, arc)

    gifs = len([1 for _, fn, _, _ in POSTS if os.path.exists(os.path.join(GIF_DIR, fn))])
    size = os.path.getsize(ZIP_PATH) / 1e6
    print(f"packaged {gifs} gifs + captions")
    if missing:
        print("MISSING gifs:", missing)
    print(f"zip: {ZIP_PATH}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
