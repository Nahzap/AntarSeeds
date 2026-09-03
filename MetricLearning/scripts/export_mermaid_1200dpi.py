from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image


def main() -> None:
    md = Path(
        "C:/Users/askna/Documents/GitHub/MetricLearning/docs/20260519_1700_V3_CORRECCION_FIGURAS_ABCD_PIPELINE_REAL.md"
    )
    text = md.read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(.*?)```", text, flags=re.S)
    if len(blocks) < 4:
        raise RuntimeError(f"Expected at least 4 mermaid blocks, found {len(blocks)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outdir = md.parent / f"{ts}_mmd_figures_1200dpi"
    outdir.mkdir(parents=True, exist_ok=True)

    for i, block in enumerate(blocks[:4], start=1):
        mmd_path = outdir / f"figure_{i}.mmd"
        png_path = outdir / f"figure_{i}.png"
        mmd_path.write_text(block.strip() + "\n", encoding="utf-8")

        npx_bin = shutil.which("npx") or shutil.which("npx.cmd")
        if not npx_bin:
            raise RuntimeError("npx was not found in PATH")

        cmd = [
            npx_bin,
            "-y",
            "@mermaid-js/mermaid-cli",
            "-i",
            str(mmd_path),
            "-o",
            str(png_path),
            "-b",
            "white",
            "-s",
            "4",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Mermaid render failed for figure {i}: {result.stderr[:600]}")

        img = Image.open(png_path)
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        img.save(png_path, format="PNG", dpi=(1200, 1200))

    print(outdir)


if __name__ == "__main__":
    main()
