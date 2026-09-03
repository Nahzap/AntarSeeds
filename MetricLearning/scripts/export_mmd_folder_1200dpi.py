from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image


def render_mmd_to_png(mmd_path: Path, png_path: Path) -> None:
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
        raise RuntimeError(f"Mermaid render failed for {mmd_path.name}: {result.stderr[:600]}")

    img = Image.open(png_path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    img.save(png_path, format="PNG", dpi=(1200, 1200))


def main() -> None:
    folder = Path(
        "C:/Users/askna/Documents/GitHub/MetricLearning/docs/20260519_1712_mmd_figures_1200dpi"
    )
    mapping = {
        "1.figure_1.mmd": "1.General_system_diagram.png",
        "2.figure_2.mmd": "2.MetricLearning_Schema.png",
        "3.figure_3.mmd": "3.Automatic_Semiautomatic_labelling.png",
        "4.figure_4.mmd": "4.InferenceByEmbeddings.png",
    }

    for mmd_name, png_name in mapping.items():
        mmd_path = folder / mmd_name
        png_path = folder / png_name
        if not mmd_path.exists():
            raise FileNotFoundError(f"Missing source: {mmd_path}")
        render_mmd_to_png(mmd_path, png_path)
        print(f"[OK] {png_path}")


if __name__ == "__main__":
    main()
