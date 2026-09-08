"""Create an artifact index and diagnostic contact-sheet figures; keep originals."""

import argparse
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
files = [a.directory / "cold-base.png"] + sorted(a.directory.glob("cold-region*.png"))
if (a.directory / "mesh-front.png").exists():
    files.append(a.directory / "mesh-front.png")
cols = 3
rows = (len(files) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
for ax, path in zip(axes.flat, files):
    ax.imshow(Image.open(path))
    ax.set_title(path.stem, fontsize=10)
for ax in axes.flat:
    ax.axis("off")
fig.tight_layout()
fig.savefig(a.directory / "contact-sheet.jpg", dpi=100)
plt.close(fig)
body = [
    '<!doctype html><meta charset="utf-8"><title>Single A100 quality artifacts</title><style>body{font-family:sans-serif}figure{display:inline-block;width:30%;vertical-align:top}img{width:100%}</style>',
    f"<h1>{html.escape(a.directory.name)}</h1>",
    "<p>Original output PNGs and unchanged GLB export. Contact sheet and mesh rendering are diagnostic previews.</p>",
]
for path in files:
    name = html.escape(path.name)
    body.append(
        f'<figure><a href="{name}"><img src="{name}"></a><figcaption>{name}</figcaption></figure>'
    )
body.append(
    '<p><a href="cold-final.glb">Final GLB mesh</a> · <a href="settings.json">Released workflow settings</a></p>'
)
(a.directory / "index.html").write_text("\n".join(body))
