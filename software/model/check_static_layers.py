"""
Standalone check of the static layers, independent of the forecast pipeline.
Run this from software/model/ (or wherever puts data/ within reach):

    python check_static_layers.py

It just loads and plots sea_mask.npy on its own -- no radar overlay, no
model. If this alone looks wrong (rotated, stretched, wrong shape), the
bug is upstream in Static_tensor.py's mask/terrain generation, not in
app.py's rendering. If it looks correct here, the bug is in how app.py
aligns it against the radar frames.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
STATIC_DIR = REPO_ROOT / "data" / "training" / "static_layers"

sea_mask = np.load(STATIC_DIR / "sea_mask.npy")
print("sea_mask.shape:", sea_mask.shape)
print("sea_mask dtype:", sea_mask.dtype)
print("unique values:", np.unique(sea_mask)[:10])
print("expected grid shape (rows, cols): (2175, 1725)")

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].imshow(sea_mask, cmap="Greens", origin="upper")
axes[0].set_title(f"origin='upper', shape={sea_mask.shape}")
axes[1].imshow(sea_mask, cmap="Greens", origin="lower")
axes[1].set_title(f"origin='lower', shape={sea_mask.shape}")
plt.tight_layout()
out_path = THIS_DIR / "sea_mask_check.png"
plt.savefig(out_path, dpi=120)
print(f"Saved: {out_path}")