import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from advection import build_advection_prior

class NowcastDataset(Dataset):
    def __init__(
        self,
        dynamic_dir="data/training/dynamic_layers",
        static_dir="data/training/static_layers",
        event_files=None,
        t_in=4,
        t_out=6,
        patch_size=128,
        samples_per_event=4,
        train=True,
        use_advection_prior=True,
        advection_damping=0.0,
    ):
        
        self.t_in = t_in
        self.t_out = t_out
        self.patch_size = patch_size
        self.samples_per_event = samples_per_event
        self.train = train
        self.use_advection_prior = use_advection_prior
        self.advection_damping = advection_damping

        self.event_files = event_files if event_files is not None else sorted(
            glob.glob(os.path.join(dynamic_dir, "event_*.npy"))
        )
        if len(self.event_files) == 0:
            raise FileNotFoundError(
                f"No event_*.npy files found in {dynamic_dir!r}. "
                "Did Training_data_extract.py finish running?"
            )
        if self.use_advection_prior and t_in < 2:
            raise ValueError("use_advection_prior needs t_in >= 2 (motion requires two frames)")

        # Static layers are small enough to keep resident in memory for the
        self.sea_mask = np.load(os.path.join(static_dir, "sea_mask.npy")).astype(np.float32)
        terrain = np.load(os.path.join(static_dir, "terrain.npy")).astype(np.float32)
        land_type = np.load(os.path.join(static_dir, "land_type.npy")).astype(np.float32)

        # Normalize terrain (elevation in meters) to roughly [0, 1]. Adjust
        # max_elev if your domain includes higher terrain than mainland UK.
        max_elev = 1350.0  # ~ Ben Nevis, tallest point in the UK
        self.terrain = np.clip(terrain, 0.0, max_elev) / max_elev

        
        lt_min, lt_max = land_type.min(), land_type.max()
        self.land_type = (land_type - lt_min) / max(lt_max - lt_min, 1e-6)

        self.static_full = np.stack(
            [self.sea_mask, self.terrain, self.land_type], axis=0
        ).astype(np.float32)  # (3, H, W)

        self.grid_h, self.grid_w = self.sea_mask.shape

    def __len__(self):
        return len(self.event_files) * self.samples_per_event

    def _random_crop_bounds(self, height, width):
        p = self.patch_size
        if height < p or width < p:
            raise ValueError(
                f"patch_size={p} is larger than the grid ({height}x{width})"
            )
        if self.train:
            top = random.randint(0, height - p)
            left = random.randint(0, width - p)
        else:
            top = (height - p) // 2
            left = (width - p) // 2
        return top, left, top + p, left + p

    def __getitem__(self, idx):
        event_idx = idx // self.samples_per_event
        event_path = self.event_files[event_idx]

        seq = np.load(event_path, mmap_mode="r")
        total_t = seq.shape[0]
        window = self.t_in + self.t_out

        if total_t < window:
            raise ValueError(
                f"{event_path} has only {total_t} frames, need at least "
                f"{window} (t_in={self.t_in} + t_out={self.t_out}). This "
                "event probably had too many failed frame fetches -- "
                "consider filtering it out of event_files upstream."
            )

        if self.train:
            start = random.randint(0, total_t - window)
        else:
            start = 0
        frames = np.array(seq[start:start + window])  # materialize the slice, (window, 2, H, W)

        top, left, bottom, right = self._random_crop_bounds(self.grid_h, self.grid_w)
        frames_crop = frames[:, :, top:bottom, left:right].astype(np.float32)
        static_crop = self.static_full[:, top:bottom, left:right]

        input_frames = frames_crop[: self.t_in]                 # (t_in, 2, p, p)
        target_frames = frames_crop[self.t_in: self.t_in + self.t_out, 0:1]  # (t_out, 1, p, p), radar channel only

        if self.use_advection_prior:
            radar_prev = input_frames[-2, 0]  # (p, p)
            radar_curr = input_frames[-1, 0]  # (p, p)
            prior = build_advection_prior(
                radar_prev, radar_curr, self.t_out, damping=self.advection_damping
            )  # (t_out, p, p)
            prior = prior[:, None, :, :]  # (t_out, 1, p, p)
        else:
            prior = np.zeros_like(target_frames)

        return (
            torch.from_numpy(input_frames).float(),
            torch.from_numpy(static_crop).float(),
            torch.from_numpy(target_frames).float(),
            torch.from_numpy(prior).float(),
        )


def make_train_val_datasets(
    dynamic_dir="data/training/dynamic_layers",
    static_dir="data/training/static_layers",
    t_in=4,
    t_out=6,
    patch_size=128,
    samples_per_event=4,
    val_fraction=0.15,
    seed=42,
    use_advection_prior=True,
    advection_damping=0.0,
):
    """Splits events (not frames) between train/val so validation windows
    never come from the same storm event as training windows."""
    all_files = sorted(glob.glob(os.path.join(dynamic_dir, "event_*.npy")))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No event_*.npy files found in {dynamic_dir!r}")

    rng = random.Random(seed)
    shuffled = all_files[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_files = sorted(shuffled[:n_val])
    train_files = sorted(shuffled[n_val:])

    train_ds = NowcastDataset(
        dynamic_dir=dynamic_dir, static_dir=static_dir, event_files=train_files,
        t_in=t_in, t_out=t_out, patch_size=patch_size,
        samples_per_event=samples_per_event, train=True,
        use_advection_prior=use_advection_prior, advection_damping=advection_damping,
    )
    val_ds = NowcastDataset(
        dynamic_dir=dynamic_dir, static_dir=static_dir, event_files=val_files,
        t_in=t_in, t_out=t_out, patch_size=patch_size,
        samples_per_event=max(1, samples_per_event // 2), train=False,
        use_advection_prior=use_advection_prior, advection_damping=advection_damping,
    )
    return train_ds, val_ds


if __name__ == "__main__":
    # quick smoke test against real data on disk, if present
    try:
        ds = NowcastDataset(patch_size=128, samples_per_event=1)
        x, static, y, prior = ds[0]
        print("input:", x.shape, "static:", static.shape, "target:", y.shape, "prior:", prior.shape)
    except FileNotFoundError as e:
        print("Smoke test skipped (no data found yet):", e)