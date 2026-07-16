import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from advection import build_advection_prior

class NowcastDataset(Dataset):

    """
    Physics-based geospatial dataset for precipitation nowcasting.

    Loads satellite/radar temporal sequences with static geospatial features.
    Rejects impossible storm movements.
    """
    def __init__(
        self,
        dynamic_dir="data/training/dynamic_layers",
        static_dir="data/training/static_layers",
        event_files=None,
        t_in=4, # Number of historic frames fed into model
        t_out=6, # Number of frames predicted by model
        cadence_minutes=5,
        patch_size=128,
        samples_per_event=4,
        train=True,
        use_advection_prior=True, # Enable/disable the physical optical flow advection projection.
        advection_damping=0.0, # Temporal decay rate applied to the advection velocity vectors.
        reject_frame_gaps=True,
        max_speed_kmh=150.0,
        gap_relative_threshold=2.5, # Max acceleration multiple allowed before flagging an anomalous frame gap.
        max_gap_retries=5 # Number of crop/time sliding window attempts before giving up and using the current slice.
    ):
        
        self.t_in = t_in
        self.t_out = t_out
        self.patch_size = patch_size
        self.samples_per_event = samples_per_event
        self.train = train
        self.use_advection_prior = use_advection_prior
        self.advection_damping = advection_damping
        self.reject_frame_gaps = reject_frame_gaps
        self.max_speed_kmh = max_speed_kmh
        self.gap_relative_threshold = gap_relative_threshold
        self.max_gap_retries = max_gap_retries
        self.grid_res_km = 1.0  # matches the true 1km "uk-1km" NIMROD composite resolution
        self.cadence_minutes = cadence_minutes

        # Converts maximum speed to max displacement in pixels per frame.
        self.max_step_px = (max_speed_kmh * (cadence_minutes / 60.0)) / self.grid_res_km

        self.event_files = event_files if event_files is not None else sorted(
            glob.glob(os.path.join(dynamic_dir, "event_*.npy"))
        )
        if len(self.event_files) == 0:
            raise FileNotFoundError(
                f"No event_*.npy files found in {dynamic_dir!r}. "
                "Did Training_data_extract.py finish running?"
            )
        
        # Physical flow requires at least two historic steps to measure velocity vectors
        if self.use_advection_prior and t_in < 2:
            raise ValueError("use_advection_prior needs t_in >= 2 (motion requires two frames)")

        # Static layers are small enough to keep resident in memory for the RAM
        self.sea_mask = np.load(os.path.join(static_dir, "sea_mask.npy")).astype(np.float32)
        terrain = np.load(os.path.join(static_dir, "terrain.npy")).astype(np.float32)
        land_type = np.load(os.path.join(static_dir, "land_type.npy")).astype(np.float32)

        # Normalize terrain (elevation in meters) to roughly [0, 1]. Adjust
        # max_elev if your domain includes higher terrain than mainland UK.
        max_elev = 1350.0  # ~ Ben Nevis, tallest point in the UK
        self.terrain = np.clip(terrain, 0.0, max_elev) / max_elev

        # Normalize Land Type categorical labels to fall within [0, 1].
        lt_min, lt_max = land_type.min(), land_type.max()
        self.land_type = (land_type - lt_min) / max(lt_max - lt_min, 1e-6)

        # Stack into a static geographic reference block of shape (3, H, W)
        self.static_full = np.stack(
            [self.sea_mask, self.terrain, self.land_type], axis=0
        ).astype(np.float32)  # (3, H, W)

        self.grid_h, self.grid_w = self.sea_mask.shape


    # multiplies files by samples per event to draw random times for each heavy storm file
    def __len__(self):
        return len(self.event_files) * self.samples_per_event
    

    # Calculates centre of mass of rain within a frame to track trajectory of storm
    def _frame_centroid(self, frame, threshold=0.05):
        mask = frame >= threshold
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        weights = frame[mask]
        return (float(np.average(ys, weights=weights)), float(np.average(xs, weights=weights)))
    

    # Checks if the storm violates the laws of physics to see whether the frame drops
    def _window_has_gap(self, radar_frames):      
        if not self.reject_frame_gaps or radar_frames.shape[0] < 3:
            return False  
    
        # Calculates centroid coordinates for each frame
        centroids = [self._frame_centroid(radar_frames[t]) for t in range(radar_frames.shape[0])]
        displacements = []

        # Calculates distance steps between consecutive frames
        for t in range(len(centroids) - 1):
            c0, c1 = centroids[t], centroids[t + 1]
            if c0 is None or c1 is None:
                continue  
            displacements.append(float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])))
 
        if not displacements:
            return False
        
        # Checks if the storm has travelled farther than the max speed limit or if acceleration is too high
        displacements = np.array(displacements)
        if np.any(displacements > self.max_step_px):
            return True
        median_disp = np.median(displacements)
        if median_disp > 1e-6:  
            if np.any(displacements > self.gap_relative_threshold * median_disp):
                return True
        return False
    

    # Randomises crop coordinates to validate training data
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
            attempts = self.max_gap_retries + 1
        else:
            attempts = 1  # keep validation deterministic -- always the same window/crop

        
        frames_crop = None
        for attempt in range(attempts):
            if self.train:
                start = random.randint(0, total_t - window)
            else:
                start = 0
            frames = np.array(seq[start:start + window])  # materialize the slice, (window, 2, H, W)
 
            top, left, bottom, right = self._random_crop_bounds(self.grid_h, self.grid_w)
            candidate = frames[:, :, top:bottom, left:right].astype(np.float32)
            static_crop = self.static_full[:, top:bottom, left:right]
 
            if not self._window_has_gap(candidate[:, 0]):
                frames_crop = candidate
                break
            frames_crop = candidate  # keep the last-tried one in case every attempt is flagged
 
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
    reject_frame_gaps=True,
    max_speed_kmh=150.0
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
        reject_frame_gaps=reject_frame_gaps, max_speed_kmh=max_speed_kmh,
    )
    val_ds = NowcastDataset(
        dynamic_dir=dynamic_dir, static_dir=static_dir, event_files=val_files,
        t_in=t_in, t_out=t_out, patch_size=patch_size,
        samples_per_event=max(1, samples_per_event // 2), train=False,
        use_advection_prior=use_advection_prior, advection_damping=advection_damping,
        reject_frame_gaps=reject_frame_gaps, max_speed_kmh=max_speed_kmh,
    )
    return train_ds, val_ds


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamic-dir", default="data/training/dynamic_layers")
    ap.add_argument("--static-dir", default="data/training/static_layers")
    ap.add_argument("--patch-size", type=int, default=128)
    cli_args = ap.parse_args()
 
    # quick test against data on disk (real or synthetic)
    try:
        ds = NowcastDataset(
            dynamic_dir=cli_args.dynamic_dir, static_dir=cli_args.static_dir,
            patch_size=cli_args.patch_size, samples_per_event=1,
        )
        x, static, y, prior = ds[0]
        print("input:", x.shape, "static:", static.shape, "target:", y.shape, "prior:", prior.shape)
        print(f"Loaded {len(ds.event_files)} event file(s). Dataset OK.")
    except FileNotFoundError as e:
        print("Test skipped (no data found yet):", e)