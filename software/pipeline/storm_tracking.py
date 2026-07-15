from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

@dataclass
class Cell:
    """One storm cell detected in a single frame."""
    centroid: tuple          # (row, col) in pixels
    area: int                # pixel count
    max_intensity: float     # peak normalized rain rate in the cell
    mean_intensity: float
    bbox: tuple              # (row_min, row_max, col_min, col_max)
    convective_fraction: float = 0.0  # filled in by classify_convective_stratiform


def detect_cells(radar_frame, threshold=0.1, min_area=4):

    mask = radar_frame >= threshold
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return []

    cells = []
    objects = ndimage.find_objects(labeled)
    for label_id in range(1, num_features + 1):
        slc = objects[label_id - 1]
        cell_mask = labeled[slc] == label_id
        area = int(cell_mask.sum())
        if area < min_area:
            continue

        cell_values = radar_frame[slc][cell_mask]
        # centroid in full-frame pixel coordinates
        rows, cols = np.nonzero(cell_mask)
        centroid = (float(rows.mean()) + slc[0].start, float(cols.mean()) + slc[1].start)
        bbox = (slc[0].start, slc[0].stop, slc[1].start, slc[1].stop)

        cells.append(Cell(
            centroid=centroid,
            area=area,
            max_intensity=float(cell_values.max()),
            mean_intensity=float(cell_values.mean()),
            bbox=bbox,
        ))
    return cells


# ---------------------------------------------------------------------------
# 2. Convective / stratiform classification (Steiner et al. 1995)
# ---------------------------------------------------------------------------

def classify_convective_stratiform(
    radar_frame,
    rain_threshold=0.1,
    core_threshold=0.5,
    background_radius=11,
    influence_radii=((0.6, 5), (0.4, 3), (0.0, 2)),
):

    rain_mask = radar_frame >= rain_threshold
    if not rain_mask.any():
        return np.zeros_like(radar_frame, dtype=np.uint8)

    # Local background: mean rain rate in a neighborhood, ignoring dry pixels
    # so isolated cells sitting in a mostly-dry area don't get an
    # artificially low background that makes everything look "peaked".
    wet = radar_frame * rain_mask
    wet_count = ndimage.uniform_filter(rain_mask.astype(np.float32), size=background_radius)
    wet_sum = ndimage.uniform_filter(wet, size=background_radius)
    background = np.divide(
        wet_sum, wet_count, out=np.zeros_like(wet_sum), where=wet_count > 1e-6
    )

    # Peakedness margin as a step function of background intensity: higher
    # background -> smaller margin needed to count as a convective core.
    # Explicit margin schedule (background level -> required peakedness):
    margin = np.select(
        [background >= 0.6, background >= 0.4, background >= 0.2],
        [0.10, 0.20, 0.35],
        default=0.5,
    ).astype(np.float32)

    core_mask = rain_mask & (
        (radar_frame >= core_threshold) | (radar_frame - background >= margin)
    )

    result = np.where(rain_mask, 1, 0).astype(np.uint8)  # start: all rain = stratiform
    if not core_mask.any():
        return result

    # Grow each core by an intensity-dependent influence radius using
    # per-core dilation: cores in more intense background get a larger
    # radius of convective influence around them.
    convective_mask = np.zeros_like(core_mask)
    remaining_cores = core_mask.copy()
    for bg_level, radius in influence_radii:
        this_tier = remaining_cores & (background >= bg_level)
        if not this_tier.any():
            continue
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        dilated = ndimage.binary_dilation(this_tier, structure=structure)
        convective_mask |= dilated
        remaining_cores &= ~this_tier

    convective_mask &= rain_mask  # don't tag dry pixels even if inside a dilation radius
    result[convective_mask] = 2
    return result


def cell_convective_fraction(cell: Cell, classification, radar_frame_shape):

    r0, r1, c0, c1 = cell.bbox
    region = classification[r0:r1, c0:c1]
    wet = region > 0
    if wet.sum() == 0:
        cell.convective_fraction = 0.0
    else:
        cell.convective_fraction = float((region == 2).sum()) / float(wet.sum())
    return cell.convective_fraction


def label_storm_type(cell: Cell):
    
    if cell.convective_fraction < 0.15:
        return "stratiform"
    if cell.convective_fraction >= 0.6 and cell.area < 400:
        return "isolated_convective"
    if cell.convective_fraction >= 0.6 and cell.area >= 400:
        return "convective_cluster"
    return "mixed_convective_stratiform"


# ---------------------------------------------------------------------------
# 3. Tracking + trajectory extrapolation
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    history: List[dict] = field(default_factory=list)  # each: {frame, centroid, area, intensity, storm_type}
    frames_since_seen: int = 0

    @property
    def last_centroid(self):
        return self.history[-1]["centroid"]

    def velocity(self, n_recent=3):
        
        obs = [h for h in self.history if h.get("centroid") is not None]
        if len(obs) < 2:
            return (0.0, 0.0)
        obs = obs[-(n_recent + 1):]
        d_rows, d_cols = [], []
        for a, b in zip(obs[:-1], obs[1:]):
            d_rows.append(b["centroid"][0] - a["centroid"][0])
            d_cols.append(b["centroid"][1] - a["centroid"][1])
        return (float(np.mean(d_rows)), float(np.mean(d_cols)))

    def predict_trajectory(self, n_steps, n_recent=3):
        
        vr, vc = self.velocity(n_recent=n_recent)
        r0, c0 = self.last_centroid
        return [(r0 + vr * (t + 1), c0 + vc * (t + 1)) for t in range(n_steps)]


class StormTracker:
 
    def __init__(self, max_match_distance=15.0, max_frames_lost=2):
        
        self.max_match_distance = max_match_distance
        self.max_frames_lost = max_frames_lost
        self.tracks: List[Track] = []
        self._next_id = 0
        self.frame_idx = -1

    def update(self, cells: List[Cell]):
        
        self.frame_idx += 1
        active_tracks = [t for t in self.tracks if t.frames_since_seen <= self.max_frames_lost]

        assignments = []
        if active_tracks and cells:
            cost = np.zeros((len(active_tracks), len(cells)))
            for i, track in enumerate(active_tracks):
                # predict where the track *should* be this frame using its
                # current velocity, rather than matching against its stale
                # last-seen position -- meaningfully better for fast movers.
                pred = track.predict_trajectory(1)[0] if len(track.history) >= 2 else track.last_centroid
                for j, cell in enumerate(cells):
                    dist = np.hypot(pred[0] - cell.centroid[0], pred[1] - cell.centroid[1])
                    cost[i, j] = dist

            row_idx, col_idx = linear_sum_assignment(cost)
            matched_tracks, matched_cells = set(), set()
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= self.max_match_distance:
                    assignments.append((active_tracks[r], cells[c]))
                    matched_tracks.add(r)
                    matched_cells.add(c)

            unmatched_cells = [cells[j] for j in range(len(cells)) if j not in matched_cells]
        else:
            unmatched_cells = cells

        matched_track_ids = {id(t) for t, _ in assignments}
        for track in self.tracks:
            if id(track) in matched_track_ids:
                continue
            track.frames_since_seen += 1

        for track, cell in assignments:
            track.history.append({
                "frame": self.frame_idx, "centroid": cell.centroid,
                "area": cell.area, "intensity": cell.max_intensity,
                "storm_type": getattr(cell, "storm_type", None),
            })
            track.frames_since_seen = 0

        for cell in unmatched_cells:
            new_track = Track(track_id=self._next_id)
            self._next_id += 1
            new_track.history.append({
                "frame": self.frame_idx, "centroid": cell.centroid,
                "area": cell.area, "intensity": cell.max_intensity,
                "storm_type": getattr(cell, "storm_type", None),
            })
            self.tracks.append(new_track)
            assignments.append((new_track, cell))

        self.tracks = [t for t in self.tracks if t.frames_since_seen <= self.max_frames_lost]
        return assignments

    def active_tracks(self):
        """Tracks currently being followed (seen this frame or recently)."""
        return [t for t in self.tracks if t.frames_since_seen == 0]


def process_sequence(radar_sequence, threshold=0.1, min_area=4, max_match_distance=15.0):
    
    tracker = StormTracker(max_match_distance=max_match_distance)
    all_frame_cells = []
    for t in range(radar_sequence.shape[0]):
        frame = radar_sequence[t]
        cells = detect_cells(frame, threshold=threshold, min_area=min_area)
        classification = classify_convective_stratiform(frame, rain_threshold=threshold)
        for cell in cells:
            cell_convective_fraction(cell, classification, frame.shape)
            cell.storm_type = label_storm_type(cell)
        tracker.update(cells)
        all_frame_cells.append(cells)
    return tracker, all_frame_cells


if __name__ == "__main__":
    # Smoke test with synthetic data: a single blob translating diagonally
    # across 6 frames, so we can sanity-check detection + tracking + the
    # trajectory forecast without needing real radar data on disk.
    H, W = 80, 80
    frames = []
    for t in range(6):
        frame = np.zeros((H, W), dtype=np.float32)
        cy, cx = 20 + 4 * t, 15 + 3 * t
        yy, xx = np.mgrid[0:H, 0:W]
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6.0 ** 2)))
        frame += blob * 0.9
        frames.append(frame)
    seq = np.stack(frames, axis=0)

    tracker, all_cells = process_sequence(seq, threshold=0.1, min_area=4)
    print(f"Detected {len(tracker.tracks)} total track(s) over {seq.shape[0]} frames")
    for track in tracker.tracks:
        types = [h["storm_type"] for h in track.history]
        print(f"  track {track.track_id}: {len(track.history)} obs, types={types}")
        forecast = track.predict_trajectory(n_steps=3)
        print(f"    next-3-frame trajectory forecast: {forecast}")