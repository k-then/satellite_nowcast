"""
Streamlit GUI for the UK rainfall nowcasting pipeline.

Self-contained apart from your three core pipeline modules --
ai_weather_model.py, advection.py, storm_tracking.py -- which just need
to be importable (either sitting next to this file, or on PYTHONPATH).
It does NOT depend on inference.py.

Data/checkpoint paths are resolved relative to THIS file's location, not
the current working directory, so it doesn't matter where you launch
`streamlit run` from. Adjust REPO_ROOT below if your folder layout
differs from software/model/app.py sitting two levels under the repo
root.

    pip install streamlit
    streamlit run app.py

Focused on historical events for now: pick a saved event_*.npy sequence,
feed the most recent real frames into the model as "now", and animate
straight through into the forecast horizon -- observed radar flowing
into predicted radar, one continuous timeline. Live data comes later.
"""

import glob
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
from pyproj import Transformer

from ai_weather_model import NowcastNet
from advection import build_advection_prior
from storm_tracking import (
    detect_cells,
    classify_convective_stratiform,
    cell_convective_fraction,
    label_storm_type,
    StormTracker,
)

# ---------------------------------------------------------------------------
# Paths -- resolved relative to this file, not the shell's cwd.
# This file lives at software/model/app.py; the repo root (containing
# data/ and checkpoints_v6/) is two levels up. Edit REPO_ROOT if that
# layout ever changes.
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

DYNAMIC_DIR = REPO_ROOT / "data" / "training" / "dynamic_layers"
STATIC_DIR = REPO_ROOT / "data" / "training" / "static_layers"
CHECKPOINT = REPO_ROOT / "checkpoints_v6" / "best.pt"

VIEW_MODES = {
    "Full model": 1.0,       # network's learned residual on top of advection
    "Advection only": 0.0,   # pure physics baseline, no learned correction
}

STORM_TYPES = [
    "isolated_convective",
    "convective_cluster",
    "mixed_convective_stratiform",
    "stratiform",
]

STORM_TYPE_COLORS = {
    "isolated_convective": "#D85A30",
    "convective_cluster": "#993C1D",
    "mixed_convective_stratiform": "#BA7517",
    "stratiform": "#378ADD",
}

# Grid geometry -- matches Static_tensor.py's target grid (1km/pixel).
X_MIN, Y_TOP = -404500.0, 1549500.0
RES_M = 1000.0
OSGB_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
WGS84_TO_OSGB = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


def pixel_to_latlon(row, col):
    x = X_MIN + col * RES_M
    y = Y_TOP - row * RES_M
    lon, lat = OSGB_TO_WGS84.transform(x, y)
    return lat, lon


def latlon_to_pixel(lat, lon):
    x, y = WGS84_TO_OSGB.transform(lon, lat)
    col = (x - X_MIN) / RES_M
    row = (Y_TOP - y) / RES_M
    return row, col


def velocity_to_speed_bearing(d_row, d_col, frame_interval_minutes):
    east_km = d_col * RES_M / 1000.0
    north_km = -d_row * RES_M / 1000.0
    dist_km = float(np.hypot(east_km, north_km))
    hours = frame_interval_minutes / 60.0
    speed_kmh = dist_km / hours if hours > 0 else 0.0
    bearing_deg = float(np.degrees(np.arctan2(east_km, north_km))) % 360.0
    return speed_kmh, bearing_deg


# ---------------------------------------------------------------------------
# Pipeline plumbing (inlined from inference.py, trimmed to what the GUI needs)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device, residual_scale=0.7):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})
    model = NowcastNet(
        dynamic_channels=2,
        static_channels=3,
        hidden_channels=tuple(train_args.get("hidden_channels", [32, 64, 64])),
        forecast_steps=train_args.get("t_out", 6),
        residual_scale=residual_scale,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, train_args


def load_static_layers(static_dir, target_shape=None):
    sea_mask = np.load(static_dir / "sea_mask.npy").astype(np.float32)
    terrain = np.load(static_dir / "terrain.npy").astype(np.float32)
    land_type = np.load(static_dir / "land_type.npy").astype(np.float32)

    if target_shape is not None:
        target_h, target_w = target_shape
        if sea_mask.shape == (target_w, target_h):
            sea_mask = sea_mask.T
            terrain = terrain.T
            land_type = land_type.T

    terrain_norm = np.clip(terrain, 0.0, 1350.0) / 1350.0
    lt_min, lt_max = land_type.min(), land_type.max()
    land_norm = (land_type - lt_min) / max(lt_max - lt_min, 1e-6)
    return np.stack([sea_mask, terrain_norm, land_norm], axis=0).astype(np.float32)


def load_input_frames(event_file, t_in, t_out=None, start=None):
    seq = np.load(event_file)
    total_t = seq.shape[0]
    if start is None:
        if seq.shape[0] < t_in:
            raise ValueError(f"{event_file} has only {seq.shape[0]} frames, need >= {t_in}")
        return seq[-t_in:].astype(np.float32), None
    if start < t_in or (t_out is not None and start + t_out > total_t):
        raise ValueError(
            f"start={start} invalid for t_in={t_in}, t_out={t_out}, total_t={total_t}."
        )
    input_frames = seq[start - t_in:start].astype(np.float32)
    true_future = seq[start:start + t_out, 0].astype(np.float32) if t_out else None
    return input_frames, true_future


def tiled_run_forecast(model, inp_frames, static_layers, device, t_out,
                        tile_size=128, overlap=32, return_all_channels=False):
    """Runs the model over the full grid in overlapping tiles and stitches
    the results together to avoid processing the whole grid at once.

    return_all_channels=False (default): output shape (t_out, H, W), radar only.
    return_all_channels=True: output shape (t_out, 2, H, W), radar + satellite/IR.
    """
    _, _, full_h, full_w = inp_frames.shape
    n_channels = 2 if return_all_channels else 1
    stride = tile_size - overlap
    output = np.zeros((t_out, n_channels, full_h, full_w), dtype=np.float32)
    weight = np.zeros((full_h, full_w), dtype=np.float32)

    ramp = np.minimum(np.arange(tile_size) + 1, tile_size - np.arange(tile_size))
    blend_1d = np.clip(ramp / (overlap + 1), 0, 1) if overlap > 0 else np.ones(tile_size)
    blend_2d = np.outer(blend_1d, blend_1d).astype(np.float32)

    tops = list(range(0, full_h - tile_size + 1, stride))
    if tops[-1] != full_h - tile_size:
        tops.append(full_h - tile_size)
    lefts = list(range(0, full_w - tile_size + 1, stride))
    if lefts[-1] != full_w - tile_size:
        lefts.append(full_w - tile_size)

    for top in tops:
        for left in lefts:
            frame_tile = inp_frames[:, :, top:top + tile_size, left:left + tile_size]
            static_tile = static_layers[:, top:top + tile_size, left:left + tile_size]

            radar_prev = frame_tile[-2, 0]
            radar_curr = frame_tile[-1, 0]
            prior = build_advection_prior(radar_prev, radar_curr, t_out)

            x = torch.from_numpy(frame_tile).unsqueeze(0).to(device)
            static_t = torch.from_numpy(static_tile).unsqueeze(0).to(device)
            prior_t = torch.from_numpy(prior[:, None]).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(x, static_t, forecast_steps=t_out, advection_prior=prior_t,
                             return_all_channels=return_all_channels)
            # pred: (1, t_out, C, tile, tile) if return_all_channels else (1, t_out, 1, tile, tile)
            pred_np = pred.squeeze(0).cpu().numpy()
            if not return_all_channels:
                pred_np = pred_np[:, 0:1]  # keep a channel axis for consistent indexing below

            for t in range(t_out):
                output[t, :, top:top + tile_size, left:left + tile_size] += (
                    pred_np[t] * blend_2d[None]
                )
            weight[top:top + tile_size, left:left + tile_size] += blend_2d

            del x, static_t, prior_t, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

    weight = np.maximum(weight, 1e-6)
    output = output / weight[None, None, :, :]
    return output if return_all_channels else output[:, 0]


def track_and_summarize(input_radar, forecast_radar, frame_interval_minutes, now=None,
                         threshold=0.1, min_area=4, max_match_distance=15.0):
    now = now or datetime.now(timezone.utc)
    t_in = input_radar.shape[0]
    full_seq = np.concatenate([input_radar, forecast_radar], axis=0)

    tracker = StormTracker(max_match_distance=max_match_distance)
    for t in range(full_seq.shape[0]):
        frame = full_seq[t]
        cells = detect_cells(frame, threshold=threshold, min_area=min_area)
        classification = classify_convective_stratiform(frame, rain_threshold=threshold)
        for cell in cells:
            cell_convective_fraction(cell, classification, frame.shape)
            cell.storm_type = label_storm_type(cell)
        tracker.update(cells)

    summaries = []
    for track in tracker.tracks:
        obs_now = next((h for h in track.history if h["frame"] == t_in - 1), None)
        if obs_now is None:
            continue

        forecast_obs = [h for h in track.history if h["frame"] >= t_in]
        vr, vc = track.velocity(n_recent=3)
        speed_kmh, bearing_deg = velocity_to_speed_bearing(vr, vc, frame_interval_minutes)
        lat, lon = pixel_to_latlon(*obs_now["centroid"])

        forecast_points = []
        for h in forecast_obs:
            f_lat, f_lon = pixel_to_latlon(*h["centroid"])
            eta_min = (h["frame"] - (t_in - 1)) * frame_interval_minutes
            forecast_points.append({
                "eta_minutes": eta_min,
                "time_utc": (now + timedelta(minutes=eta_min)).isoformat(),
                "lat": round(f_lat, 4), "lon": round(f_lon, 4),
                "intensity": round(h["intensity"], 3),
            })

        summaries.append({
            "track_id": track.track_id,
            "storm_type": obs_now["storm_type"],
            "current_position": {"lat": round(lat, 4), "lon": round(lon, 4)},
            "current_intensity": round(obs_now["intensity"], 3),
            "area_km2": obs_now["area"],
            "speed_kmh": round(speed_kmh, 1),
            "bearing_deg": round(bearing_deg, 1),
            "forecast_track": forecast_points,
        })
    return summaries


# ---------------------------------------------------------------------------
# Cached loaders -- these are the expensive bits (model weights, disk reads)
# ---------------------------------------------------------------------------

@st.cache_data
def list_events():
    files = sorted(glob.glob(str(DYNAMIC_DIR / "event_*.npy")))
    return [Path(f).name for f in files]


@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Known NIMROD uk-1km composite grid shape (rows, cols) -- matches
# Static_tensor.py's num_rows, num_cols. Passing this to load_static_layers
# lets its built-in transpose auto-fix actually run; previously it was
# called with target_shape=None so a transposed static array (a known
# issue from Static_tensor.py's swapped resolution derivation) never got
# corrected, which is what was rotating/garbling the coastline overlay.
GRID_SHAPE = (2175, 1725)


@st.cache_resource
def get_static_layers():
    return load_static_layers(STATIC_DIR, target_shape=GRID_SHAPE)


@st.cache_resource
def get_bad_pixel_mask():
    path = STATIC_DIR / "bad_pixel_mask.npy"
    return np.load(path) if path.exists() else None


@st.cache_resource
def get_model(residual_scale):
    device = get_device()
    model, train_args = load_model(CHECKPOINT, device, residual_scale=residual_scale)
    return model, train_args


@st.cache_data(show_spinner="Running forecast...")
def run_pipeline(event_name, residual_scale, frame_interval_minutes, tile_size, overlap):
    device = get_device()
    model, train_args = get_model(residual_scale)
    t_in = train_args.get("t_in", 4)
    t_out = train_args.get("t_out", 6)

    static_layers = get_static_layers()
    bad_pixel_mask = get_bad_pixel_mask()

    event_path = DYNAMIC_DIR / event_name
    # start=None -> only the most recent t_in real frames are used as input;
    # everything from here on is genuinely forecast, not compared to anything.
    input_frames, _ = load_input_frames(event_path, t_in, t_out, start=None)
    if bad_pixel_mask is not None:
        input_frames[:, 0][:, bad_pixel_mask] = 0.0

    forecast_both = tiled_run_forecast(
        model, input_frames, static_layers, device, t_out,
        tile_size=tile_size, overlap=overlap, return_all_channels=True,
    )
    forecast_radar = forecast_both[:, 0]
    forecast_cloud = forecast_both[:, 1]

    summaries = track_and_summarize(
        input_frames[:, 0], forecast_radar, frame_interval_minutes
    )

    return {
        "t_in": t_in,
        "t_out": t_out,
        "input_radar": input_frames[:, 0],
        "input_cloud": input_frames[:, 1],
        "forecast_radar": forecast_radar,
        "forecast_cloud": forecast_cloud,
        "sea_mask": static_layers[0],
        "terrain": static_layers[1],
        "summaries": summaries,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap

# Rough approximation of the blue-to-green precipitation gradient common in
# weather apps. Low values are made transparent via alpha in render_frame
# (not via the colormap itself) so the basemap shows through dry areas.
RAIN_CMAP = LinearSegmentedColormap.from_list(
    "rain", ["#eaf6fb", "#a9d6f0", "#3d85c6", "#0b3d75", "#34a853"]
)

# Cloud layer uses the raw IR-brightness-temp channel, not a true cloud-mask
# product -- colder cloud tops show up as LOW normalized values (see
# Training_data_extract.py's norm_satellite scaling), so we invert it for an
# intuitive "more cloud = whiter" look. Treat this as illustrative, not a
# calibrated cloud-fraction estimate.
CLOUD_CMAP = LinearSegmentedColormap.from_list("cloud", ["#c7ccd1", "#e8ecef", "#ffffff"])

SEA_COLOR = np.array([0.09, 0.14, 0.27])
LAND_LOW_COLOR = np.array([0.36, 0.52, 0.30])
LAND_HIGH_COLOR = np.array([0.58, 0.52, 0.38])


def build_basemap(sea_mask, terrain):
    """Cheap relief-style basemap from data we already have on disk --
    no imagery/internet required. Sea is flat dark navy; land is shaded
    green-to-brown by normalized elevation."""
    land_rgb = (
        LAND_LOW_COLOR[None, None, :]
        + (LAND_HIGH_COLOR - LAND_LOW_COLOR)[None, None, :] * terrain[..., None]
    )
    is_land = (sea_mask > 0.5)[..., None]
    return np.where(is_land, land_rgb, SEA_COLOR[None, None, :])


def render_frame(frame, sea_mask, terrain, title, layer="precipitation"):
    fig, ax = plt.subplots(figsize=(5, 5.5))

    basemap = build_basemap(sea_mask, terrain)
    ax.imshow(basemap, origin="upper")

    if layer == "precipitation":
        # mask near-zero rain so the basemap shows through dry areas,
        # instead of painting the whole grid a flat "no rain" colour
        display = np.ma.masked_less(frame, 0.03)
        im = ax.imshow(display, cmap=RAIN_CMAP, vmin=0, vmax=1, origin="upper", alpha=0.9)
    else:  # cloud
        cloud_amount = 1.0 - frame  # invert: low IR-temp (cold tops) -> more "cloud"
        display = np.ma.masked_less(cloud_amount, 0.15)
        im = ax.imshow(display, cmap=CLOUD_CMAP, vmin=0, vmax=1, origin="upper", alpha=0.85)

    # imshow and contour handle origin='upper' differently internally
    # (imshow flips the axis extent, contour flips the data values) --
    # stacking both with just origin="upper" on each causes a double
    # vertical flip. Forcing contour to share imshow's exact extent
    # sidesteps that entirely instead of relying on origin to reconcile it.
    ax.contour(sea_mask, levels=[0.5], colors="black", linewidths=0.6,
               origin="upper", extent=im.get_extent())
    # Crop to the UK region -- otherwise the full 2175x1725 grid shows,
    # which is mostly empty padding around the actual landmass.
    ax.set_aspect("equal")
    ax.set_xlim(405, 1105)
    ax.set_ylim(1550, 350)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


def frame_label(idx, t_in, frame_interval_minutes):
    eta = (idx - (t_in - 1)) * frame_interval_minutes
    if idx < t_in:
        return f"input, t{eta:+d} min"
    return f"forecast, t+{eta} min"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="UK rainfall nowcast", layout="wide")
    st.title("UK rainfall nowcast")
    st.caption("Recent radar leading into the model's forecast of what happens next")

    events = list_events()
    if not events:
        st.error(
            f"No event_*.npy files found in {DYNAMIC_DIR}. "
            "Run Training_data_extract.py first, or check REPO_ROOT at the top of app.py."
        )
        st.stop()
    if not CHECKPOINT.exists():
        st.error(f"No checkpoint found at {CHECKPOINT}. Check REPO_ROOT / CHECKPOINT at the top of app.py.")
        st.stop()

    with st.sidebar:
        st.subheader("Event")
        event_name = st.selectbox("Storm event", events)

        st.subheader("Layer")
        layer = st.radio("Show", ["Precipitation", "Cloud cover"], index=0)

        st.subheader("Storm type")
        selected_types = [
            t for t in STORM_TYPES
            if st.checkbox(t.replace("_", " "), value=True, key=f"type_{t}")
        ]

        st.subheader("View mode")
        view_mode = st.radio(
            "Show", list(VIEW_MODES.keys()), index=0,
            help="Cloud layer is always the network's own prediction -- "
                 "there's no physics prior for it, so this only affects precipitation.",
        )

        st.subheader("Settings")
        frame_interval = st.number_input("Frame interval (min)", value=5, min_value=1)
        tile_size = st.select_slider("Tile size", options=[64, 128, 256], value=128)

    residual_scale = VIEW_MODES[view_mode]
    result = run_pipeline(event_name, residual_scale, frame_interval, tile_size, 32)

    t_in, t_out = result["t_in"], result["t_out"]
    if layer == "Precipitation":
        full_seq = np.concatenate([result["input_radar"], result["forecast_radar"]], axis=0)
        layer_key = "precipitation"
    else:
        full_seq = np.concatenate([result["input_cloud"], result["forecast_cloud"]], axis=0)
        layer_key = "cloud"
    total_frames = full_seq.shape[0]
    filtered = [s for s in result["summaries"] if s["storm_type"] in selected_types]

    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = t_in

    map_col, stats_col = st.columns([3, 1])

    with map_col:
        play_col, speed_col = st.columns([1, 2])
        with play_col:
            playing = st.toggle("▶ Play", value=False, key="playing")
        with speed_col:
            frame_delay = st.slider("Playback speed", 0.1, 1.5, 0.5, 0.1,
                                     label_visibility="collapsed",
                                     help="Seconds per frame")

        if playing:
            idx = st.session_state.frame_idx
        else:
            idx = st.slider(
                "Timeline", min_value=0, max_value=total_frames - 1,
                value=min(st.session_state.frame_idx, total_frames - 1), key="scrub",
            )
            st.session_state.frame_idx = idx

        fig = render_frame(
            full_seq[idx], result["sea_mask"], result["terrain"],
            "Nowcast" if idx >= t_in else "Observed",
            layer=layer_key,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(frame_label(idx, t_in, frame_interval))

        # Advance one frame per rerun. Because this is a full rerun rather
        # than a blocking loop, the "Play" toggle above gets re-read every
        # frame -- so switching it off actually stops the animation on the
        # next tick. Stop (don't wrap) at the last forecast frame -- looping
        # back to frame 0 mid-play looked like the rain reversing direction.
        if playing:
            if idx >= total_frames - 1:
                st.session_state.playing = False
                st.rerun()
            else:
                time.sleep(frame_delay)
                st.session_state.frame_idx = idx + 1
                st.rerun()

    with stats_col:
        st.subheader("Storm tracks")
        st.metric("Active tracks", len(filtered))
        if filtered:
            lead = max(filtered, key=lambda s: s["current_intensity"])
            st.metric("Lead storm speed", f"{lead['speed_kmh']} km/h")
            st.metric("Lead storm bearing", f"{lead['bearing_deg']}°")
        st.caption(f"Forecast horizon: {t_out * frame_interval} min ahead")

        with st.expander("Track details"):
            st.json(filtered if filtered else result["summaries"])


if __name__ == "__main__":
    main()