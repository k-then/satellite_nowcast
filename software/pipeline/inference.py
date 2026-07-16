
import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import torch
from pyproj import Transformer

from model import NowcastNet
from advection import build_advection_prior
from storm_tracking import detect_cells, classify_convective_stratiform, \
    cell_convective_fraction, label_storm_type, StormTracker
from datetime import timezone

"""
Grid geometry -- matches Static_tensor.py's target grid. Resolution is
fixed at 1000m/pixel here (the true "uk-1km" resolution) rather than
derived from num_rows/num_cols, since that derivation is currently swapped
in Static_tensor.py (see note above).
"""
X_MIN, Y_TOP = -404500.0, 1549500.0
RES_M = 1000.0
OSGB_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


# Translates a 2D grid pixel index back to physical WGS84 Latitude and Longitude coordinates.
def pixel_to_latlon(row, col):
    x = X_MIN + col * RES_M
    y = Y_TOP - row * RES_M
    lon, lat = OSGB_TO_WGS84.transform(x, y)
    return lat, lon


# Converts 2D pixel displacement per frame step into physical Speed (km/h) and Meteorological Bearing (degrees clockwise from True North).
def velocity_to_speed_bearing(d_row, d_col, frame_interval_minutes):
    east_km = d_col * RES_M / 1000.0
    north_km = -d_row * RES_M / 1000.0
    dist_km = float(np.hypot(east_km, north_km))
    hours = frame_interval_minutes / 60.0
    speed_kmh = dist_km / hours if hours > 0 else 0.0
    bearing_deg = float(np.degrees(np.arctan2(east_km, north_km))) % 360.0
    return speed_kmh, bearing_deg


# Loads checkpoint, retrieves saved training parameters, and builds NowcastNet in eval mode.
def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})
    model = NowcastNet(
        dynamic_channels=2,
        static_channels=3,
        hidden_channels=tuple(train_args.get("hidden_channels", [32, 64, 64])),
        forecast_steps=train_args.get("t_out", 6),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, train_args


# Loads, normalizes, and stacks geographical land layers into a (3, H, W) tensor.
def load_static_layers(static_dir):
    sea_mask = np.load(os.path.join(static_dir, "sea_mask.npy")).astype(np.float32)
    terrain = np.load(os.path.join(static_dir, "terrain.npy")).astype(np.float32)
    land_type = np.load(os.path.join(static_dir, "land_type.npy")).astype(np.float32)
    terrain_norm = np.clip(terrain, 0.0, 1350.0) / 1350.0
    lt_min, lt_max = land_type.min(), land_type.max()
    land_norm = (land_type - lt_min) / max(lt_max - lt_min, 1e-6)
    return np.stack([sea_mask, terrain_norm, land_norm], axis=0).astype(np.float32)  # (3, H, W)


# Loads the sequential radar data array and slices the most recent t_in frames.
def load_input_frames(event_file, t_in):
    seq = np.load(event_file)
    if seq.shape[0] < t_in:
        raise ValueError(f"{event_file} has only {seq.shape[0]} frames, need >= {t_in}")
    return seq[-t_in:].astype(np.float32)


"""
Prepares input tensors, builds the advection physical prior baseline, 
and passes them through the neural network to output the forecasted maps.
"""
@torch.no_grad()
def run_forecast(model, input_frames, static_layers, device, t_out):
    
    x = torch.from_numpy(input_frames).unsqueeze(0).to(device)         # (1, t_in, 2, H, W)
    static = torch.from_numpy(static_layers).unsqueeze(0).to(device)   # (1, 3, H, W)

    radar_prev, radar_curr = input_frames[-2, 0], input_frames[-1, 0]
    prior = build_advection_prior(radar_prev, radar_curr, t_out)       # (t_out, H, W)
    prior_t = torch.from_numpy(prior[:, None]).unsqueeze(0).to(device)  # (1, t_out, 1, H, W)

    pred = model(x, static, forecast_steps=t_out, advection_prior=prior_t)  # (1, t_out, 1, H, W)
    return pred.squeeze(0).squeeze(1).cpu().numpy()  # (t_out, H, W)


"""
    Links segmented storm cells across historical and forecasted timesteps.
    Generates velocity statistics and projected latitude/longitude coordinate points.
"""
def track_and_summarize(input_radar, forecast_radar, frame_interval_minutes, now=None,
                         threshold=0.1, min_area=4, max_match_distance=15.0):
    
    now = now or datetime.now(timezone.utc)
    t_in = input_radar.shape[0]
    t_out = forecast_radar.shape[0]
    full_seq = np.concatenate([input_radar, forecast_radar], axis=0)  # (t_in + t_out, H, W)

    tracker = StormTracker(max_match_distance=max_match_distance)

    # Steps through every frame (past and future) to construct coherent storm tracks
    for t in range(full_seq.shape[0]):
        frame = full_seq[t]
        cells = detect_cells(frame, threshold=threshold, min_area=min_area)
        classification = classify_convective_stratiform(frame, rain_threshold=threshold)
        for cell in cells:
            cell_convective_fraction(cell, classification, frame.shape)
            cell.storm_type = label_storm_type(cell)
        tracker.update(cells)

    summaries = []
    # Analyze and serialize complete storm tracks
    for track in tracker.tracks:
        obs_now = next((h for h in track.history if h["frame"] == t_in - 1), None)
        if obs_now is None:
            continue

        """
        this track wasn't present "now" -- either dissipated already
        or only appeared later in the forecast (a newly-triggered
        cell the network predicted forming from an existing one;
        still physically meaningful, but we report it separately below)
        """
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
            "area_km2": obs_now["area"],  # 1 pixel = 1 km^2 at 1km resolution
            "speed_kmh": round(speed_kmh, 1),
            "bearing_deg": round(bearing_deg, 1),
            "forecast_track": forecast_points,
        })

    return summaries


# Generates and saves visual PNG plots mapping forecast frames with tracked overlays.
def plot_forecast_maps(input_radar, forecast_radar, summaries, output_dir,
                        frame_interval_minutes, sea_mask=None):
    
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    t_out = forecast_radar.shape[0]

    for t in range(t_out):
        fig, ax = plt.subplots(figsize=(8, 10))
        if sea_mask is not None:
            ax.imshow(sea_mask, cmap="Greys", alpha=0.15)
        ax.imshow(forecast_radar[t], cmap="YlOrRd", vmin=0, vmax=1, alpha=0.85)

        eta_min = (t + 1) * frame_interval_minutes
        for storm in summaries:
            points = [p for p in storm["forecast_track"] if p["eta_minutes"] <= eta_min]
            if not points:
                continue
            latest = points[-1]
            # re-projects the forecast lat/lon back to pixel space just for plotting
            x, y = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True).transform(
                latest["lon"], latest["lat"]
            )
            col = (x - X_MIN) / RES_M
            row = (Y_TOP - y) / RES_M
            ax.plot(col, row, marker="x", color="blue", markersize=10, mew=2)
            ax.annotate(f"#{storm['track_id']} {storm['storm_type']}",
                        (col, row), color="blue", fontsize=8, xytext=(5, 5),
                        textcoords="offset points")

        ax.set_title(f"Forecast t+{eta_min} min")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"forecast_t{eta_min:03d}min.png"), dpi=120)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--event-file", required=True)
    p.add_argument("--static-dir", default="data/training/static_layers")
    p.add_argument("--output-dir", default="forecast_output")
    p.add_argument("--frame-interval-minutes", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load neural network configurations
    model, train_args = load_model(args.checkpoint, device)
    t_in = train_args.get("t_in", 4)
    t_out = train_args.get("t_out", 6)

    # Reads inputs
    static_layers = load_static_layers(args.static_dir)
    input_frames = load_input_frames(args.event_file, t_in)

    # Runs Nowcast model
    forecast_radar = run_forecast(model, input_frames, static_layers, device, t_out)

    # Performs storm tracking on complete temporal line (historical + predicted)
    summaries = track_and_summarize(
        input_frames[:, 0], forecast_radar, args.frame_interval_minutes
    )

    # Exports structured prediction metrics
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "storm_summary.json"), "w") as f:
        json.dump({"generated_at_utc": datetime.utcnow().isoformat(), "storms": summaries},
                   f, indent=2)

    # Renders visualizations
    plot_forecast_maps(
        input_frames[:, 0], forecast_radar, summaries, args.output_dir,
        args.frame_interval_minutes, sea_mask=static_layers[0],
    )

    print(f"Wrote {len(summaries)} storm summaries and {t_out} forecast maps to {args.output_dir}/")


if __name__ == "__main__":
    main()