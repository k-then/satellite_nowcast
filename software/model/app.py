import os
import sys
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go

# Add model path if app.py is in root
sys.path.append(os.path.abspath("software/model"))

from ai_weather_model import NowcastNet
from advection import build_advection_prior
from storm_tracking import detect_cells, classify_convective_stratiform, cell_convective_fraction, label_storm_type, StormTracker
from inference import load_static_layers, pixel_to_latlon, velocity_to_speed_bearing

# -----------------------------------------------------------------------------
# 1. PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Weather Nowcasting Engine", layout="wide")
st.title("⚡ AI Weather Nowcasting & Storm Tracking System")

# -----------------------------------------------------------------------------
# 2. CACHED MODEL & STATIC DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model(checkpoint_path="software/model/checkpoints/best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        train_args = checkpoint.get("args", {})
        model = NowcastNet(
            dynamic_channels=2,
            static_channels=3,
            hidden_channels=tuple(train_args.get("hidden_channels", [32, 64, 64])),
            forecast_steps=train_args.get("t_out", 6),
            residual_scale=0.7,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, device, train_args
    else:
        # Fallback dummy initialization for testing UI without trained checkpoint
        model = NowcastNet(dynamic_channels=2, static_channels=3, hidden_channels=(16, 32), forecast_steps=6).to(device)
        model.eval()
        return model, device, {"t_in": 4, "t_out": 6}

@st.cache_data
def get_static_data():
    static_dir = "data/training/static_layers"
    if os.path.exists(os.path.join(static_dir, "sea_mask.npy")):
        return load_static_layers(static_dir)
    else:
        # Fallback synthetic background if static layer files aren't in path
        return np.zeros((3, 128, 128), dtype=np.float32)

model, device, train_args = get_model()
static_layers = get_static_data()
t_in = train_args.get("t_in", 4)
t_out = train_args.get("t_out", 6)
frame_interval = 5  # minutes per frame

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Map Controls")

show_radar = st.sidebar.checkbox("Show Radar Reflectivity", value=True)
show_sat = st.sidebar.checkbox("Show IR Satellite", value=False)
show_storms = st.sidebar.checkbox("Overlay Storm Tracks", value=True)
residual_scale = st.sidebar.slider("AI Correction Scale vs Physics Prior", 0.0, 1.0, 0.7, 0.05)

# Event Selector
st.sidebar.header("📁 Data Input")
data_mode = st.sidebar.radio("Select Data Source", ["Synthetic Demo Storm", "Local Event File (.npy)"])

if data_mode == "Local Event File (.npy)":
    event_file = st.sidebar.text_input("NPY File Path", "data/training/dynamic_layers/event_sample.npy")
    if os.path.exists(event_file):
        seq = np.load(event_file)
        input_frames = seq[:t_in].astype(np.float32)
    else:
        st.sidebar.warning("File not found! Using synthetic data instead.")
        data_mode = "Synthetic Demo Storm"

if data_mode == "Synthetic Demo Storm":
    # Generate mock 128x128 storm moving diagonally across t_in frames
    input_frames = np.zeros((t_in, 2, 128, 128), dtype=np.float32)
    for t in range(t_in):
        cy, cx = 40 + t * 4, 30 + t * 5
        yy, xx = np.mgrid[0:128, 0:128]
        blob = np.exp(-(((yy - cy)**2 + (xx - cx)**2) / (2 * 8.0**2)))
        input_frames[t, 0] = blob * 0.8  # Radar
        input_frames[t, 1] = blob * 0.5  # Satellite

# -----------------------------------------------------------------------------
# 4. INFERENCE & STORM TRACKING EXECUTION
# -----------------------------------------------------------------------------
@st.cache_data
def compute_forecast(_model_ref, input_frames, static_layers, residual_scale):
    # Slice crop to match static shape if needed
    h, w = input_frames.shape[-2:]
    static_crop = static_layers[:, :h, :w]
    
    # 1. Build Advection Prior
    radar_prev, radar_curr = input_frames[-2, 0], input_frames[-1, 0]
    prior = build_advection_prior(radar_prev, radar_curr, t_out)
    
    # 2. Run NowcastNet Neural Model
    x = torch.from_numpy(input_frames).unsqueeze(0).to(device)
    static_t = torch.from_numpy(static_crop).unsqueeze(0).to(device)
    prior_t = torch.from_numpy(prior[:, None]).unsqueeze(0).to(device)
    
    model.residual_scale = residual_scale
    with torch.no_grad():
        pred = model(x, static_t, forecast_steps=t_out, advection_prior=prior_t)
        forecast_radar = pred.squeeze(0).squeeze(1).cpu().numpy()
        
    # 3. Track Storms across historical + forecast frames
    full_radar = np.concatenate([input_frames[:, 0], forecast_radar], axis=0)
    tracker = StormTracker(max_match_distance=15.0)
    
    for t in range(full_radar.shape[0]):
        frame = full_radar[t]
        cells = detect_cells(frame, threshold=0.1, min_area=4)
        classification = classify_convective_stratiform(frame, rain_threshold=0.1)
        for cell in cells:
            cell_convective_fraction(cell, classification, frame.shape)
            cell.storm_type = label_storm_type(cell)
        tracker.update(cells)
        
    return forecast_radar, tracker.tracks, full_radar

forecast_radar, tracks, full_radar = compute_forecast(model, input_frames, static_layers, residual_scale)

# -----------------------------------------------------------------------------
# 5. TIME SLIDER & MAP RENDER
# -----------------------------------------------------------------------------
total_frames = t_in + t_out
time_labels = [f"t-{ (t_in - t - 1) * frame_interval }m (Observed)" for t in range(t_in)] + \
              [f"t+{ (t + 1) * frame_interval }m (AI Forecast)" for t in range(t_out)]

frame_idx = st.select_slider("⏱️ Timeline Scrub Controls", options=list(range(total_frames)), 
                            format_func=lambda i: time_labels[i], value=t_in - 1)

# Status Badge
if frame_idx < t_in:
    st.info(f"Displaying **Observed Historical Frame** ({time_labels[frame_idx]})")
else:
    st.success(f"Displaying **Neural Nowcast Prediction** ({time_labels[frame_idx]})")

# Plotly Map Construction
fig = go.Figure()

current_radar = full_radar[frame_idx]

# Satellite Layer
if show_sat:
    sat_frame = input_frames[frame_idx if frame_idx < t_in else -1, 1]
    fig.add_trace(go.Heatmap(z=sat_frame, colorscale="Inferno", opacity=0.5, showscale=False, name="Satellite IR"))

# Radar Layer
if show_radar:
    # Mask dry areas for UI transparency
    masked_radar = np.where(current_radar > 0.02, current_radar, np.nan)
    fig.add_trace(go.Heatmap(z=masked_radar, colorscale="YlOrRd", zmin=0.0, zmax=1.0, opacity=0.85, name="Radar"))

# Storm Trajectory Vectors
if show_storms and tracks:
    for tr in tracks:
        # Extract track points up to current frame index
        points = [h for h in tr.history if h["frame"] <= frame_idx]
        if points:
            cols = [h["centroid"][1] for h in points]
            rows = [h["centroid"][0] for h in points]
            
            # Plot path history line
            fig.add_trace(go.Scatter(x=cols, y=rows, mode="lines+markers",
                                     marker=dict(size=6), line=dict(width=2, dash="dot"),
                                     name=f"Storm #{tr.track_id} ({points[-1]['storm_type']})"))

fig.update_layout(
    width=800,
    height=600,
    yaxis=dict(autorange="reversed"),  # Top-down matrix layout matching PyTorch/Numpy
    xaxis_title="Grid Pixel X",
    yaxis_title="Grid Pixel Y",
    margin=dict(l=20, r=20, t=30, b=20)
)

col_map, col_info = st.columns([2, 1])

with col_map:
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.subheader("🌩️ Active Storm Cells")
    active_in_frame = [tr for tr in tracks if any(h["frame"] == frame_idx for h in tr.history)]
    
    if not active_in_frame:
        st.write("No active convective cells detected in this frame.")
    else:
        for tr in active_in_frame:
            latest = next(h for h in tr.history if h["frame"] == frame_idx)
            vr, vc = tr.velocity(n_recent=3)
            speed, bearing = velocity_to_speed_bearing(vr, vc, frame_interval)
            
            with st.expander(f"Storm Cell ID #{tr.track_id}", expanded=True):
                st.write(f"**Classification:** `{latest.get('storm_type', 'N/A')}`")
                st.write(f"**Peak Intensity:** `{latest['intensity']:.2f}`")
                st.write(f"**Estimated Area:** `{latest['area']} km²`")
                st.write(f"**Speed / Heading:** `{speed:.1f} km/h` @ `{bearing:.0f}°`")