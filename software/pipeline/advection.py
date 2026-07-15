import numpy as np
import cv2


def compute_motion_field(frame_prev, frame_curr):
    
    # Farneback expects 8-bit images; rescale from our [0,1] normalized
    # rain rate. Contrast doesn't need to be perfect -- we mainly need the
    # rain-shaped structures to be trackable, not radiometrically precise.
    prev_u8 = np.clip(frame_prev * 255.0, 0, 255).astype(np.uint8)
    curr_u8 = np.clip(frame_curr * 255.0, 0, 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        prev_u8, curr_u8,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow  # (H, W, 2)


def warp_frame(frame, flow):
    """Apply a motion field to a frame via backward warping (remap).
    frame: (H, W) float array
    flow: (H, W, 2) per-pixel (dx, dy) displacement
    Returns: (H, W) warped frame.
    """
    height, width = frame.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(
        frame.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return warped


def build_advection_prior(frame_prev, frame_curr, n_steps, damping=0.0):

    flow = compute_motion_field(frame_prev, frame_curr)
    frames = []
    current = frame_curr.copy()
    for step in range(n_steps):
        current = warp_frame(current, flow)
        if damping > 0.0:
            current = current * (1.0 - damping)
        frames.append(current)
    return np.stack(frames, axis=0)  # (n_steps, H, W)