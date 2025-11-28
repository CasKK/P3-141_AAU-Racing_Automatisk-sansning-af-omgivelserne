import numpy as np
import pandas as pd
import os

# ---------------- Camera parameters (match your script) ----------------
FOV_DEG = 90            # horizontal FOV in degrees
IMG_W, IMG_H = 1280, 720
ANGLE_PER_PIXEL = np.deg2rad(FOV_DEG) / IMG_W

# ---------------- Track & scenario parameters ----------------
R_CENTER = 100.0        # circle radius (centerline), meters
LANE_OFFSET = 5.0       # cone rows offset from centerline, meters
N_FRAMES = 360          # one lap, 1 deg per frame for smoothness
MAX_RANGE_M = 80.0      # depth sensor range
N_CONES = 600           # cones around the loop per side

# Optional realism knobs
ADD_NOISE = False       # set True to add slight detection noise
PIXEL_NOISE_PX = 1.0
DEPTH_NOISE_M = 0.05    # 5 cm std

# -------------- Helper geometry functions (circle) --------------
def centerline(theta):
    """Point on circular centerline (meters)."""
    return np.array([R_CENTER * np.cos(theta), R_CENTER * np.sin(theta)])

def forward_dir(theta):
    """Vehicle forward unit vector along circular path (tangent)."""
    t = np.array([-np.sin(theta), np.cos(theta)])
    n = np.linalg.norm(t)
    return t / n

def left_normal(theta):
    """Left normal (unit, 90 deg CCW from forward)."""
    f = forward_dir(theta)
    return np.array([-f[1], f[0]])

# -------------- Cones on left/right boundaries -----------------
# Cones are placed evenly around the circle on both sides
cone_angles = np.linspace(0, 2*np.pi, N_CONES, endpoint=False)
cones_world = []  # list of (x, y, side)
for th in cone_angles:
    c = centerline(th)
    nL = left_normal(th)
    left_pt = c + nL * LANE_OFFSET
    right_pt = c - nL * LANE_OFFSET
    cones_world.append((left_pt[0], left_pt[1], 'left'))
    cones_world.append((right_pt[0], right_pt[1], 'right'))

cones_world = np.array(cones_world, dtype=object)

# -------------- Synthetic camera projection --------------------
rows = []
for i in range(N_FRAMES):
    th_car = 2 * np.pi * i / N_FRAMES
    car_pos = centerline(th_car)
    fwd = forward_dir(th_car)  # unit

    for (cx, cy, side) in cones_world:
        rel = np.array([cx, cy]) - car_pos
        dist = np.linalg.norm(rel)
        if dist <= 0.1 or dist > MAX_RANGE_M:
            continue

        # Signed bearing using 2D cross/dot (relative to car forward)
        dot = np.dot(fwd, rel)
        cross = fwd[0] * rel[1] - fwd[1] * rel[0]
        bearing = np.arctan2(cross, dot)  # (-pi, pi)

        # Must be within horizontal FOV
        if abs(bearing) > np.deg2rad(FOV_DEG / 2):
            continue

        # Project to pixel_x with your FOV model
        px = IMG_W / 2 + bearing / ANGLE_PER_PIXEL

        # Simple vertical placement: nearer -> lower in image
        # (Your math doesn’t use pixel_y; this is for plausibility)
        y_frac = 0.85 - 0.5 * min(1.0, dist / MAX_RANGE_M)  # in [0.35, 0.85]
        py = IMG_H * y_frac

        # Optional noise
        if ADD_NOISE:
            px += np.random.normal(0, PIXEL_NOISE_PX)
            py += np.random.normal(0, PIXEL_NOISE_PX)
            dist_eff = max(0.0, dist + np.random.normal(0, DEPTH_NOISE_M))
        else:
            dist_eff = dist

        # Clip to image bounds and finalize
        px_i = int(np.clip(np.round(px), 0, IMG_W - 1))
        py_i = int(np.clip(np.round(py), 0, IMG_H - 1))
        depth_mm = int(np.round(dist_eff * 1000))

        rows.append({
            'frame': i,
            'pixel_x': px_i,
            'pixel_y': py_i,
            'depth_mm': depth_mm,
            'cone_type': side
        })

# -------------- Save CSV ---------------------------------------
df = pd.DataFrame(rows)
out_dir = os.path.join('Base', 'MR113_Position')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'nascar_track_cones_dataset_synth.csv')
df.to_csv(out_path, index=False)

# -------------- Console summary --------------------------------
print(f"Saved dataset to: {out_path}")
print(f"Frames with detections: {df['frame'].nunique()} / {N_FRAMES}")
print(f"Rows total: {len(df)}")
if not df.empty:
    print(f"Depth range (mm): {df['depth_mm'].min()} – {df['depth_mm'].max()}")
    print("Preview:")
    print(df.head(10).to_string(index=False))
else:
    print("Generated an empty dataset (check FOV/range parameters).")