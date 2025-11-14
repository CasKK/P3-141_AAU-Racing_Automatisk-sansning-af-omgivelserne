import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D
import numpy as np

# ---------------------------
# Parametre
# ---------------------------
track_width = 750
car_width = 375
car_length = 645
centrum = 1999.5

# ---------------------------
# Figur
# ---------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-3000, 8000)
ax.set_ylim(0, 5000)
ax.set_aspect('equal')

# ---------------------------
# Banen
# ---------------------------
# Linjer
ax.plot([0, 5000], [500, 500], 'k--', linewidth=3)
ax.plot([0, 5000], [3125 + track_width/2, 3125 + track_width/2], 'k--', linewidth=3)
ax.plot([0, 5000], [500 + track_width/2, 500 + track_width/2], 'k-', linewidth=2)
ax.plot([0, 5000], [500 - track_width/2, 500 - track_width/2], 'k-', linewidth=2)
ax.plot([0, 5000], [3125, 3125], 'k--', linewidth=2)
ax.plot([0, 5000], [3125 + track_width, 3125 + track_width], 'k--', linewidth=2)

# Safezones som rektangler
safezones_rect = [
    (0, 500 - track_width/2, 5000, 100),
    (0, 500 + track_width/2 - 100, 5000, 100),
    (0, 3500 - track_width/2, 5000, 100),
    (0, 3500 + track_width/2 - 100, 5000, 100)
]
for sx, sy, sw, sh in safezones_rect:
    ax.add_patch(Rectangle((sx, sy), sw, sh, edgecolor='red', facecolor='red'))

# Safezones som halvcirkler (bruger dine præcise værdier)
halv_cirkler = [
    (5000, centrum, 1875, -90, -270),  # halfOuter1
    (0, centrum, 1875, 90, 270),       # halfOuter2
    (5000, centrum, 1125, -90, -270),  # halfInner1
    (0, centrum, 1125, 90, 270),       # halfInner2
    (5000, centrum, 1820, -90, -270),  # halfOuterSafe1
    (0, centrum, 1820, 90, 270),       # halfOuterSafe2
    (5000, centrum, 1180, -90, -270),  # halfInnerSafe1
    (0, centrum, 1180, 90, 270),       # halfInnerSafe2
]

for cx, cy, r, t1, t2 in halv_cirkler:
    ax.add_patch(Arc((cx, cy), 2*r, 2*r, angle=0, theta1=t1, theta2=t2, edgecolor='red', linewidth=3))

# ---------------------------
# Banen som punkter
# ---------------------------
xstraight1 = np.linspace(0, 5000, 100)
ystraight1 = np.full_like(xstraight1, 500)

theta_arc1 = np.linspace(-np.pi/2, np.pi/2, 100)
xarc1 = 5000 + np.cos(theta_arc1) * 1500
yarc1 = centrum + np.sin(theta_arc1) * 1500

xstraight2 = np.linspace(5000, 0, 100)
ystraight2 = np.full_like(xstraight2, 3500)

theta_arc2 = np.linspace(np.pi/2, 3*np.pi/2, 100)
xarc2 = 0 + np.cos(theta_arc2) * 1500
yarc2 = centrum + np.sin(theta_arc2) * 1500

trackx = np.concatenate([xstraight1, xarc1, xstraight2, xarc2])
tracky = np.concatenate([ystraight1, yarc1, ystraight2, yarc2])

dx = np.gradient(trackx)
dy = np.gradient(tracky)
angles = np.degrees(np.arctan2(dy, dx))

# ---------------------------
# Bil
# ---------------------------
car = Rectangle((trackx[0] - car_length/2, tracky[0] - car_width/2),
                car_length, car_width, edgecolor='blue', facecolor='lightblue')
ax.add_patch(car)

# ---------------------------
# Afstand funktioner
# ---------------------------
def distance_to_rect(bx, by, bw, bl, rx, ry, rw, rh):
    dx = max(rx - (bx + bl/2), 0, (bx - bl/2) - (rx + rw))
    dy = max(ry - (by + bw/2), 0, (by - bw/2) - (ry + rh))
    return np.sqrt(dx**2 + dy**2)

def distance_to_circle(bx, by, bw, bl, cx, cy, r):
    # Brug midten af bilen
    return max(np.sqrt((bx - cx)**2 + (by - cy)**2) - r, 0)

# ---------------------------
# Animation
# ---------------------------
total_frames = len(trackx)
min_distances = []

def update(frame):
    x = trackx[frame % total_frames]
    y = tracky[frame % total_frames]
    angle = angles[frame % total_frames]

    # Opdater bilens position og rotation
    trans = Affine2D().rotate_deg_around(x, y, angle) + ax.transData
    car.set_xy((x - car_length/2, y - car_width/2))
    car.set_transform(trans)

    # Beregn afstand til alle safezones
    distances = []
    for sx, sy, sw, sh in safezones_rect:
        distances.append(distance_to_rect(x, y, car_width, car_length, sx, sy, sw, sh))
    for cx, cy, r, _, _ in halv_cirkler:
        distances.append(distance_to_circle(x, y, car_width, car_length, cx, cy, r))
    
    min_distances.append(min(distances))
    return car,

ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True, repeat=False)
plt.show()

# Mindste målte afstand
print(f"Mindste målte afstand til safezones: {min(min_distances):.2f} mm")
