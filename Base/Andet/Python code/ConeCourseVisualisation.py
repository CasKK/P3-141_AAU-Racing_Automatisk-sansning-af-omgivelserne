import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
from matplotlib.animation import PillowWriter
import numpy as np
# Bane og bil dimensioner (mm)
track_width = 750
car_width = 375
car_length = 645
safe_zone = 100
centrum = 1999.5
# Figur
fig, ax = plt.subplots(figsize=(8, 6))

metadata = dict(title="track", artist="Din mor")
writer = PillowWriter(fps=15, metadata=metadata)


ax.set_xlim(-3000, 8000)
ax.set_ylim(0, 5000)
ax.set_aspect('equal')

# Tegn banen (to linjer)

#MIDTERLINJER
ax.plot([0, 5000], [500, 500], 'k--', linewidth=3)  # midterlinje
ax.plot([0, 5000], [3125 + track_width/2 ,3125 + track_width/2], 'k--', linewidth=3)  # midterlinje
halfMiddle1 = Arc((5000, centrum), 1500*2, 1500*2, angle=0,theta1=-90, theta2=-270, linewidth=3,edgecolor='k', linestyle='--')  # black solid line
halfMiddle2 = Arc((0, centrum), 1500*2, 1500*2, angle=0,theta1=90, theta2=270, linewidth=3,edgecolor='k', linestyle='--')
ax.add_patch(halfMiddle1)
ax.add_patch(halfMiddle2)


ax.plot([0, 5000], [500 + track_width/2, 500 + track_width/2], color='k', linestyle='-', linewidth=2)
ax.plot([0, 5000], [500 - track_width/2, 500 - track_width/2], color='k', linestyle='-', linewidth=2)


# Tegn banen (to linjer)


ax.plot([0, 5000], [3125, 3125], color='k', linestyle='--', linewidth=2)
ax.plot([0, 5000], [3125 + track_width, 3125 + track_width], color='k', linestyle='--', linewidth=2)



Safezone1 = Rectangle((0, 500 - track_width/2), 5000, 100, edgecolor='red', facecolor='red')
ax.add_patch(Safezone1)
Safezone2 = Rectangle((0, 500 + track_width/2), 5000, -100, edgecolor='red', facecolor='red')
ax.add_patch(Safezone2)
Safezone3 = Rectangle((0, 3500 - track_width/2 ), 5000, 100, edgecolor='red', facecolor='red')
ax.add_patch(Safezone3)
Safezone3 = Rectangle((0, 3500 + track_width/2 ), 5000, -100, edgecolor='red', facecolor='red')
ax.add_patch(Safezone3)

# Safezone Line 2
ax.plot([0, 5000], [3125, 3125], 'black', linewidth=2)
ax.plot([0, 5000], [3125 + track_width, 3125 + track_width], 'black', linewidth=2)


halfOuter1 = Arc((5000, centrum), 1875*2, 1875*2, angle=0, theta1=-90, theta2=-270, linewidth=2)
halfOuter2 = Arc((0, centrum), 1875*2, 1875*2, angle=0, theta1=90, theta2=270, linewidth=2)
halfInner1 = Arc((5000, centrum), 1125*2, 1125*2, angle=0, theta1=-90, theta2=-270, linewidth=2)
halfInner2 = Arc((0, centrum), 1125*2, 1125*2, angle=0, theta1=90, theta2=270, linewidth=2)

ax.add_patch(halfInner1)
ax.add_patch(halfInner2)
ax.add_patch(halfOuter1)
ax.add_patch(halfOuter2)

halfOuterSafe1 = Arc((5000, centrum), 1820*2, 1820*2, angle=0, theta1=-90, theta2=-270, linewidth=8, edgecolor='red', facecolor='red')
halfOuterSafe2 = Arc((0, centrum), 1820*2, 1820*2, angle=0, theta1=90, theta2=270, linewidth=8, edgecolor='red', facecolor='red')
halfInnerSafe1 = Arc((5000, centrum), 1180*2, 1180*2, angle=0, theta1=-90, theta2=-270, linewidth=8, edgecolor='red', facecolor='red')
halfInnerSafe2 = Arc((0, centrum), 1180*2, 1180*2, angle=0, theta1=90, theta2=270, linewidth=8, edgecolor='red', facecolor='red')


ax.add_patch(halfInnerSafe1)
ax.add_patch(halfInnerSafe2)
ax.add_patch(halfOuterSafe1)
ax.add_patch(halfOuterSafe2)



xstraight1 = []
ystraight1 = []
for x in np.linspace(0,5000,100):
    xstraight1.append(x)
    ystraight1.append(500)
xarc1 = []
yarc1 = []
for t in np.linspace(-np.pi/2, np.pi/2, 100):
    x = 5000 + np.cos(t) * 1500
    y = 1999.5 + np.sin(t) * 1500
    xarc1.append(x)
    yarc1.append(y)

xstraight2 = []
ystraight2 = []
for x in np.linspace(5000,0,100):
    xstraight2.append(x)
    ystraight2.append(3500)
xarc2 = []
yarc2 = []
for t in np.linspace(np.pi/2, 3*np.pi/2,  100):
    x = 0 + np.cos(t) * 1500
    y = 1999.5 + np.sin(t) * 1500
    xarc2.append(x)
    yarc2.append(y)
xtraight1 = np.asarray(xstraight1)
xtraight2 = np.asarray(xstraight2)
xarc1 = np.asarray(xarc1)
xarc2 = np.asarray(xarc2)

ytraight1 = np.asarray(ystraight1)
ytraight2 = np.asarray(ystraight2)
yarc1 = np.asarray(yarc1)
yarc2 = np.asarray(yarc2)


trackx = np.concatenate([xstraight1, xarc1, xstraight2, xarc2])
tracky = np.concatenate([ystraight1, yarc1, ystraight2, yarc2])

dx = np.gradient(trackx)
dy = np.gradient(tracky)
angles = np.degrees(np.arctan2(dy, dx))

# Tegn bilen
car_x = 400  # startposition
car_y = 500
car = Rectangle((car_x - car_length/2, car_y - car_width/2), car_length, car_width, edgecolor='blue', facecolor='lightblue')

ax.add_patch(car)

# Tekst
# ax.text(920, 250, f"Track width: {track_width} mm", fontsize=10)
# ax.text(920, 230, f"Car width: {car_width} mm", fontsize=10)
# ax.text(920, 210, f"Margin: ±{(track_width - car_width)/2 - safe_zone:.1f} mm", fontsize=10)

from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D

total_frames = len(trackx)  # et frame per punkt

def update(frame):
    x = trackx[frame % total_frames]
    y = tracky[frame % total_frames]
    angle = angles[frame % total_frames]

    # Flyt og roter bilen omkring dens center
    trans = Affine2D().rotate_deg_around(x, y, angle) + ax.transData
    car.set_xy((x - car_length/2, y - car_width/2))
    car.set_transform(trans)
    return car,

ani = FuncAnimation(fig, update, frames=total_frames, interval=5, blit=True, repeat=True)
plt.show()
