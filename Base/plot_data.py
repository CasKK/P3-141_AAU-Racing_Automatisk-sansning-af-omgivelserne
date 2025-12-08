import csv
import json
import matplotlib.pyplot as plt

fil = "m113log"

frames = []

with open(fil) as f:
    reader = csv.reader(f)

    for row in reader:
        timestamp = float(row[0])
        blue = json.loads(row[1])
        yellow = json.loads(row[2])

        frames.append({
            "timestamp": timestamp,
            "blue": blue,
            "yellow": yellow
        })

plt.ion()
fig, ax = plt.subplots()
plt.xlim(-3000, 3000)
plt.ylim(-500, 6000)
ax.set_aspect("equal")
ax.grid(True)

yscatter = ax.scatter([], [], c='yellow', edgecolors='black')
bscatter = ax.scatter([], [], c='blue', edgecolors='black')

for frame in frames:
    blue = frame["blue"]
    yellow = frame["yellow"]

    if len(blue) == 0 or len(yellow) == 0:
        bx, by = [-100, 100], [100, 100]
        yx, yy = [-100, 100], [200, 200]
    else:
        bx = [p[0] for p in blue]
        by = [p[1] for p in blue]
        yx = [p[0] for p in yellow]
        yy = [p[1] for p in yellow]

    bscatter.set_offsets(list(zip(bx, by)))
    yscatter.set_offsets(list(zip(yx, yy)))

    plt.pause(0.02)

