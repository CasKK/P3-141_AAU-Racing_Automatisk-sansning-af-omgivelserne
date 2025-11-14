import numpy as np
import random
import matplotlib.pyplot as plt

def generate_points(n_points=20, 
                    x_max=125, x_min_dist=75,
                    y_max=1250, y_min_step=250):
    pointsA = []
    pointsB = []
    
    # Start y somewhere within range
    y = random.uniform(0, y_max - y_min_step * (n_points - 1))
    
    for i in range(n_points):
        # Ensure we don't exceed max Y
        if y > y_max:
            break
        
        # Random X for list A
        xA = random.uniform(-x_max, x_max)
        
        # For list B, ensure it's at least x_min_dist away
        # Randomly decide if it’s on the opposite side or same side
        if random.random() < 0.5:
            # opposite side of x-axis
            if xA >= 0:
                xB = xA - random.uniform(x_min_dist, 2 * x_max - xA)
            else:
                xB = xA + random.uniform(x_min_dist, 2 * x_max + xA)
        else:
            # same side, just offset by at least x_min_dist
            offset = random.choice([-1, 1]) * random.uniform(x_min_dist, x_max)
            xB = xA + offset
        
        # Clamp xB to range
        xB = np.clip(xB, -x_max, x_max)
        
        # Append both
        pointsA.append([xA, y])
        pointsB.append([xB, y])
        
        # Increase y by random step
        y += random.uniform(y_min_step, y_min_step * 1.5)
    
    return np.array(pointsA), np.array(pointsB)


# ---- Generate ----
distanceListA, distanceListB = generate_points()

# ---- Plot ----
plt.figure(figsize=(8, 6))
plt.scatter(distanceListA[:,0], distanceListA[:,1], color='blue', label='List A')
plt.scatter(distanceListB[:,0], distanceListB[:,1], color='yellow', edgecolor='black', label='List B')

# Connect corresponding points
for a, b in zip(distanceListA, distanceListB):
    plt.plot([a[0], b[0]], [a[1], b[1]], 'k--', alpha=0.4)

plt.title("Procedurally Generated Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

inputVectorsB = [[-368.0, 540.0, 0],
                [-381.0, 2580.0, 0],
                [-363.0, 2250.0, 0],
                [-344.0, 3540.0, 0],
                [-365.0, 1100.0, 0],
                [-353.0, 4690.0, 0]]

inputVectorsY = [[353.0, 670.0, 1],
                [353.0, 3510.0, 1],
                [351.0, 2150.0, 1],
                [379.0, 2630.0, 1],
                [377.0, 1330.0, 1],
                [377.0, 4490.0, 1]]
