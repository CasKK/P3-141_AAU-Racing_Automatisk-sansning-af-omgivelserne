import math

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

some1 = 1500

oldPointsB = [
    [-300, 5, 0, 0],
    [-300, 750, 0, 0],
    [-300, 1500, 0, 0],
    [-368.0, 540.0 + some1, 0, 0],
    [-365.0, 1100.0 + some1, 0, 0],
    [-403.0, 2250.0 + some1, 0, 0],
    [-1.0, 3680.0 + some1, 0, 0],
    [1044.0, 4540.0 + some1, 0, 0],
    [1847.0, 4690.0 + some1, 0, 0],
    [2477.0, 4800.0 + some1, 0, 0],
    [2900.0, 5200.0 + some1, 0, 0],
    [3000.0, 5600.0 + some1, 0, 0],
    [2910.0, 6300.0 + some1, 0, 0]
]

oldPointsY = [
    [300, 1, 1, 0],
    [300, 755, 1, 0],
    [300, 1505, 1, 0],
    [353.0, 670.0 + some1, 1, 0],
    [377.0, 1330.0 + some1, 1, 0],
    [351.0, 2150.0 + some1, 1, 0],
    [779.0, 3230.0 + some1, 1, 0],
    [1353.0, 3710.0 + some1, 1, 0],
    [1977.0, 3890.0 + some1, 1, 0],
    [3000.0, 4000.0 + some1, 1, 0],
    [3700.0, 4700.0 + some1, 1, 0],
    [3800.0, 5700.0 + some1, 1, 0],
    [3810.0, 6300.0 + some1, 1, 0]
]

N = min(len(oldPointsB), len(oldPointsY))

# Compute distances
dist_B_to_B_next = [dist(oldPointsB[i], oldPointsB[i+1]) for i in range(N - 1)]
dist_Y_to_Y_next = [dist(oldPointsY[i], oldPointsY[i+1]) for i in range(N - 1)]
dist_B_to_Y      = [dist(oldPointsB[i], oldPointsY[i])   for i in range(N)]
dist_B_to_Y_next = [dist(oldPointsB[i], oldPointsY[i+1]) for i in range(N - 1)]
dist_Y_to_B_next = [dist(oldPointsY[i], oldPointsB[i+1]) for i in range(N - 1)]

# Pretty printing
print("\n===== DISTANCES (Index-Aligned) =====\n")

print("B[i] → B[i+1]:")
for i, d in enumerate(dist_B_to_B_next):
    print(f"  B[{i}] → B[{i+1}] = {d:.2f}")
print()

print("Y[i] → Y[i+1]:")
for i, d in enumerate(dist_Y_to_Y_next):
    print(f"  Y[{i}] → Y[{i+1}] = {d:.2f}")
print()

print("B[i] → Y[i]:")
for i, d in enumerate(dist_B_to_Y):
    print(f"  B[{i}] → Y[{i}] = {d:.2f}")
print()

print("B[i] → Y[i+1]:")
for i, d in enumerate(dist_B_to_Y_next):
    print(f"  B[{i}] → Y[{i+1}] = {d:.2f}")
print()

print("Y[i] → B[i+1]:")
for i, d in enumerate(dist_Y_to_B_next):
    print(f"  Y[{i}] → B[{i+1}] = {d:.2f}")
print()
