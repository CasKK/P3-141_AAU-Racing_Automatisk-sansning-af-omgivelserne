import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.interpolate import splprep, splev

car = [0,1500]
blue = []
yellow = []
L = 480
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


def closestNP1(vectorList):########### Sort point list based on distance from (0,0) ############
    vectorList = copy.deepcopy(vectorList)
    for i, vector in enumerate(vectorList):
        result = np.sqrt(vector[0]**2 + vector[1]**2)
        if i < len(vectorList)-1:
            nextVector = vectorList[i+1]
            nextDist = np.sqrt((vector[0] - nextVector[0])**2 + (vector[1] - nextVector[1])**2)
        else:
            nextDist = 10000
        vectorList[i].extend([result, nextDist])
    vectorList = np.array(sorted(vectorList, key=lambda x: x[-2]))
    return vectorList


def calculateCenters(distanceListA, distanceListB):############# Canculate center points from two point lists ############
    centers = [car.copy()] #, car1.copy(), car2.copy()]#, [0, 1600], [0, 1700]]#, [0,50], [0,100], [0,150], [0,200]] #, [0,250], [0,350], [0,300], [0,25], [0,75], [0,125], [0,175], [0,225], [0,275], [0,325], [0,375]
    # distanceListA = copy.deepcopy(distanceListA)
    # distanceListB = copy.deepcopy(distanceListB)
    # print(car)
    # print(f"if {distanceListA[0][3]} > {distanceListB[0][3]}")
    # if distanceListA[0][3] > distanceListB[0][3] + 200:
    #     print("someFix")

    for i, (vecA, vecB) in enumerate(zip(distanceListA, distanceListB)):
        centers.append([(vecA[0] + vecB[0]) / 2, (vecA[1] + vecB[1]) / 2])
        if i < len(distanceListA) - 1 and i < len(distanceListB) - 1:
            next_vecA = distanceListA[i + 1]
            next_vecB = distanceListB[i + 1]
            centers.append([((next_vecA[0] + vecB[0]) / 2 + ((next_vecB[0] + vecA[0]) / 2)) / 2, ((next_vecA[1] + vecB[1]) / 2 + (next_vecB[1] + vecA[1]) / 2) / 2])
    for i, center in enumerate(centers):
        centers[i].append(np.sqrt(center[0]**2 + center[1]**2))
    centers = np.array(sorted(centers, key=lambda x: x[-2]))
    _, idx = np.unique(centers, axis=0, return_index=True)
    centers = centers[np.sort(idx)]
    if len(centers) < 6:
        centers = np.array([car.copy(), [0, 1600], [0, 1700], [0, 1800], [0, 1900], [0, 2000]])
    
    # print(centers)
    return centers


def BSpline(points):########### Make and fit Basis-spline ############### 
    d = 0
    t = [0]
    for i, point in enumerate(points):
        if i < len(points)-1:
            nextPoint = points[i+1]
            d += (np.sqrt((point[0] - nextPoint[0])**2 + (point[1] - nextPoint[1])**2)) 
            t = np.append(t, d)
    t = t / t[-1]  # normalize between 0--1
    
    tck, u = splprep([points[:,0], points[:,1]], u=t, s=50000, k=5)
    u_fine = np.linspace(0, 1, 800)
    x_u, y_u = splev(u_fine, tck)
    dx_u, dy_u = splev(u_fine, tck, der=1)
    ddx_u, ddy_u = splev(u_fine, tck, der=2)
    
    s = np.sqrt(dx_u**2 + dy_u**2)              # Formel fra "Calculus: A Complete Course, 10Ce" Omskrevet til 2D fra 3D
    kp = (dx_u * ddy_u - dy_u * ddx_u) / (s**3) # chapter 12.5: Curvature and Torsion for General Parametrizations side 676 (første side af kapitlet).
    v_max = np.sqrt(1 / (np.abs(kp) + 1e-6))    # <--- Hastigheds-formel fra chat
    v_max = np.clip(v_max, 0, 80)
    #v_target = np.convolve(v_max, np.ones(50)/10, mode='same')
    return v_max, kp, x_u, y_u, dx_u, dy_u #np.array([dx_u, dy_u]), np.array([ddx_u, ddy_u])

def carClosestPoint(listA, listB): # Find the list index of the closest point to the car  
    newList = []
    for i, (x, y) in enumerate(zip(listA, listB)):
        nextDist = np.sqrt((x - car[0])**2 + (y - car[1])**2)
        newList.append([nextDist, i])
    newList = np.array(sorted(newList, key=lambda x: x[0]))
    return newList[0, 1]

def main():
    # inputVectorsBTurn = blueConesFromM113()
    # inputVectorsYTurn = yellowConesFromM113()
    global distanceSortedPointsB, distanceSortedPointsY
    distanceSortedPointsB = closestNP1(blue)
    distanceSortedPointsY = closestNP1(yellow)

    # for point in distanceSortedPointsY: # To offset the whole track in x direction for testing purposes
    #     point[0] += 300
    # for point in distanceSortedPointsY:
    #     point[0] += 300

    global centers
    centers = calculateCenters(distanceSortedPointsB, distanceSortedPointsY)

    global s, kp, x_smooth, y_smooth, dx_u, dy_u
    s, kp, x_smooth, y_smooth, dx_u, dy_u = BSpline(centers)
    global closest_u
    closest_u = carClosestPoint(x_smooth, y_smooth) # Find the index 
    global steering, steer_now
    steering = np.arctan(L * kp) # 3-line steering-angle bicycle formula
    steer_now = steering[int(closest_u)]







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
    main()

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

    plt.pause(0.05)

