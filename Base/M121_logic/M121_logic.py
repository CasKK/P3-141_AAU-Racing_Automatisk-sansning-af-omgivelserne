import numpy as np
from scipy.interpolate import splprep, splev
import time
import copy
# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyArrowPatch
# import plotly.express as px
import csv


def closest(vectorList):########### Sort point list based on distance from (0,0) (Simple) ############
    vectorList = copy.deepcopy(vectorList)
    for i, vector in enumerate(vectorList): #
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
    centers = []
    
    lenA = len(distanceListA)
    lenB = len(distanceListB)
    
    for i, (vecA, vecB) in enumerate(zip(distanceListA, distanceListB)): # determine midpoints between the two lists
        centers.append([(vecA[0] + vecB[0]) / 2, (vecA[1] + vecB[1]) / 2])
        if i < lenA - 1 and i < lenB - 1:
            next_vecA = distanceListA[i + 1]
            next_vecB = distanceListB[i + 1]
            centers.append([((next_vecA[0] + vecB[0]) / 2 + ((next_vecB[0] + vecA[0]) / 2)) / 2, ((next_vecA[1] + vecB[1]) / 2 + (next_vecB[1] + vecA[1]) / 2) / 2])
    
    if lenA != lenB and min(lenA, lenB) > 0: # Handle case where lists are of unequal length
        if lenA > lenB:
            last = distanceListB[-1]
            extra_points = distanceListA[lenB:]
        else:
            last = distanceListA[-1]
            extra_points = distanceListB[lenA:]
        for p in extra_points:
            x = (last[0] + p[0]) / 2
            y = (last[1] + p[1]) / 2
            centers.append([x, y])
    
    lenC = len(centers)
    for i in range(lenC): # Interpolate additional points if less than 6 centers found
        if i < lenC - 1:
            next_cen = centers[i + 1]
            x = (next_cen[0] + centers[i][0]) / 2
            y = (next_cen[1] + centers[i][1]) / 2
            centers.append([x, y])

    for i, center in enumerate(centers): # Remove duplicates based on distance from car
        centers[i].append(np.sqrt(center[0]**2 + center[1]**2))
    centers = np.array(sorted(centers, key=lambda x: x[-1]))
    _, idx = np.unique(centers, axis=0, return_index=True)
    centers = centers[np.sort(idx)]

    if len(centers) < 6:
        centers = np.array([car.copy(), [0, 1600], [0, 1700], [0, 1800], [0, 1900], [0, 2000]])
    
    return centers

def BSpline(points, smoothing, k):########### Make and fit Basis-spline ############### 
    d = 0
    t = [0]
    for i, point in enumerate(points):
        if i < len(points)-1:
            nextPoint = points[i+1]
            d += (np.sqrt((point[0] - nextPoint[0])**2 + (point[1] - nextPoint[1])**2)) 
            t = np.append(t, d)
    t = t / t[-1]  # normalize between 0--1
    
    tck, u = splprep([points[:,0], points[:,1]], u=t, s=smoothing, k=k) # Make B-Spline with smoothing.
    u_fine = np.linspace(0, 1, 800)
    x_u, y_u = splev(u_fine, tck)
    dx_u, dy_u = splev(u_fine, tck, der=1)
    ddx_u, ddy_u = splev(u_fine, tck, der=2)
    
    s = np.sqrt(dx_u**2 + dy_u**2)              # Calculus: A Complete Course, 10Ce
    kp = (dx_u * ddy_u - dy_u * ddx_u) / (s**3) # chapter 12.5: Curvature and Torsion for General Parametrizations.
    v_max = np.sqrt(1 / (np.abs(kp) + 1e-6))    # Speed
    v_max = np.clip(v_max, 0, 80)
    return v_max, x_u, y_u

def targetPoint(listA, listB): # Find the target point at a set distance ahead along the spline
    newList = []
    for i, (x, y) in enumerate(zip(listA, listB)):
        nextDist = np.sqrt((x - car[0])**2 + (y - car[1])**2)
        newList.append([nextDist, i])
    newList = np.array(sorted(newList, key=lambda x: x[0]))
    
    u0 = int(newList[0, 1])
    s = 0
    uL = u0

    for i in range(u0, len(listA) - 1):
        ds = np.sqrt((listA[i+1] - listA[i])**2 + (listB[i+1] - listB[i])**2)
        s += ds
        if s >= tDistance:
            uL = i
            break
    xL = listA[uL] - car[0]
    yL = listB[uL] - car[1]

    return xL, yL

def steeringAngle(x, y): # Calculate steering angle based on target point
    alpha = np.arctan2(x, y)
    Ld = np.hypot(x, y)
    delta = np.arctan2(2 * L * np.sin(alpha), Ld)
    return delta


############## Pre program stuff #############

inputVectorsB = []
inputVectorsY = []
car = [0, 1500] # Car position is constant in this module. Used to ajust spline to get correct 'current' steering angle.
L = 480
tDistance = 800  # mm

############## Program ##################

def main(): # Main processing function
    global distanceSortedPointsB, distanceSortedPointsY
    distanceSortedPointsB = closest(inputVectorsB) # Sort blue points by distance from 0,0
    distanceSortedPointsY = closest(inputVectorsY) # Sort yellow points by distance from 0,0

    global centers
    centers = calculateCenters(distanceSortedPointsB, distanceSortedPointsY) # Calculate center points between blue and yellow points

    global s, x_smooth, y_smooth
    s, x_smooth, y_smooth = BSpline(centers, 20000, 5) # Fit B-spline to center points
    global steer_now, targetX, targetY
    targetX, targetY = targetPoint(x_smooth, y_smooth) # Find target point along the spline
    steer_now = steeringAngle(targetX, targetY) # Calculate steering angle to target point

def run(input_queue, serial_queue): # Run function for m121
    time.sleep(0.5)
    global inputVectorsB, inputVectorsY
    main() # Initial run to setup spline
    # Visualization setup
    # plt.ion()
    # fig, ax = plt.subplots()
    # fig.set_size_inches(8, 8)
    # plt.xlim(-3000, 5000)
    # plt.ylim(-500, 8000)
    # if len(inputVectorsB) == 0 or len(inputVectorsY) == 0:
    #     bx = [-100, 100]
    #     by = [100, 100]
    #     yx = [-100, 100]
    #     yy = [200, 200]
    # else: 
    #     bx = [p[0] for p in inputVectorsB]
    #     by = [p[1] for p in inputVectorsB]
    #     yx = [p[0] for p in inputVectorsY]
    #     yy = [p[1] for p in inputVectorsY]
    # cx = [p[0] for p in centers]
    # cy = [p[1] for p in centers]
    
    # yscatter = ax.scatter(yx, yy, c='yellow', edgecolors='black')
    # bscatter = ax.scatter(bx, by, c='blue', edgecolors='black')
    # cscatter = ax.scatter(bx, by, c='red', edgecolors='black')
    # line, = ax.plot(x_smooth, y_smooth, label="Smoothed B-spline fit", linewidth=2)

    # start = (car[0], car[1])
    # end = (targetX + car[0], targetX + car[1])
    # arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15, color='green')
    # ax.add_patch(arrow)
    
    # ax.set_aspect('equal')
    
    # plt.show(block=False)
    # plt.pause(0.01)
    with open("m121log", "a", newline="") as f: # Open the m121 log file for appending
        writer = csv.writer(f)
        while True:
            
            inputVectorsB, inputVectorsY = input_queue.get() # Get new points from m113
            main() # Run main processing function

            steer_deg = np.degrees(-steer_now) # Convert steering angle to degrees and invert
            steer_deg = np.clip(steer_deg, -90, 90) # Clip steering angle to valid range
            if serial_queue.qsize() >= 5: # limit queue size to prevent serial manager falling behind
                try:
                    serial_queue.get_nowait()  # remove oldest item
                except:
                    pass
            steer_deg = steer_deg + 90
            serial_queue.put(int(steer_deg)) # Send steering command to serial manager
            print(f"steer_deg: {steer_deg}")

            #Log
            timestamp = time.time()
            writer.writerow([timestamp, steer_deg])
            f.flush()
            
            # if len(inputVectorsB) == 0 or len(inputVectorsY) == 0:
            #     bx = [-100, 100]
            #     by = [100, 100]
            #     yx = [-100, 100]
            #     yy = [200, 200]
            # else: 
            #     bx = [p[0] for p in inputVectorsB]
            #     by = [p[1] for p in inputVectorsB]
            #     yx = [p[0] for p in inputVectorsY]
            #     yy = [p[1] for p in inputVectorsY]
            # cx = [p[0] for p in centers]
            # cy = [p[1] for p in centers]

            # start = (car[0], car[1])
            # end = (targetX + car[0], targetX + car[1])
            # arrow.set_positions(start, end)

            # yscatter.set_offsets(list(zip(yx, yy)))
            # bscatter.set_offsets(list(zip(bx, by)))
            # cscatter.set_offsets(list(zip(cx, cy)))
            # line.set_data(x_smooth, y_smooth)
            # plt.pause(0.001)
