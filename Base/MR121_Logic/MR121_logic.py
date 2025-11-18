
import numpy as np
import pandas as pd
import time
import copy
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev


inputVectorsB = [[-350.13, 50.90, 0],
                [-350.34, 100.90, 0],
                [-350.56, 200.90, 0],
                [-350.13, 250.90, 0],
                [-350.34, 300.90, 0],
                [-350.56, 420.90, 0]]
inputVectorsY = [[350.13, 40.90, 1],
                [350.34, 100.90, 1],
                [350.56, 200.90, 1],
                [350.13, 250.90, 1],
                [350.34, 300.90, 1],
                [350.56, 400.90, 1]]

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

inputVectorsBTurn = [[-368.0, 540.0, 0],
                    [-365.0, 1100.0, 0],
                    [-403.0, 2250.0, 0],
                    [-1.0, 3680.0, 0],
                    [1044.0, 4540.0, 0],
                    [1847.0, 4690.0, 0]]
inputVectorsYTurn = [[353.0, 670.0, 1],
                    [377.0, 1330.0, 1],
                    [351.0, 2150.0, 1],
                    [779.0, 3230.0, 1],
                    [1353.0, 3710.0, 1],
                    [1977.0, 3890.0, 1]]

def closestNP(vectorListA, vectorListB):########### Sort two point lists based on distance from car(0,0) ############
    # time_start = time.time()
    vectorListA = copy.deepcopy(vectorListA)
    vectorListB = copy.deepcopy(vectorListB)
    for i, vector in enumerate(vectorListA):
        result = np.sqrt(vector[0]**2 + vector[1]**2)
        if i < len(vectorListA)-1:
            nextVector = vectorListA[i+1]
            nextDist = np.sqrt((vector[0] - nextVector[0])**2 + (vector[1] - nextVector[1])**2)
        else:
            nextDist = 1000
        vectorListA[i].extend([result, nextDist])
        
    for i, vector in enumerate(vectorListB):
        result = np.sqrt(vector[0]**2 + vector[1]**2)
        if i < len(vectorListB)-1:
            nextVector = vectorListB[i+1]
            nextDist = np.sqrt((vector[0] - nextVector[0])**2 + (vector[1] - nextVector[1])**2)
        else:
            nextDist = 1000
        vectorListB[i].extend([result, nextDist])
    vectorListA = np.array(sorted(vectorListA, key=lambda x: x[-2]))
    vectorListB = np.array(sorted(vectorListB, key=lambda x: x[-2]))

    # time_end = time.time()
    # print(f"Runtime: {time_end - time_start:.4f} seconds")
    # print(vectorListA)
    # print(vectorListB)
    return vectorListA, vectorListB

def closestPandasQuick(localVectorListA, localVectorListB):
    time_start = time.time()
    localVectorListA = copy.deepcopy(localVectorListA)
    localVectorListB = copy.deepcopy(localVectorListB)

    for i, vector in enumerate(localVectorListA):
        result = np.sqrt(vector[0]**2 + vector[1]**2)
        if i < len(localVectorListA)-1:
            nextVector = localVectorListA[i+1]
            nextDist = np.sqrt((vector[0] - nextVector[0])**2 + (vector[1] - nextVector[1])**2)
            localVectorListA[i] += [result]
            localVectorListA[i] += [nextDist]
    for i, vector in enumerate(localVectorListB):
        result = np.sqrt(vector[0]**2 + vector[1]**2)
        if i < len(localVectorListB)-1:
            nextVector = localVectorListB[i+1]
            nextDist = np.sqrt((vector[0] - nextVector[0])**2 + (vector[1] - nextVector[1])**2)
            localVectorListB[i].extend([result, nextDist])
    df1 = pd.DataFrame(localVectorListA, columns=['x', 'y', 'color', 'distanceCar', 'distanceNext'])
    df2 = pd.DataFrame(localVectorListB, columns=['x', 'y', 'color', 'distanceCar', 'distanceNext'])

    time_end = time.time()
    print(f"Runtime: {time_end - time_start:.4f} seconds")
    # print(df1)
    # print(df2)
    return df1, df2

def closestPandasSimple(localVectorListA, localVectorListB):
    time_start = time.time()

    df1 = pd.DataFrame(localVectorListA, columns=['x', 'y', 'color'])
    df2 = pd.DataFrame(localVectorListB, columns=['x', 'y', 'color'])
    df1['distanceCar'] = np.sqrt(df1['x']**2 + df1['y']**2)
    df2['distanceCar'] = np.sqrt(df2['x']**2 + df2['y']**2)
    df1['dist_to_next'] = np.sqrt((df1['x'].shift(-1) - df1['x'])**2 + (df1['y'].shift(-1) - df1['y'])**2)
    df2['dist_to_next'] = np.sqrt((df2['x'].shift(-1) - df2['x'])**2 + (df2['y'].shift(-1) - df2['y'])**2)

    time_end = time.time()
    print(f"Runtime: {time_end - time_start:.4f} seconds")
    print(df1)
    print(df2)
    return df1, df2


def calculateCenters(distanceListA, distanceListB):############# Canculate center points from two point lists ############
    centers = []
    centers.append([-300,0])
    for i, (vecA, vecB) in enumerate(zip(distanceListA, distanceListB)):
        centers.append([((vecA[0] - vecB[0]) / 2) + vecB[0], ((vecA[1] - vecB[1]) / 2) + vecB[1]])
        if i < len(distanceListA) - 1:
            next_vecA = distanceListA[i + 1]
            centers.append([((next_vecA[0] - vecB[0]) / 2) + vecB[0], ((next_vecA[1] - vecB[1]) / 2) + vecB[1]])
        if i < len(distanceListB) - 1:
            next_vecB = distanceListB[i + 1]
            centers.append([((next_vecB[0] - vecA[0]) / 2) + vecA[0], ((next_vecB[1] - vecA[1]) / 2) + vecA[1]])
    for i, center in enumerate(centers):
        centers[i].append(np.sqrt(center[0]**2 + center[1]**2))
    centers = np.array(sorted(centers, key=lambda x: x[-2]))
    # print(centers)
    return centers


def BSpline(points):########### Make and fit B-spline ############### 
    x = points[:,0]
    y = points[:,1]

    # Initial parameter based on chord length (helps avoid oscillation)
    d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    t = np.concatenate([[0], np.cumsum(d)])
    t = t / t[-1]  # normalize to 0..1

    # ★ KEY PART: smoothing factor s > 0 (tune this!)
    # s=0 → exact through points (bad for racing line)
    # s=50–200 → smooth but still follows general shape
    tck, u = splprep([x, y], u=t, s=50000, k=5)

    # Generate smoothed line
    u_fine = np.linspace(0, 1, 800)
    x_smooth, y_smooth = splev(u_fine, tck)

    dx, dy = splev(u_fine, tck, der=1)
    ddx, ddy = splev(u_fine, tck, der=2)
    return np.array([dx, dy]), np.array([ddx, ddy]), x_smooth, y_smooth


############## Pre program stuff #############

temp = 50
time_start = time.time() # start time for measuring perfomance


############## Program ##################

distanceListB, distanceListY = closestNP(inputVectorsBTurn, inputVectorsYTurn)
# distanceLista, distanceListd = closestPandasQuick(inputVectorsB, inputVectorsY)
# distanceLista, distanceListd = closestPandasSimple(inputVectorsB, inputVectorsY)

centers = calculateCenters(distanceListB, distanceListY)

direction, direction1, x_smooth, y_smooth = BSpline(centers)


################# Post program stuff #####################

time_end = time.time() # stop time
print(f"Runtime: {time_end - time_start:.5f} seconds")

print(direction[:, temp])
print(direction1[:, temp])


################# plot ##############

plt.figure(figsize=(8,8))
plt.scatter(distanceListY[:, 0], distanceListY[:, 1], c='blue', label='distanceListA')
plt.scatter(distanceListB[:, 0], distanceListB[:, 1], c='yellow', edgecolor='black', label='distanceListB')
plt.scatter(centers[:,0], centers[:,1], color='red', label="Original waypoints")
plt.plot(x_smooth, y_smooth, label="Smoothed B-spline fit", linewidth=2)
plt.quiver(x_smooth[temp], y_smooth[temp], direction[0, temp] / 10, direction[1, temp] / 10, angles='xy', scale_units='xy', scale=1, color='green', label='Vector')
plt.quiver(x_smooth[temp], y_smooth[temp], direction1[0, temp] / 10, direction1[1, temp] / 10, angles='xy', scale_units='xy', scale=1, color='green', label='Vector')
plt.axis('equal')
#plt.legend()
#plt.title("Smoothed racing line fit (B-spline with smoothing)")
plt.show()


############################ Polyfit ###################################

# from scipy.interpolate import lagrange

# degree = 2
# coeffs = np.polyfit(centers[:, 0], centers[:, 1], degree)
# p = np.poly1d(coeffs)
# # p = lagrange(centers[:, 0], centers[:, 1])

# # Make a smooth range of x-values for plotting the curve
# x_plot = np.linspace(min(centers[:, 0]), max(centers[:, 0]), 200)
# y_plot = p(x_plot)

# # Plot
# plt.scatter(centers[:, 0], centers[:, 1], label='Data Points')   # original points
# plt.plot(x_plot, y_plot, label='Polynomial Fit')

# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Polynomial Fit")
# plt.grid(True)
# plt.show()


######################## 'pass-by-value' vs 'pass-by-reference' test #############################

# inputA =    [[-350.13, 50.90, "blue"],
#             [-350.34, 100.90, "blue"],
#             [-350.56, 200.90, "blue"],
#             [-350.13, 250.90, "blue"],
#             [-350.34, 300.90, "blue"],
#             [-350.56, 400.90, "blue"]]

# print(inputA)

# def someFunction(localInputA):
#     for i, vector in enumerate(localInputA):
#         result = vector[0] + 350
#         localInputA[i] += [result]
#     return localInputA

# someVar = someFunction(inputA)

# print(inputA)
# print(someVar)

# # result:
# # [[-350.13, 50.9, 'blue'], [-350.34, 100.9, 'blue'], [-350.56, 200.9, 'blue'], [-350.13, 250.9, 'blue'], [-350.34, 300.9, 'blue'], [-350.56, 400.9, 'blue']]
# # [[-350.13, 50.9, 'blue', -0.12999999999999545], [-350.34, 100.9, 'blue', -0.339999999999975], [-350.56, 200.9, 'blue', -0.5600000000000023], [-350.13, 250.9, 'blue', -0.12999999999999545], [-350.34, 300.9, 'blue', -0.339999999999975], [-350.56, 400.9, 'blue', -0.5600000000000023]]
# # [[-350.13, 50.9, 'blue', -0.12999999999999545], [-350.34, 100.9, 'blue', -0.339999999999975], [-350.56, 200.9, 'blue', -0.5600000000000023], [-350.13, 250.9, 'blue', -0.12999999999999545], [-350.34, 300.9, 'blue', -0.339999999999975], [-350.56, 400.9, 'blue', -0.5600000000000023]]


