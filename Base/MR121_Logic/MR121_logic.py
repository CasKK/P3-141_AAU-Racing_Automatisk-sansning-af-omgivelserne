
import numpy as np
import pandas as pd
import time
import copy
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.interpolate import splprep, splev, make_lsq_spline


# inputVectorsB = [[-350.13, 50.90, 0],
#                 [-350.34, 100.90, 0],
#                 [-350.56, 200.90, 0],
#                 [-350.13, 250.90, 0],
#                 [-350.34, 300.90, 0],
#                 [-350.56, 420.90, 0]]
# inputVectorsY = [[350.13, 40.90, 1],
#                 [350.34, 100.90, 1],
#                 [350.56, 200.90, 1],
#                 [350.13, 250.90, 1],
#                 [350.34, 300.90, 1],
#                 [350.56, 400.90, 1]]

# inputVectorsB = [[-368.0, 540.0, 0],
#                 [-381.0, 2580.0, 0],
#                 [-363.0, 2250.0, 0],
#                 [-344.0, 3540.0, 0],
#                 [-365.0, 1100.0, 0],
#                 [-353.0, 4690.0, 0]]
# inputVectorsY = [[353.0, 670.0, 1],
#                 [353.0, 3510.0, 1],
#                 [351.0, 2150.0, 1],
#                 [379.0, 2630.0, 1],
#                 [377.0, 1330.0, 1],
#                 [377.0, 4490.0, 1]]

##### Original turn #######
inputVectorsYTurn = [[-368.0, 540.0, 0],
                    [-365.0, 1100.0, 0],
                    [-403.0, 2250.0, 0],
                    [-1.0, 3680.0, 0],
                    [1044.0, 4540.0, 0],
                    [1847.0, 4690.0, 0]]
inputVectorsBTurn = [[353.0, 670.0, 1],
                    [377.0, 1330.0, 1],
                    [351.0, 2150.0, 1],
                    [779.0, 3230.0, 1],
                    [1353.0, 3710.0, 1],
                    [1977.0, 3890.0, 1]]

####### Expanded turn #########
inputVectorsYTurn = [[-368.0, 540.0, 0],
                    [-365.0, 1100.0, 0],
                    [-403.0, 2250.0, 0],
                    [-1.0, 3680.0, 0],
                    [1044.0, 4540.0, 0],
                    [1847.0, 4690.0, 0],
                    [2477.0, 4800.0, 0],
                    [2900.0, 5200.0, 0],
                    [3000.0, 5600.0, 0],
                    [2910.0, 6300.0, 0]]
inputVectorsBTurn = [[353.0, 670.0, 1],
                    [377.0, 1330.0, 1],
                    [351.0, 2150.0, 1],
                    [779.0, 3230.0, 1],
                    [1353.0, 3710.0, 1],
                    [1977.0, 3890.0, 1],
                    [3000.0, 4000.0, 1],
                    [3700.0, 4700.0, 1],
                    [3800.0, 5700.0, 1],
                    [3810.0, 6300.0, 1]]

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
    centers = [[0,0], [0,300], [0,100], [0,150], [0,200]] #, [0,250], [0,350], [0,300], [0,25], [0,75], [0,125], [0,175], [0,225], [0,275], [0,325], [0,375]
    #centers.append([0,0], [0,100])
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


def BSpline1232():
    # x = points[:,0]
    # y = points[:,1]

    # d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    # t = np.concatenate([[0], np.cumsum(d)])

    print("something")

    # # cubic
    # k = 3
    # # smoothing: fewer knots → more smooth
    # num_knots = 12
    # knots = np.linspace(0, 1, num_knots)[1:-1]
    # spl_x = make_lsq_spline(t, points[:,0], knots, k)
    # spl_y = make_lsq_spline(t, points[:,1], knots, k)
    # t_fine = np.linspace(0, 1, 800)
    # x_smooth = spl_x(t_fine)
    # y_smooth = spl_y(t_fine)

def BSpline(points):########### Make and fit Basis-spline ############### 
    d = 0
    t = [0]
    for i, point in enumerate(points):
        if i < len(points)-1:
            nextPoint = points[i+1]
            d += (np.sqrt((point[0] - nextPoint[0])**2 + (point[1] - nextPoint[1])**2)) 
            t = np.append(t, d)
    t = t / t[-1]  # normalize to 0..1
    
    tck, u = splprep([points[:,0], points[:,1]], u=t, s=50000, k=5)
    u_fine = np.linspace(0, 1, 800)
    x_u, y_u = splev(u_fine, tck)
    dx_u, dy_u = splev(u_fine, tck, der=1)
    ddx_u, ddy_u = splev(u_fine, tck, der=2)
    
    s = np.sqrt(dx_u**2 + dy_u**2)              # Formel fra "Calculus: A Complete Course, 10Ce" Omskrevet til 2D fra 3D
    kp = (dx_u * ddy_u - dy_u * ddx_u) / (s**3) # chapter 12.5: Curvature and Torsion for General Parametrizations side 676 (første side af kapitlet).
    v_max = np.sqrt(1 / (np.abs(kp) + 1e-6))    # <--- Formel fra chat
    v_max = np.clip(v_max, 0, 80)
    #v_target = np.convolve(v_max, np.ones(50)/10, mode='same')
    return v_max, kp, x_u, y_u, dx_u, dy_u #np.array([dx_u, dy_u]), np.array([ddx_u, ddy_u])


############## Pre program stuff #############

temp = 500
time_start = time.time() # start time for measuring perfomance


############## Program ##################

distanceListB, distanceListY = closestNP(inputVectorsYTurn, inputVectorsBTurn)
# distanceLista, distanceListd = closestPandasQuick(inputVectorsB, inputVectorsY)
# distanceLista, distanceListd = closestPandasSimple(inputVectorsB, inputVectorsY)

for distance in distanceListB:
    distance[0] += 300
for distance in distanceListY:
    distance[0] += 300

centers = calculateCenters(distanceListB, distanceListY)

s, kp, x_smooth, y_smooth, dx_u, dy_u = BSpline(centers)

# plt.plot(kp)
# plt.show()


################# Post program stuff #####################

time_end = time.time() # stop time
print(f"Runtime: {time_end - time_start:.5f} seconds")

print(s[temp])
print(kp[temp])
L = 1000
steering = np.arctan(L * kp)

steer_first = steering[0]
print("steering (rad):", steer_first)
print("steering (deg):", np.degrees(steer_first))


################# plot ##############


df = pd.DataFrame({
    'x': x_smooth,
    'y': y_smooth,
    'z': s
})


fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        opacity=0.8,
        size_max=5
    )

min_val = min(df['x'].min(), df['y'].min())
max_val = max(df['x'].max(), df['y'].max())

# Update layout to set equal ranges
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[min_val, max_val]),
        yaxis=dict(range=[min_val, max_val])
    )
)


fig.show()

plt.figure(figsize=(8,8))
plt.scatter(distanceListY[:, 0], distanceListY[:, 1], c='blue', label='distanceListA')
plt.scatter(distanceListB[:, 0], distanceListB[:, 1], c='yellow', edgecolor='black', label='distanceListB')
plt.scatter(centers[:,0], centers[:,1], color='red', label="Original waypoints")
#plt.scatter(x_smooth, y_smooth, color='red', label="smooth waypoints")
plt.plot(x_smooth, y_smooth, label="Smoothed B-spline fit", linewidth=2)
#plt.quiver(x_smooth[temp], y_smooth[temp], direction[0, temp] / 10, direction[1, temp] / 10, angles='xy', scale_units='xy', scale=1, color='green', label='Vector')
#plt.quiver(x_smooth[temp], y_smooth[temp], direction1[0, temp] / 10, direction1[1, temp] / 10, angles='xy', scale_units='xy', scale=1, color='green', label='Vector')

idx = np.linspace(0, len(x_smooth)-1, 30).astype(int)
arrow_scale = 200
for i in idx:
    tx, ty = dx_u[i], dy_u[i]
    t_norm = np.hypot(tx, ty)
    tx /= t_norm
    ty /= t_norm

    delta = steering[i]  # steering angle in radians

    wx =  np.cos(delta)*tx - np.sin(delta)*ty
    wy =  np.sin(delta)*tx + np.cos(delta)*ty

    plt.arrow(x_smooth[i],
              y_smooth[i],
              wx * arrow_scale,
              wy * arrow_scale,
              head_width=40,
              color="green")

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


######################## 'pass-by-value' vs 'pass-by-reference'? mby... test #############################

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


