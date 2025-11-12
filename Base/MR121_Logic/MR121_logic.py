
import numpy as np
import pandas as pd
import time
import copy


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

def closestNP(vectorListA, vectorListB):
    time_start = time.time()
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

    time_end = time.time()
    print(f"Runtime: {time_end - time_start:.4f} seconds")
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


def centers(distanceListA, distanceListB):
    center = []

    for i, (vecA, vecB) in enumerate(zip(distanceListA, distanceListB)):
        center.append([(vecA[0] - vecB[0]) / 2, (vecA[1] - vecB[1]) / 2])
    center = np.array(center)
    print(center)



distanceListB, distanceListY = closestNP(inputVectorsB, inputVectorsY)

distanceLista, distanceListd = closestPandasQuick(inputVectorsB, inputVectorsY)

distanceLista, distanceListd = closestPandasSimple(inputVectorsB, inputVectorsY)


centers(distanceListB, distanceListY)































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


# [[-350.13, 50.9, 'blue'], [-350.34, 100.9, 'blue'], [-350.56, 200.9, 'blue'], [-350.13, 250.9, 'blue'], [-350.34, 300.9, 'blue'], [-350.56, 400.9, 'blue']]
# [[-350.13, 50.9, 'blue', -0.12999999999999545], [-350.34, 100.9, 'blue', -0.339999999999975], [-350.56, 200.9, 'blue', -0.5600000000000023], [-350.13, 250.9, 'blue', -0.12999999999999545], [-350.34, 300.9, 'blue', -0.339999999999975], [-350.56, 400.9, 'blue', -0.5600000000000023]]
# [[-350.13, 50.9, 'blue', -0.12999999999999545], [-350.34, 100.9, 'blue', -0.339999999999975], [-350.56, 200.9, 'blue', -0.5600000000000023], [-350.13, 250.9, 'blue', -0.12999999999999545], [-350.34, 300.9, 'blue', -0.339999999999975], [-350.56, 400.9, 'blue', -0.5600000000000023]]