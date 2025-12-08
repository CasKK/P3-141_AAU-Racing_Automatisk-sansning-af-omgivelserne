import math
import time
import csv
import json

# Following Code Will, use an input image coordinate and depth to generate xy vector to a cone for a Driverless Vehicle

def pixel_to_relative_coordinates(coordinates, depth, fov, image_width, image_height):
    Image_centerline = image_width / 2
    FOV_radians = math.radians(fov)

    angle_per_pixel = FOV_radians / image_width
    pixel_offset = coordinates[0] - Image_centerline
    angle_to_cone = pixel_offset * angle_per_pixel

    y_vector = depth * math.cos(angle_to_cone)
    x_vector = depth * math.sin(angle_to_cone)

    Relative_coordinate_array = [x_vector, y_vector + car[1], coordinates[2]]
    return Relative_coordinate_array

def one_frame_cone_positions(coordinates_list, depth_list, fov, image_width, image_height):

    Processed_list = []

    for i, vector in enumerate(coordinates_list):
        if depth_list[i] < 6000:
            Processed_list.append(pixel_to_relative_coordinates(vector, depth_list[i], fov, image_width, image_height))

    return Processed_list


def matchPoints(points, oldPoints, maxDist = 200*200):
    pointList = []
    updated = set()
    for i, point in enumerate(points):
        for j, oldPoint in enumerate(oldPoints):
            dist = (oldPoint[0] - point[0])**2 + (oldPoint[1] - point[1])**2
            dist1 = (point[0])**2 + (point[1])**2
            pointList.append([dist, i, j, dist1])
    pointList = sorted(pointList, key=lambda x: x[0])

    for dist, i, j, dist1 in pointList:
        if dist > maxDist + (dist1 * 0.005):
            break
        if j not in updated:
            oldPoints[int(j)][0:2] = points[int(i)][0:2]
            updated.add(j)
    for i in points:
        if i not in oldPoints:
            oldPoints.append(i)
            updated.add(len(oldPoints)-1)

def rotatePointsAroundPoint(points, car, angle):
    cos = math.cos(angle)
    sin = math.sin(angle)
    for point in points:
        x = point[0]
        y = point[1]
        point[0] = ((x-car[0])*cos - (y-car[1])*sin) + car[0]
        point[1] = ((x-car[0])*sin + (y-car[1])*cos) + car[1]

def movePoints(oldPoints, distance):
    for i in range(len(oldPoints) - 1, -1, -1):
        oldPoints[i][1] -= distance
        if oldPoints[i][1] < 0:
            del oldPoints[i]

def movePoints1(oldPointsA, oldPointsB, distance):
    for i in range(len(oldPointsB) - 1, -1, -1):
        oldPointsB[i][1] -= distance
    for i in range(len(oldPointsA) - 1, -1, -1):
        oldPointsA[i][1] -= distance
        if oldPointsA[i][1] < 0 and oldPointsB[i][1] < 0:
            del oldPointsA[i]
            del oldPointsB[i]

def roundPoints(points):
    for point in points:
        point[0] = int(point[0])
        point[1] = int(point[1])

################ Setup ##############

# Camera parameters
fov = 60
image_width = 1280
image_height = 720

# Other initial parameters
coordinates_listB = [[36,300, 0],
                    [370,450, 0],
                    [470,550, 0],
                    [520,666, 0]]
coordinates_listY = [[1240,300, 1],
                    [850,450, 1],
                    [750,550, 1],
                    [700,666, 1]]
depth_listB = [500, 1500, 2500, 3500]
depth_listY = [500, 1500, 2500, 3500]
angle = 0
lastAngle = 0
distance = 0
lastDistance = 0
car = [0, 1500]
newPointsB = one_frame_cone_positions(coordinates_listB, depth_listB, fov, image_width, image_height)
newPointsY = one_frame_cone_positions(coordinates_listY, depth_listY, fov, image_width, image_height)
oldPointsB = [[-300,5, 0],[-300,750, 0],[-300,1500, 0]]
oldPointsY = [[300,1, 1],[300,755, 1],[300,1505, 1]]
matchPoints(newPointsB, oldPointsB)
matchPoints(newPointsY, oldPointsY)

def main():
    global lastAngle
    global lastDistance

    rotatePointsAroundPoint(oldPointsB, car, angle - lastAngle)
    rotatePointsAroundPoint(oldPointsY, car, angle - lastAngle)
    lastAngle = angle

    movePoints1(oldPointsB, oldPointsY, distance - lastDistance)
    lastDistance = distance

    newPointsB = one_frame_cone_positions(coordinates_listB, depth_listB, fov, image_width, image_height)
    newPointsY = one_frame_cone_positions(coordinates_listY, depth_listY, fov, image_width, image_height)

    matchPoints(newPointsB, oldPointsB)
    matchPoints(newPointsY, oldPointsY)
    roundPoints(oldPointsB)
    roundPoints(oldPointsY)

def run(output_queue, serial_queue):
 
    # with open("m113log", "a", newline="") as f:
    #     writer = csv.writer(f)
    while True:
    
        global coordinates_listB, coordinates_listY, depth_listB, depth_listY, angle, distance
        wheel_circumference = 577.6
        pulses_per_revolution = 100

        #while not serial_queue.empty():
        angle, encoder = serial_queue.get()
        distance = encoder * wheel_circumference / pulses_per_revolution
        print("Angle:", angle, "   distance:", distance) 
        angle = math.radians(angle)

        main()
        if output_queue.qsize() >= 5:
            try:
                output_queue.get_nowait()
            except:
                pass

        coordinates_listB = []
        coordinates_listY = []
        depth_listB = []
        depth_listY = []

        output_queue.put((oldPointsB, oldPointsY))

        # timestamp = time.time()
        # writer.writerow([timestamp, json.dumps(oldPointsB), json.dumps(oldPointsY)])
        # f.flush()

        print(f"OutFromM113nr2: {oldPointsB} --- {oldPointsY} --- {time.time()}")
        #time.sleep(0.2)