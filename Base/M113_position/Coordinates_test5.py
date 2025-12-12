import math
import time
import csv

# Following Code Will, use an input image coordinate and depth to generate xy vector to a cone for a Driverless Vehicle

def pixel_to_relative_coordinates(coordinates, depth, fov, image_width, image_height):
    #Convert pixel coordinates angle and distance to cone vector
    # parameters:
    # coordinates: (x, y, type) pixel coordinates for the cone
    # depth: distance from camera to cone in millimeters
    # fov: camera field of view in degrees
    # image_width: width of the image in pixels
    # image_height: height of the image in pixels

    Image_centerline = image_width / 2
    FOV_radians = math.radians(fov)

    # Calculate the angle per pixel
    angle_per_pixel = FOV_radians / image_width
    pixel_offset = coordinates[0] - Image_centerline
    angle_to_cone = pixel_offset * angle_per_pixel

    # Calcualte cone vector in vehicle coordinate
    y_vector = depth * math.cos(angle_to_cone)
    x_vector = depth * math.sin(angle_to_cone)

    Relative_coordinate_array = [x_vector, y_vector + car[1], coordinates[2]]
     # Return x,y + cone type
    return Relative_coordinate_array

def one_frame_cone_positions(coordinates_list, depth_list, fov, image_width, image_height):
    # Process a list of cone coordinates 
    #parameters: 
    # coordinates_list: (x,y,type) list of pixel coordinates for cone
    # depth: list of distances from dv to cones in millimeters
    # fov: camera field of view,  image_width 
    # image_width: width of the image in pixels 
    # image height: height of the image in pixels

    # print(f"input list: {coordinates_list}")
    # print(f"depth list: {depth_list}")

    #Define ouput list
    Processed_list = []

    #Iterate through each coordinate
    for i, vector in enumerate(coordinates_list):
        if depth_list[i] < 6000:
            Processed_list.append(pixel_to_relative_coordinates(vector, depth_list[i], fov, image_width, image_height)) ######## Add cone type into the data here ##
    
    # print(f"Output list {Processed_list}")

    return Processed_list


def matchPoints(points, oldPoints, maxDist = 200*200):######################## COEFFICIENT HERE!!! #############
    pointList = []
    updated = set()
    miss_threshold = 5
    
    #Build All Candidate pairs and sort by distance
    for i, point in enumerate(points):
        for j, oldPoint in enumerate(oldPoints):
            dist = (oldPoint[0] - point[0])**2 + (oldPoint[1] - point[1])**2
            dist1 = (point[0])**2 + (point[1])**2
            pointList.append([dist, i, j, dist1])
    pointList = sorted(pointList, key=lambda x: x[0])# np.array( 

    used_i = set() # set that keeps track of unmatched cones 

    #Update positions of matched points that dont exceed distance threshold
    for dist, i, j, dist1 in pointList: # Match points
        if dist > maxDist + (dist1 * 0.005): ################################## COEFFICIENT HERE!!! #############
            break
        if i in used_i:
            continue
        if j not in updated:
            oldPoints[int(j)][0:2] = points[int(i)][0:2]
            updated.add(j)
            used_i.add(i)
    # print(updated) ############## print ##########

    #Add new points (Give them IDS)

    # for p in points: # Add new points
    #     exists = False
    #     for op in oldPoints:
    #         if op[0] == p[0] and op[1] == p[1] and p[2] == p[2]:
    #             exists = True
    #             break
    #     if not exists:
    #         oldPoints.append([p[0], p[1], p[2], 0])
    #         updated.add(len(oldPoints)-1)

    

    #Buffer of candidates
    cands = _PENDING.get(id(oldPoints), [])

    # A list of points that were not matched in the previous section.
    unmatched = [points[i] for i in range(len(points)) if i not in used_i]

    # Compare and match detections to previous candidates.
    pairs = []
    for i, unmatched_point in enumerate(unmatched): # Loops through all unmatched points this frame
        for j, candidates in enumerate(cands): #Loops through all potential candidates last frame
            dist2 = (unmatched_point[0] - candidates["x"])**2 + (unmatched_point[1] - candidates["y"])**2   #Calculates distance
            pairs.append((dist2, i, j)) #appends distance and, the counter variables i and j
    pairs.sort(key=lambda x: x[0]) #Sorts everything


    # creates two lists of ids for the used unmatched points. ie those that got matched, and a list of ids for the candidates that got matched.
    used_unmatched_points = set() 
    matched_candidates_ids = set()

    for dist2, i, j in pairs: # Loop through all of the pairs
        if dist2 > maxDist: # if the distance is greater than max distance, stop matching points
            break
        if i in used_unmatched_points: # Skip if unmatched point is already used
            continue
        if j in matched_candidates_ids: # Skip if unmatched point is already used
            continue
        # Update candidate with current detection
        cands[j]["x"] = unmatched[i][0] # Update candidate position x
        cands[j]["y"] = unmatched[i][1] # Update canditepositio y
        cands[j]["hit"] = cands[j].get("hit", 0) + 1 # Increment the hit count
        cands[j]["miss"] = 0 # reset miss count
        used_unmatched_points.add(i) # mark the unmatched point as used 
        matched_candidates_ids.add(j) # mark the candidate as matched

    # Simple, loop through all candidates and increment miss count for those that were not matched.
    for idx, c in enumerate(cands):
        if idx not in matched_candidates_ids:
            c["miss"] = c.get("miss", 0) + 1

    # Actually new candidates (not matched prior to this)
    for i, unmatched_point in enumerate(unmatched):
        if i not in used_unmatched_points:
            cands.append({
                "x": unmatched_point[0],
                "y": unmatched_point[1],
                "type": unmatched_point[2] if len(unmatched_point) >= 3 else None,
                "hit": 1,
                "miss": 0
            }) # append new candidate

    # promote confirmed candidates to oldPoints
    for idx in range(len(cands) - 1, -1, -1): # Loop backwards through candidates
        candidate = cands[idx] #Current candidate
        if candidate["hit"] >= CONFIRM_FRAMES: # If candidate has enough hits
            oldPoints.append([candidate["x"], candidate["y"], candidate["type"], 0])  # Add to oldPoints
            updated.add(len(oldPoints) - 1) # Mark as updated
            del cands[idx]  # Remove from candidates

    # destroy candidates that missed too many frames.
    for idx in range(len(cands) - 1, -1, -1):
        if cands[idx]["miss"] > CANDIDATE_MISS_MAX:
            del cands[idx]

    # Update global candidate buffer
    _PENDING[id(oldPoints)] = cands
    # ----------------------------------------------------------------------    
    # keep track of missed countss

    for idx, op in enumerate(oldPoints):
        if idx in updated:
            op[3] = 0 #Reset misscount
        else:
            op[3] += 1
    
    # Destroy fake points
    for k in range(len(oldPoints) -1 , -1, -1):
        if oldPoints[k][3] >= miss_threshold:
            if abs(oldPoints[k][0]) >= 2000:
                del oldPoints[k]
            elif slope*oldPoints[k][0]+car[1] < oldPoints[k][1] and (-slope)*oldPoints[k][0]+car[1] < oldPoints[k][1]:
                del oldPoints[k]

            
        
    # print(pointList)

def rotatePointsAroundPoint(points, car, angle):
    cos = math.cos(angle)
    sin = math.sin(angle)
    for point in points:
        x = point[0]
        y = point[1]
        point[0] = ((x-car[0])*cos - (y-car[1])*sin) + car[0]
        point[1] = ((x-car[0])*sin + (y-car[1])*cos) + car[1]

def movePoints(oldPoints, distance):
    for i in range(len(oldPoints) - 1, -1, -1):  # Loop from last to first
        oldPoints[i][1] -= distance
        if oldPoints[i][1] < 0:
            del oldPoints[i]        # removedPoint = points.pop(1)

def movePoints1(oldPointsA, oldPointsB, distance):
    for i in range(len(oldPointsB) - 1, -1, -1):  # Loop from last to first
        oldPointsB[i][1] -= distance
    for i in range(len(oldPointsA) - 1, -1, -1):  # Loop from last to first
        oldPointsA[i][1] -= distance
        if oldPointsA[i][1] < 0 and oldPointsB[i][1] < 0:
            del oldPointsA[i]        # removedPoint = points.pop(1)
            del oldPointsB[i]

def roundPoints(points):
    for point in points:
        point[0] = int(point[0])# round(point[0], 1)
        point[1] = int(point[1])# round(point[1], 1)

################ Setup ##############
"""Some things have to run before 'rotatePointsAroundPoint(oldPoints, car, currentAngle)' and 
'movePoint(oldPoints, distance)' to make them function correctly."""

# Camera parameters
fov = 60
image_width = 640
image_height = 480


# Buffer for candidates
_PENDING = {}               # id(oldPoints) -> list of {'x','y','type','hit','miss'}
# ['x'] = [0]

CONFIRM_FRAMES = 2         # consecutive matches threshold
CANDIDATE_MISS_MAX = 2      # drop a candidate if it disappears for more than 



slope = math.tan(math.radians((180-50)/2))

# Other initial parameters
coordinates_listB = []#[36,300, 0],
                    # [370,450, 0],
                    # [470,550, 0],
                    # [520,666, 0]]
coordinates_listY = []#[1240,300, 1],
                    # [850,450, 1],
                    # [750,550, 1],
                    # [700,666, 1]]
depth_listB = []#500, 1500, 2500, 3500]     # Some 'random' distances for testing purposes as no real test data is available currently.
depth_listY = []#500, 1500, 2500, 3500]
angle = 0                       # readGyro(z) initial start value
lastAngle = 0                   # Initial "zero" / start orientation ##########################
distance = 0                    # readEncoder() initial start value
lastDistance = 0                # Initial "zero" / start encoder value ###########################
car = [0, 1500]                 # Car position (constant, the world moves around the car)
# newPointsB = one_frame_cone_positions(coordinates_listB, depth_listB, fov, image_width, image_height) # Initial frame of cones.
# newPointsY = one_frame_cone_positions(coordinates_listY, depth_listY, fov, image_width, image_height) # Initial frame of cones.
some1 = 1500
oldPointsB =    [[-300,     5,      0,  0],
                [-300,      750,    0,  0],
                [-300,      1500,   0,  0],
                [-368.0,    540.0 + some1,  0,  0],
                [-365.0,    1100.0 + some1, 0,  0],
                [-403.0,    2250.0 + some1, 0,  0],
                [-1.0,      3680.0 + some1, 0,  0],
                [1044.0,    4540.0 + some1, 0,  0],
                [1847.0,    4690.0 + some1, 0,  0],
                [2477.0,    4800.0 + some1, 0,  0],
                [2900.0,    5200.0 + some1, 0,  0],
                [3000.0,    5600.0 + some1, 0,  0],
                [2910.0,    6300.0 + some1, 0,  0]] # Some initial old points behind the car to ensure correct b-spline.
oldPointsY =    [[300,      1,      1,  0],
                [300,       755,    1,  0],
                [300,       1505,   1,  0],
                [353.0,     670.0 + some1,  1,  0],
                [377.0,     1330.0 + some1, 1,  0],
                [351.0,     2150.0 + some1, 1,  0],
                [779.0,     3230.0 + some1, 1,  0],
                [1353.0,    3710.0 + some1, 1,  0],
                [1977.0,    3890.0 + some1, 1,  0],
                [3000.0,    4000.0 + some1, 1,  0],
                [3700.0,    4700.0 + some1, 1,  0],
                [3800.0,    5700.0 + some1, 1,  0],
                [3810.0,    6300.0 + some1, 1,  0]] # Some initial old points behind the car to ensure correct b-spline.
# matchPoints(newPointsB, oldPointsB)
# matchPoints(newPointsY, oldPointsY)


################ Program (loop) ##############
"""When new data is ready, the predicted cone locations are calculated (based on encoder and gyro/magno). 
Then the old and new points are matched."""

def main():
    global lastAngle
    global lastDistance

    rotatePointsAroundPoint(oldPointsB, car, angle - lastAngle)
    rotatePointsAroundPoint(oldPointsY, car, angle - lastAngle)
    lastAngle = angle

    movePoints1(oldPointsB, oldPointsY, distance - lastDistance)
    lastDistance = distance

    # newPointsB = one_frame_cone_positions(coordinates_listB, depth_listB, fov, image_width, image_height)
    # newPointsY = one_frame_cone_positions(coordinates_listY, depth_listY, fov, image_width, image_height)

    # matchPoints(newPointsB, oldPointsB)
    # matchPoints(newPointsY, oldPointsY)
    # Export 'oldPoints' to use later in pipeline (M121)
    roundPoints(oldPointsB)
    roundPoints(oldPointsY)
    # print(f"OutFromM113: {oldPoints}")


######### run ###########

def run(output_queue, serial_queue):
    global coordinates_listB, coordinates_listY, depth_listB, depth_listY, angle, distance
    main()

    with open("m113log", "a", newline="") as f:
        writer = csv.writer(f)

        while True: ##########################
            
            wheel_circumference = 577.6 # in mm
            pulses_per_revolution = 100
            
            angle, encoder = serial_queue.get()
            distance = encoder * wheel_circumference / pulses_per_revolution
            print("Angle:", angle, "   distance:", distance)
            angle = math.radians(angle)
            
            main()
            # Enforce max queue length of 5
            if output_queue.qsize() >= 5:
                try:
                    output_queue.get_nowait()  # remove oldest item
                except:
                    pass

            output_queue.put((oldPointsB, oldPointsY))

            #Log
            timestamp = time.time()
            writer.writerow([timestamp, oldPointsB, oldPointsY])
            f.flush()

            #print(f"OutFromM113nr2: {oldPointsB} --- {oldPointsY} --- {time.time()}")
            # plt.pause(0.01)
