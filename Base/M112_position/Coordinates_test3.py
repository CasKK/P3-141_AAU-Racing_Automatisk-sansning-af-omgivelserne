import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
import time
import serial
import csv
import datetime
import json

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
newPointsB = one_frame_cone_positions(coordinates_listB, depth_listB, fov, image_width, image_height) # Initial frame of cones.
newPointsY = one_frame_cone_positions(coordinates_listY, depth_listY, fov, image_width, image_height) # Initial frame of cones.
oldPointsB = [[-300,5, 0, 0],[-300,750, 0, 0],[-300,1500, 0, 0]] #,[300,6000],[-300,6000]] # Some initial old points behind the car to ensure correct b-spline.
oldPointsY = [[300,1, 1, 0],[300,755, 1, 0],[300,1505, 1, 0]] #,[300,6000],[-300,6000]] # Some initial old points behind the car to ensure correct b-spline.
matchPoints(newPointsB, oldPointsB)
matchPoints(newPointsY, oldPointsY)


################ Program (loop) ##############
"""When new data is ready, the predicted cone locations are calculated (based on encoder and gyro/magno). 
Then the old and new points are matched."""

def main():
    global lastAngle
    global lastDistance

    rotatePointsAroundPoint(oldPointsB, car, angle - lastAngle) # Rotate old blue points based gyro feedback
    rotatePointsAroundPoint(oldPointsY, car, angle - lastAngle) # Rotate old yellow points based on gyro feedback
    lastAngle = angle   # Update last angle

    movePoints1(oldPointsB, oldPointsY, distance - lastDistance) # Move old points based on encoder feedback
    lastDistance = distance 

    newPointsB = one_frame_cone_positions(coordinates_listB, depth_listB, fov, image_width, image_height) # Get new blue cone positions
    newPointsY = one_frame_cone_positions(coordinates_listY, depth_listY, fov, image_width, image_height) # Get new yellow cone positions

    matchPoints(newPointsB, oldPointsB) # Match new blue points to old blue points
    matchPoints(newPointsY, oldPointsY) # Match new yellow points to old yellow points
    # Export 'oldPoints' to use later in pipeline (M121)
    roundPoints(oldPointsB) # Round points for easier handling
    roundPoints(oldPointsY) # Round points for easier handling
    # print(f"OutFromM113: {oldPoints}")


######### run ###########

def run(input_queue, output_queue, serial_queue): # Run function for m113
    global coordinates_listB, coordinates_listY, depth_listB, depth_listY, angle, distance
    main() # Initial run to setup oldPoints

    with open("m113log", "a", newline="") as f: # Open the m113 log file for appending
        writer = csv.writer(f)

        while True: # Main loop
            
            wheel_circumference = 520.3 # in mm
            pulses_per_revolution = 100 # encoder pulses per wheel revolution
            
            depth_listB = [] # initialize helios depth list for blue cones
            depth_listY = [] # initialize helios depth list for yellow cones
            coordinates_listB, coordinates_listY = input_queue.get() # get new coordinates from m111
            depth_listB = [p[2] for p in coordinates_listB] # extract depth values for blue cones
            depth_listY = [p[2] for p in coordinates_listY] # extract depth values for yellow cones

            while not serial_queue.empty(): # get all available serial data
                angle, encoder = serial_queue.get() # get angle and encoder values
                distance = encoder * wheel_circumference / pulses_per_revolution
                print("Angle:", angle, "   distance:", distance)
                angle = math.radians(angle)
            
            main() # Run main processing function
            # Enforce max queue length of 5
            if output_queue.qsize() >= 5: # limit queue size to prevent m121 falling behind
                try:
                    output_queue.get_nowait()  # remove oldest item
                except:
                    pass

            output_queue.put((oldPointsB, oldPointsY)) # send updated points to m121

            #Log
            timestamp = time.time()
            writer.writerow([timestamp, oldPointsB, oldPointsY])
            f.flush()

            #print(f"OutFromM113nr2: {oldPointsB} --- {oldPointsY} --- {time.time()}")
            # plt.pause(0.01)




# coordinates_list = [[36,280, 0],
#                     [1240,280, 1],
#                     [370,430, 0],
#                     [850,430, 1],
#                     [470,530, 0],
#                     [750,530, 1],
#                     [520,646, 0],
#                     [700,646, 1]]
# depth_list = [450, 450, 1450, 1450, 2450, 2450, 3450, 3450]







############# Victor stuff: ############



# Track cones across frames and compute global position
# def translate_cone_vectors_to_global_coordinates(frames, match_threshold=3000, max_missing_frames=5):
#     """
#     frames: list of (cone_positions) for each frame, where cone_positions is Nx2 array of [x, y]
#     match_threshold: max distance to consider cones as same
#     max_missing_frames: remove cones if unseen for these many frames
#     """
#     global_position = np.array([0.0, 0.0])
#     cone_tracker = {}  # {id: {pos: np.array([x,y]), last_seen: frame_index}}
#     next_id = 0
#     trajectory = []

#     for frame_idx, cones in enumerate(frames):
#         # Match cones to tracker
#         matched_ids = set()
#         for cone in cones:
#             best_match = None
#             best_dist = float('inf')
#             for cid, data in cone_tracker.items():
#                 dist = np.linalg.norm(cone - data['pos'])
#                 if dist < match_threshold and dist < best_dist:
#                     best_match = cid
#                     best_dist = dist
#             if best_match is not None:
#                 # Update existing cone
#                 cone_tracker[best_match]['pos'] = cone
#                 cone_tracker[best_match]['last_seen'] = frame_idx
#                 matched_ids.add(best_match)
#             else:
#                 # Add new cone
#                 cone_tracker[next_id] = {'pos': cone, 'last_seen': frame_idx}
#                 matched_ids.add(next_id)
#                 next_id += 1

#         # Remove cones not seen for too long
#         to_remove = [cid for cid, data in cone_tracker.items() if frame_idx - data['last_seen'] > max_missing_frames]
#         for cid in to_remove:
#             del cone_tracker[cid]

#         # Compute translation using matched cones from previous frame
#         if frame_idx > 0:
#             prev_frame = frames[frame_idx - 1]
#             shifts = []
#             for cone in cones:
#                 if len(prev_frame) == 0:
#                     continue
#                 dists = np.linalg.norm(prev_frame - cone, axis=1)
#                 min_idx = np.argmin(dists)
#                 if dists[min_idx] < match_threshold:
#                     shift = prev_frame[min_idx] - cone
#                     shifts.append(shift)
#             if shifts:
#                 avg_shift = np.mean(shifts, axis=0)
#                 global_position += -avg_shift  # negative because cones move opposite to car

#         trajectory.append(global_position.copy())
#         print(f"Frame {frame_idx}: Global Position = {global_position}")

#     return global_position, np.array(trajectory)

# Load dataset
# df = pd.read_csv(fr"MR113_Position\nascar_track_cones_dataset_synth3.csv")
# plt.scatter(df["pixel_x"], df["pixel_y"], c='yellow', edgecolors='black', label='Pixel')
# plt.axis('equal')
# plt.show()


# # Step 1: Process frames into local cone positions
# frames = []
# for frame_idx, group in df.groupby("frame"):
#     coordinates_list = group[["pixel_x", "pixel_y"]].values
#     depth_list = group["depth_mm"].values
#     processed_list = one_frame_cone_positions(coordinates_list, depth_list, fov, image_width, image_height)
#     frames.append(processed_list)

# # Step 2: Compute global trajectory
# _, trajectory = translate_cone_vectors_to_global_coordinates(frames)

# # Step 3: Plot interactive trajectory
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=trajectory[:, 0],
#     y=trajectory[:, 1],
#     mode='lines+markers',
#     name='Vehicle Path'
# ))
# fig.update_layout(
#     title='Live Vehicle Trajectory on NASCAR Track',
#     xaxis_title='X Position (mm)',
#     yaxis_title='Y Position (mm)',
#     xaxis=dict(scaleanchor="y", scaleratio=1),
#     template='plotly_dark'
# )
# fig.show()