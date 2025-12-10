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

    used_i = set()

    #Update positions of matched points that dont exceed distance threshold
    for dist, i, j, dist1 in pointList: # Match points
        if dist > maxDist + (dist1 * 0.005): ################################## COEFFICIENT HERE!!! #############
            break
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

    
   # Only add to oldPoints after CONFIRM_FRAMES consecutive matches

    # Candidate buffer for this tracked list (Blue/Yellow separated by id(oldPoints))
    cands = _PENDING.get(id(oldPoints), [])

    # 1) Detections NOT consumed by tracked matching
    unmatched = [points[i] for i in range(len(points)) if i not in used_i]

    # 2) Associate unmatched detections to existing candidates (greedy by distance)
    pairs = []
    for ui, d in enumerate(unmatched):
        for cj, c in enumerate(cands):
            # Optional type check (B/Y calls already keep types separate)
            if len(d) >= 3 and c.get("type") is not None and c["type"] != d[2]:
                continue
            dist2 = (d[0] - c["x"])**2 + (d[1] - c["y"])**2
            pairs.append((dist2, ui, cj))
    pairs.sort(key=lambda x: x[0])

    used_ui = set()
    matched_c_idxs = set()
    for dist2, ui, cj in pairs:
        if dist2 > maxDist:
            break
        if ui in used_ui:
            continue
        # Update candidate with current detection
        cands[cj]["x"] = unmatched[ui][0]
        cands[cj]["y"] = unmatched[ui][1]
        cands[cj]["hit"] = cands[cj].get("hit", 0) + 1
        cands[cj]["miss"] = 0
        used_ui.add(ui)
        matched_c_idxs.add(cj)

    # 3) Candidates not matched this frame → miss++
    for idx, c in enumerate(cands):
        if idx not in matched_c_idxs:
            c["miss"] = c.get("miss", 0) + 1

    # 4) Truly new detections → start as candidates (hit=1)
    for ui, d in enumerate(unmatched):
        if ui not in used_ui:
            cands.append({
                "x": d[0],
                "y": d[1],
                "type": d[2] if len(d) >= 3 else None,
                "hit": 1,
                "miss": 0
            })

    # 5) Promote confirmed candidates to tracked (add with miss_count=0), mark updated
    for idx in range(len(cands) - 1, -1, -1):
        c = cands[idx]
        if c["hit"] >= CONFIRM_FRAMES:
            oldPoints.append([c["x"], c["y"], c["type"], 0])  # [x, y, type, miss_count=0]
            updated.add(len(oldPoints) - 1)
            del cands[idx]

    # 6) Prune flickering candidates that disappeared
    for idx in range(len(cands) - 1, -1, -1):
        if cands[idx]["miss"] > CANDIDATE_MISS_MAX:
            del cands[idx]

    # 7) Persist candidate buffer for this list
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


# Candidate buffers keyed per tracked list (Blue/Yellow separated by id(oldPoints))
_PENDING = {}               # id(oldPoints) -> list of {'x','y','type','hit','miss'}

CONFIRM_FRAMES = 3          # require 3 consecutive matches to start tracking (set to 2 if you prefer)
CANDIDATE_MISS_MAX = 2      # drop a candidate if it disappears for 2 frames



slope = math.tan(math.radians((fov)/2))

# Other initial parameters
coordinates_listB = [[36,300, 0],
                    [370,450, 0],
                    [470,550, 0],
                    [520,666, 0]]
coordinates_listY = [[1240,300, 1],
                    [850,450, 1],
                    [750,550, 1],
                    [700,666, 1]]
depth_listB = [500, 1500, 2500, 3500]     # Some 'random' distances for testing purposes as no real test data is available currently.
depth_listY = [500, 1500, 2500, 3500]
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

    rotatePointsAroundPoint(oldPointsB, car, angle - lastAngle)
    rotatePointsAroundPoint(oldPointsY, car, angle - lastAngle)
    lastAngle = angle

    movePoints1(oldPointsB, oldPointsY, distance - lastDistance)
    lastDistance = distance

    newPointsB = one_frame_cone_positions(coordinates_listB, depth_listB, fov, image_width, image_height)
    newPointsY = one_frame_cone_positions(coordinates_listY, depth_listY, fov, image_width, image_height)

    matchPoints(newPointsB, oldPointsB)
    matchPoints(newPointsY, oldPointsY)
    # Export 'oldPoints' to use later in pipeline (M121)
    roundPoints(oldPointsB)
    roundPoints(oldPointsY)
    # print(f"OutFromM113: {oldPoints}")


######### run ###########

def run(input_queue, output_queue, serial_queue): #
    global coordinates_listB, coordinates_listY, depth_listB, depth_listY, angle, distance
    main()

    with open("m113log", "a", newline="") as f:
        writer = csv.writer(f)

        while True: ##########################
            
            wheel_circumference = 577.6 # in mm
            pulses_per_revolution = 100
            
            depth_listB = []
            depth_listY = []
            coordinates_listB, coordinates_listY = input_queue.get()
            depth_listB = [p[2] for p in coordinates_listB]
            depth_listY = [p[2] for p in coordinates_listY]

            while not serial_queue.empty():
                angle, encoder = serial_queue.get()
                distance = encoder * wheel_circumference / pulses_per_revolution
                print("Angle:", angle, "   distance:", distance)
                # angle = math.radians(angle)
            
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