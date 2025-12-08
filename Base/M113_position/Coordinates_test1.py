import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
import time



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
    for i, point in enumerate(points):
        for j, oldPoint in enumerate(oldPoints):
            dist = (oldPoint[0] - point[0])**2 + (oldPoint[1] - point[1])**2
            dist1 = (point[0])**2 + (point[1])**2
            pointList.append([dist, i, j, dist1])
    pointList = sorted(pointList, key=lambda x: x[0])# np.array( 

    for dist, i, j, dist1 in pointList: # Match points
        if dist > maxDist + (dist1 * 0.005): ################################## COEFFICIENT HERE!!! #############
            break
        if j not in updated:
            oldPoints[int(j)][0:2] = points[int(i)][0:2]
            updated.add(j)
    # print(updated) ############## print ##########
    for i in points: # Add new points
        if i not in oldPoints:
            oldPoints.append(i)
            updated.add(len(oldPoints)-1)
    
    # for i, oldPoint in enumerate(oldPoints): # Move old points # Old version
    #     if i not in updated:
    #         rotatePointAroundPoint(oldPoint, car, currentAngle)
    #         movePoint(oldPoint, distance)
    #         updated.add(i)
    
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
depth_listB = [500, 1500, 2500, 3500]     # Some 'random' distances for testing purposes as no real test data is available currently.
depth_listY = [500, 1500, 2500, 3500]
angle = 0                           # readGyro(z) initial start value
lastAngle = 0                   # Initial "zero" / start orientation
distance = 0                        # readEncoder() initial start value
lastDistance = 0             # Initial "zero" / start encoder value
car = [0, 1500]                     # Car position (constant, the world moves around the car)
newPointsB = one_frame_cone_positions(coordinates_listB, depth_listB, fov, image_width, image_height) # Initial frame of cones.
newPointsY = one_frame_cone_positions(coordinates_listY, depth_listY, fov, image_width, image_height) # Initial frame of cones.
oldPointsB = [[-300,5, 0],[-300,750, 0],[-300,1500, 0]] #,[300,6000],[-300,6000]] # Some initial old points behind the car to ensure correct b-spline.
oldPointsY = [[300,1, 1],[300,755, 1],[300,1505, 1]] #,[300,6000],[-300,6000]] # Some initial old points behind the car to ensure correct b-spline.
matchPoints(newPointsB, oldPointsB)
matchPoints(newPointsY, oldPointsY)

################ Program (loop) ##############
"""When new data is ready, the predicted cone locations are calculated (based on encoder and gyro/magno). 
Then the old and new points are matched."""

def main():
    global lastAngle
    global lastDistance

    # while dataFromM111AndHelios == none: ###### something like this
    #     wait()
    # coordinates_list = dataFromM111()                         ###### Input from M111 #########                   
    # depth_list = getHeliosDistances(coordinates_list)         ###### Input from Helios #######

    # angle = readGyro(z)                                       ###### Input from gyro/magno ##########
    rotatePointsAroundPoint(oldPointsB, car, angle - lastAngle)
    rotatePointsAroundPoint(oldPointsY, car, angle - lastAngle)
    lastAngle = angle

    # distance = readEncoder()                                  ###### Input from encoder ##########
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

def run(output_queue, serial_queue):
 
    while True: ##########################
    
    #for frame in range(200):
        global coordinates_listB, coordinates_listY, depth_listB, depth_listY, angle, distance #
        wheel_circumference = 577.6 # in mm
        pulses_per_revolution = 100
        
        # points_ = input_queue.get()
        # coordinates_list = []
        # depth_list = []
        # for point in points_:
        #     coordinates_list.append([point[0], point[1], point[3]])
        #     depth_list.append(point[2])
        # distance += 50

        while not serial_queue.empty():
            angle, encoder = serial_queue.get()
            distance = encoder * wheel_circumference / pulses_per_revolution
            print("Angle:", angle, "   distance:", distance) 
            angle = math.radians(angle)

        main()

        coordinates_listB = []
        coordinates_listY = []
        depth_listB = []
        depth_listY = []

        output_queue.put((oldPointsB, oldPointsY))
        # print(f"OutFromM113: {oldPoints} {time.time()}")
        plt.pause(0.2)




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