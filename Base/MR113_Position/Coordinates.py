import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math



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

    print(f"input list:")
    print(coordinates_list)

    #Define ouput list
    Processed_list = []

    #Iterate through each coordinate
    for i, vector in enumerate(coordinates_list):
        Processed_list.append(pixel_to_relative_coordinates(vector, depth_list[i], fov, image_width, image_height)) ######## Add cone type into the data here ##
    
    print(f"Output list")
    print(Processed_list)

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
    print(updated) ############## print ##########
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

def rotatePointAroundPoint(point, car, angle):
    x = point[0]
    y = point[1]
    lastAngle = angle           #################
    angle = lastAngle - angle   #################
    cos = math.cos(angle)
    sin = math.sin(angle)
    temp = ((x-car[0])*cos - (y-car[1])*sin) + car[0]
    y = ((x-car[0])*sin + (y-car[1])*cos) + car[1]
    x = temp
    return

def movePoints(oldPoints, distance):
    for i, point in enumerate(oldPoints):
        point[1] -= distance
        if point[1] < 0:
            del oldPoints[i]           # removedPoint = points.pop(1)

def movePoint(oldPoint, distance):
    oldPoint[1] -= distance



################ Setup ##############
"""Some things have to run before 'rotatePointsAroundPoint(oldPoints, car, currentAngle)' and 
'movePoint(oldPoints, distance)' to make them function correctly."""

# Camera parameters
fov = 90
image_width = 1280
image_height = 720

# Other initial parameters
coordinates_list = [[36,300, 0],       ######## Initial input from M111 #########
                    [1240,300, 1],     # Some 'random' coordinates for testing purposes as no real test data is available currently.
                    [370,450, 0],      # Replace with the initial incomming data.
                    [850,450, 1],      # Same goes for depth_list
                    [470,550, 0],
                    [750,550, 1],
                    [520,666, 0],
                    [700,666, 1]]                                  ###### getHeliosDistances(coordinates_list) initial start value #######
depth_list = [500, 500, 1500, 1500, 2500, 2500, 3500, 3500]     # Some 'random' distances for testing purposes as no real test data is available currently.
angle = 0                           # readGyro(z) initial start value
lastAngle = 0                   # Initial "zero" / start orientation
distance = 0                        # readEncoder() initial start value
lastDistance = 0             # Initial "zero" / start encoder value
car = [0, 1500]                     # Car position (constant, the world moves around the car)
newPoints = one_frame_cone_positions(coordinates_list, depth_list, fov, image_width, image_height) # Initial frame of cones.
oldPoints = [[300,1500, 1],[-300,1500, 0],[300,750, 1],[-300,750, 0],[300,1, 1],[-300,1, 0]] #,[300,6000],[-300,6000]] # Some initial old points behind the car to ensure correct b-spline.
matchPoints(newPoints, oldPoints)


################ Program (loop) ##############
"""When new data is ready, the predicted cone locations are calculated (based on encoder and gyro/magno). 
Then the old and new points are matched."""

def main():
    global lastAngle
    global lastDistance

    # coordinates_list = dataFromM111()                         ###### Input from M111 #########                   
    # depth_list = getHeliosDistances(coordinates_list)         ###### Input from Helios #######

    # angle = readGyro(z)                                       ###### Input from gyro/magno ##########
    rotatePointsAroundPoint(oldPoints, car, angle - lastAngle)
    lastAngle = angle

    # distance = readEncoder()                                  ###### Input from encoder ##########
    movePoints(oldPoints, distance - lastDistance)
    lastDistance = distance

    newPoints = one_frame_cone_positions(coordinates_list, depth_list, fov, image_width, image_height)

    matchPoints(newPoints, oldPoints)
    # Export 'oldPoints' to use later in pipeline (M121)


################################################################


blueList = []
yellowList = []

for point in oldPoints:
    if point[2] == 0:
        blueList.append(point)
    else:
        yellowList.append(point)
        
yellowList = np.array(yellowList)
blueList = np.array(blueList)

plt.scatter(yellowList[:,0], yellowList[:,1], c='yellow', edgecolors='black', label='Pixel')
plt.scatter(blueList[:,0], blueList[:,1], c='blue', edgecolors='black', label='Pixel')
plt.axis('equal')
plt.show()


coordinates_list = [[36,280, 0],       ######## Initial input from M111 #########
                    [1240,280, 1],     # Some 'random' coordinates for testing purposes as no real test data is available currently.
                    [370,430, 0],      # Replace with the initial incomming data.
                    [850,430, 1],      # Same goes for depth_list
                    [470,530, 0],
                    [750,530, 1],
                    [520,646, 0],
                    [700,646, 1]]                                  ###### getHeliosDistances(coordinates_list) initial start value #######
depth_list = [450, 450, 1450, 1450, 2450, 2450, 3450, 3450]     # Some 'random' distances for testing purposes as no real test data is available currently.
angle = math.radians(5)                           # readGyro(z) initial start value
distance = 100                        # readEncoder() initial start value

main()

blueList = []
yellowList = []

for point in oldPoints:
    if point[2] == 0:
        blueList.append(point)
    else:
        yellowList.append(point)
        
yellowList = np.array(yellowList)
blueList = np.array(blueList)

plt.scatter(yellowList[:,0], yellowList[:,1], c='yellow', edgecolors='black', label='Pixel')
plt.scatter(blueList[:,0], blueList[:,1], c='blue', edgecolors='black', label='Pixel')
plt.axis('equal')
plt.show()



coordinates_list = []                                  ###### getHeliosDistances(coordinates_list) initial start value #######
depth_list = []     # Some 'random' distances for testing purposes as no real test data is available currently.
angle = math.radians(10)                           # readGyro(z) initial start value
distance = 200                        # readEncoder() initial start value

main()

blueList = []
yellowList = []

for point in oldPoints:
    if point[2] == 0:
        blueList.append(point)
    else:
        yellowList.append(point)
        
yellowList = np.array(yellowList)
blueList = np.array(blueList)

plt.scatter(yellowList[:,0], yellowList[:,1], c='yellow', edgecolors='black', label='Pixel')
plt.scatter(blueList[:,0], blueList[:,1], c='blue', edgecolors='black', label='Pixel')
plt.axis('equal')
plt.show()










# Track cones across frames and compute global position
def translate_cone_vectors_to_global_coordinates(frames, match_threshold=3000, max_missing_frames=5):
    """
    frames: list of (cone_positions) for each frame, where cone_positions is Nx2 array of [x, y]
    match_threshold: max distance to consider cones as same
    max_missing_frames: remove cones if unseen for these many frames
    """
    global_position = np.array([0.0, 0.0])
    cone_tracker = {}  # {id: {pos: np.array([x,y]), last_seen: frame_index}}
    next_id = 0
    trajectory = []

    for frame_idx, cones in enumerate(frames):
        # Match cones to tracker
        matched_ids = set()
        for cone in cones:
            best_match = None
            best_dist = float('inf')
            for cid, data in cone_tracker.items():
                dist = np.linalg.norm(cone - data['pos'])
                if dist < match_threshold and dist < best_dist:
                    best_match = cid
                    best_dist = dist
            if best_match is not None:
                # Update existing cone
                cone_tracker[best_match]['pos'] = cone
                cone_tracker[best_match]['last_seen'] = frame_idx
                matched_ids.add(best_match)
            else:
                # Add new cone
                cone_tracker[next_id] = {'pos': cone, 'last_seen': frame_idx}
                matched_ids.add(next_id)
                next_id += 1

        # Remove cones not seen for too long
        to_remove = [cid for cid, data in cone_tracker.items() if frame_idx - data['last_seen'] > max_missing_frames]
        for cid in to_remove:
            del cone_tracker[cid]

        # Compute translation using matched cones from previous frame
        if frame_idx > 0:
            prev_frame = frames[frame_idx - 1]
            shifts = []
            for cone in cones:
                if len(prev_frame) == 0:
                    continue
                dists = np.linalg.norm(prev_frame - cone, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < match_threshold:
                    shift = prev_frame[min_idx] - cone
                    shifts.append(shift)
            if shifts:
                avg_shift = np.mean(shifts, axis=0)
                global_position += -avg_shift  # negative because cones move opposite to car

        trajectory.append(global_position.copy())
        print(f"Frame {frame_idx}: Global Position = {global_position}")

    return global_position, np.array(trajectory)

# Load dataset
df = pd.read_csv(fr"MR113_Position\nascar_track_cones_dataset_synth3.csv")
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