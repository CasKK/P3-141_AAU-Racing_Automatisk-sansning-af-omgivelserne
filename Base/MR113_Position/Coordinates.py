import numpy as np
import cv2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
import copy


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
    FOV_radians = np.deg2rad(fov)

    # Calculate the angle per pixel
    angle_per_pixel = FOV_radians / image_width
    pixel_offset = coordinates[0] - Image_centerline
    angle_to_cone = pixel_offset * angle_per_pixel

    # Calcualte cone vector in vehicle coordinate
    y_vector = depth * np.cos(angle_to_cone)
    x_vector = depth * np.sin(angle_to_cone)

    Relative_coordinate_array = np.array([x_vector, y_vector])
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
    Processed_list = np.zeros((len(coordinates_list), 2))

    #Iterate through each coordinate
    for i, vector in enumerate(coordinates_list):
        Processed_list[i] = pixel_to_relative_coordinates(vector, depth_list[i], fov, image_width, image_height)
    
    print(f"Output list")
    print(Processed_list)

    return Processed_list


def matchPoints(points, oldPoints, maxDist = 300*300):
    pointList = []
    updated = set()
    # oldPoints = copy.deepcopy(oldPoints)
    for i, point in enumerate(points):
        for j, oldPoint in enumerate(oldPoints):
            dist = (oldPoint[0] - point[0])**2 + (oldPoint[1] - point[1])**2
            pointList.append([dist, i, j])
    pointList = np.array(sorted(pointList, key=lambda x: x[0]))

    for dist, i, j in pointList:
        if dist > maxDist:
            break
        if j not in updated:
            oldPoints[int(j)][0:2] = points[int(i)][0:2]
            updated.add(j)


    print(pointList)

def rotatePointsAroundPoint(points, car, angle):
    x = points[:,0]
    y = points[:,1]
    cos = math.cos(angle)
    sin = math.sin(angle)
    for n in range(len(x)):
        temp = ((x[n]-car[0])*cos - (y[n]-car[1])*sin) + car[0]
        y[n] = ((x[n]-car[0])*sin + (y[n]-car[1])*cos) + car[1]
        x[n] = temp
    return






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
plt.scatter(df["pixel_x"], df["pixel_y"], c='yellow', edgecolors='black', label='Pixel')
plt.axis('equal')
plt.show()
# Camera parameters
fov = 90
image_width = 1280
image_height = 720

# Step 1: Process frames into local cone positions
coordinates_list = []
processed_list = []
frames = []
for frame_idx, group in df.groupby("frame"):
    coordinates_list = group[["pixel_x", "pixel_y"]].values
    depth_list = group["depth_mm"].values

    processed_list = one_frame_cone_positions(coordinates_list, depth_list, fov, image_width, image_height)
    

    frames.append(processed_list)

plt.scatter(processed_list[:,0], processed_list[:,1], c='yellow', edgecolors='black', label='Pixel')
plt.axis('equal')
plt.show()

matchPoints(processed_list, processed_list)

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