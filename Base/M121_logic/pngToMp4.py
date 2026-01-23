import cv2
import os

# ===================== CONFIGURATION =====================
image_folder = "M121_logic\plot"        # Folder containing PNG images
output_video = "output.mp4"    # Output video file
fps = 30.15                       # Frames per second
video_codec = "mp4v"           # Codec for MP4 (use 'XVID' for AVI)
# =========================================================

# Get sorted list of PNG files
images = sorted([
    img for img in os.listdir(image_folder)
    if img.lower().endswith(".png")
])

if not images:
    raise ValueError("No PNG images found in the specified folder.")

# Read first image to get frame size
first_frame_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_frame_path)

if frame is None:
    raise ValueError(f"Failed to read image: {first_frame_path}")

height, width, channels = frame.shape

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*video_codec)
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write frames
for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Warning: Skipping unreadable image {image_name}")
        continue

    # Ensure frame size consistency
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height))

    video.write(frame)

# Release resources
video.release()
cv2.destroyAllWindows()

print(f"Video successfully created: {output_video}")
