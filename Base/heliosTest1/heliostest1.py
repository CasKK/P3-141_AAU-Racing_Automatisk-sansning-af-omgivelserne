from arena_api.system import system
import numpy as np
import cv2

# Connect to camera
devices = system.create_device()
device = devices[0]
print(f"Connected to {device}")

# Start stream
print("Starting stream...")
device.start_stream()
print("Streaming started. Press ESC to quit.")

width = 640
height = 480
channels = 3  # A,B,C,Y

try:
    while True:
        buffer = device.get_buffer()
        data = buffer.data

        # --- Convert to bytes if needed ---
        if isinstance(data, list):
            data = bytes(data)
        elif not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Unexpected buffer data type: {type(data)}")

        arr = np.frombuffer(data, dtype=np.int16)

        expected_size = width * height * channels
        if arr.size > expected_size:
            arr = arr[:expected_size]

        frame = arr.reshape((height, width, channels))

        # Extract depth (Z) channel
        z_map = frame[:, :, 2]

        # Convert for display (normalize depth)
        z_display = cv2.convertScaleAbs(z_map, alpha=0.02)
        z_display = cv2.applyColorMap(z_display, cv2.COLORMAP_JET)

        # Show frame
        cv2.imshow("Helios Depth Stream", z_display)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

        device.requeue_buffer(buffer)

finally:
    device.stop_stream()
    system.destroy_device()
    cv2.destroyAllWindows()
    print("Stream stopped and camera released.")
