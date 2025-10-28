from arena_api.system import system
import numpy as np
import cv2

# --- Connect to the Helios 1 camera ---
devices = system.create_device()
if not devices:
    raise Exception("No Helios camera found. Make sure it’s connected and not open in ArenaView.")

device = devices[0]
model_name = device.nodemap.get_node("DeviceModelName").value
serial = device.nodemap.get_node("DeviceSerialNumber").value
print(f"Connected to {model_name} (S/N: {serial})")

# --- Start streaming frames ---
print("Starting stream...")
stream = device.start_stream()
if stream is None:
    raise RuntimeError("Failed to start stream. Try power-cycling the camera or closing ArenaView.")

try:
    print("Streaming started. Press ESC to quit.")

    # Read width/height from device nodemap
    width = device.nodemap.get_node("Width").value
    height = device.nodemap.get_node("Height").value
    pix_fmt = device.nodemap.get_node("PixelFormat").value
    print(f"Frame size: {width}x{height}, PixelFormat: {pix_fmt}")


    while True:
        buffer = device.get_buffer()  # acquire one frame
        data = np.copy(buffer.data)   # 1D uint8 array

        # Requeue for next capture
        device.requeue_buffer(buffer)

        # Interpret buffer as uint16 depth image
        image = np.frombuffer(data, dtype=np.uint16)
        image = image.reshape((height, width))

        # Convert 16-bit depth to 8-bit for display
        img_8u = cv2.convertScaleAbs(image, alpha=0.02)

        # Optional: colorize for easier viewing
        color_map = cv2.applyColorMap(img_8u, cv2.COLORMAP_JET)

        # Show the colorized image
        cv2.imshow("Helios 1 Depth Stream", color_map)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break
finally:
    device.stop_stream()
    system.destroy_device()
    cv2.destroyAllWindows()
    print("Stream stopped and camera released.")
