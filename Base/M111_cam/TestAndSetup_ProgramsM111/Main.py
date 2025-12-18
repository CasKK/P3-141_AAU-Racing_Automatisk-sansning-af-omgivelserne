from arena_api.system import system
import numpy as np
import cv2

def set_z_only_pixel_format(device):
    """Force Z-only pixel format: prefer Coord3D_Z16, else Coord3D_Z32f."""
    pf_node = device.nodemap.get_node("PixelFormat")
    if pf_node is None:
        raise RuntimeError("PixelFormat node not found on device")

    # Get available symbolic options (SDK may expose in different ways)
    try:
        available = set(pf_node.symbolics)
    except Exception:
        available = set(getattr(pf_node, "get_symbolics", lambda: [])())

    for cand in ("Coord3D_Z16", "Coord3D_Z32f"):
        if cand in available:
            if pf_node.value != cand:
                pf_node.value = cand
            return pf_node.value

    # If neither Z16 nor Z32f is available, keep current and warn
    print("[WARN] Z-only formats not available. Current PixelFormat:", pf_node.value)
    return pf_node.value

def disable_chunks_if_present(device):
    """Best effort to disable chunk data to keep payload size predictable."""
    try:
        chunk_mode = device.nodemap.get_node("ChunkModeActive")
        if chunk_mode and getattr(chunk_mode, "is_writable", False):
            chunk_mode.value = False

        chunk_selector = device.nodemap.get_node("ChunkSelector")
        chunk_enable = device.nodemap.get_node("ChunkEnable")
        if chunk_selector and chunk_enable:
            try:
                for sym in chunk_selector.symbolics:
                    chunk_selector.value = sym
                    if getattr(chunk_enable, "is_writable", False):
                        chunk_enable.value = False
            except Exception:
                pass
    except Exception:
        pass

# --- Connect to the Helios 1 camera ---
devices = system.create_device()
if not devices:
    raise Exception("No Helios camera found. Make sure it’s connected and not open in ArenaView.")

device = devices[0]
model_name = device.nodemap.get_node("DeviceModelName").value
serial = device.nodemap.get_node("DeviceSerialNumber").value
print(f"Connected to {model_name} (S/N: {serial})")

# Force Z-only pixel format and disable chunks before streaming
disable_chunks_if_present(device)
selected_pf = set_z_only_pixel_format(device)

# Read width/height from device nodemap (after setting PF)
width = device.nodemap.get_node("Width").value
height = device.nodemap.get_node("Height").value
pix_fmt = device.nodemap.get_node("PixelFormat").value
print(f"Frame size: {width}x{height}, PixelFormat: {pix_fmt}")

# --- Start streaming frames ---
print("Starting stream...")
stream = device.start_stream()
if stream is None:
    raise RuntimeError("Failed to start stream. Try power-cycling the camera or closing ArenaView.")

try:
    print("Streaming started. Press ESC to quit.")
    win_name = "Helios 1 Depth Stream"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        buffer = device.get_buffer()  # acquire one frame
        try:
            # Zero-copy view of payload
            payload = memoryview(buffer.data)

            # Interpret buffer as Z-only image depending on PixelFormat
            if "Coord3D_Z16" in pix_fmt:
                # Z in millimeters as uint16
                z = np.frombuffer(payload, dtype=np.uint16).reshape((height, width))

                # Convert to 8-bit for display (adjust alpha to your range)
                img_8u = cv2.convertScaleAbs(z, alpha=0.02)

            elif "Coord3D_Z32f" in pix_fmt:
                # Z as float32 (units may be meters or mm depending on camera config)
                z = np.frombuffer(payload, dtype=np.float32).reshape((height, width))

                # For visualization, normalize using robust percentiles
                finite = np.isfinite(z)
                if np.any(finite):
                    z_low = np.percentile(z[finite], 2)
                    z_high = np.percentile(z[finite], 98)
                    norm = np.clip((z - z_low) / max(z_high - z_low, 1e-6), 0, 1)
                else:
                    norm = np.zeros_like(z, dtype=np.float32)
                img_8u = (norm * 255).astype(np.uint8)

            else:
                # If Z-only not set (unexpected), inform and skip
                print(f"[WARN] Unexpected PixelFormat for Z-only path: {pix_fmt}")
                img_8u = None

            if img_8u is not None:
                color_map = cv2.applyColorMap(img_8u, cv2.COLORMAP_JET)
                cv2.imshow(win_name, color_map)

            if cv2.waitKey(1) == 27:  # ESC to quit
                break

        finally:
            # Requeue for next capture no matter what
            device.requeue_buffer(buffer)

finally:
    device.stop_stream()
    system.destroy_device()
    cv2.destroyAllWindows()
    print("Stream stopped and camera released.")