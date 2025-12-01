#!/usr/bin/env python3
import argparse
import signal
import sys
import time
import numpy as np
import cv2

# Arena SDK imports
from arena_api.system import system
from arena_api.enums import PixelFormat

def parse_args():
    p = argparse.ArgumentParser(
        description="Live depth heatmap from LUCID Helios/Helios2 (Arena SDK, Jetson)")
    p.add_argument("--near", type=int, default=None,
                   help="Near clamp in depth units (typically mm). If unset, use per-frame robust normalization.")
    p.add_argument("--far", type=int, default=None,
                   help="Far clamp in depth units (typically mm). If unset, use per-frame robust normalization.")
    p.add_argument("--colormap", default="inferno",
                   choices=["inferno","turbo","jet","magma","plasma","viridis"],
                   help="OpenCV colormap for heatmap.")
    p.add_argument("--win", default="Helios Heatmap", help="Window title")
    p.add_argument("--fps", type=float, default=None,
                   help="Desired acquisition FPS (optional).")
    p.add_argument("--buffers", type=int, default=8,
                   help="Number of stream buffers")
    p.add_argument("--show_fps", action="store_true", help="Overlay FPS in window")
    return p.parse_args()

COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    # OpenCV >= 4.5
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
}

stop_flag = False
def _sigint(_sig, _frm):
    global stop_flag
    stop_flag = True

def robust_minmax(depth_u16):
    """Compute robust per-frame min/max (1–99th percentile) ignoring zeros."""
    nz = depth_u16[depth_u16 > 0]
    if nz.size < 100:  # fallback
        return 0, int(depth_u16.max())
    lo = np.percentile(nz, 1.0)
    hi = np.percentile(nz, 99.0)
    if hi <= lo:
        hi = nz.max()
        lo = nz.min()
    return int(lo), int(hi)

def to_heatmap(depth_u16, dmin, dmax, cmap_id):
    # Clip & normalize to 8-bit
    depth_clipped = np.clip(depth_u16, dmin, dmax).astype(np.float32)
    if dmax == dmin:
        dmax = dmin + 1.0
    norm = ((depth_clipped - dmin) * (255.0 / (dmax - dmin))).astype(np.uint8)
    color = cv2.applyColorMap(norm, cmap_id)
    # Make invalid pixels (zeros) black
    mask_invalid = (depth_u16 == 0)
    color[mask_invalid] = (0, 0, 0)
    return color

def main():
    args = parse_args()
    signal.signal(signal.SIGINT, _sigint)

    # Acquire a device
    tries, tries_max, sleep_sec = 0, 30, 1
    devices = []
    while tries < tries_max:
        devices = system.create_device()
        if devices:
            break
        tries += 1
        print(f"Waiting for Helios device... ({tries}/{tries_max})")
        time.sleep(sleep_sec)
    if not devices:
        print("No LUCID device found. Connect Helios and try again.")
        sys.exit(1)

    device = devices[0]
    model = device.nodemap.get_node("DeviceModelName").value
    if "Helios" not in model:
        print(f"Warning: connected device is '{model}', not Helios/Helios2.")

    # Set packet size up to max for performance (GigE)
    try:
        pkt = device.nodemap["DeviceStreamChannelPacketSize"]
        pkt.value = pkt.max
    except Exception:
        pass  # not all firmwares expose this

    # Optionally set FPS
    if args.fps is not None:
        try:
            device.nodemap["AcquisitionFrameRateEnable"].value = True
            afr = device.nodemap["AcquisitionFrameRate"]
            args.fps = max(min(args.fps, afr.max), afr.min)
            afr.value = args.fps
        except Exception as ex:
            print(f"FPS control not applied: {ex}")

    # Choose depth pixel format; prefer Coord3D_C16, fallback to Coord3D_C16Y8
    pf_node = device.nodemap["PixelFormat"]
    depth_format = None
    try:
        pf_node.value = PixelFormat.Coord3D_C16
        depth_format = "Coord3D_C16"
    except Exception:
        try:
            pf_node.value = PixelFormat.Coord3D_C16Y8  # depth + 8b intensity
            depth_format = "Coord3D_C16Y8"
        except Exception as ex:
            print("ERROR: Camera does not support Coord3D_C16 or Coord3D_C16Y8. "
                  "Check model/firmware or use ArenaView to verify pixel formats.")
            raise ex

    # Start streaming
    cmap = COLORMAPS[args.colormap]
    cv2.namedWindow(args.win, cv2.WINDOW_NORMAL)
    ema_min, ema_max = None, None
    alpha = 0.2  # smoothing for min/max
    tick_last = time.time()
    frame_count = 0
    fps_smoothed = None

    try:
        with device.start_stream(args.buffers):
            while not stop_flag:
                buf = device.get_buffer()  # blocking

                # Interpret buffer as 16-bit depth (Z plane)
                h, w = buf.height, buf.width
                bpp = buf.bits_per_pixel
                # Coord3D_C16 and Coord3D_C16Y8 both contain a 16b Z plane
                # Buffer layout is contiguous; use ctypes pointer directly.
                arr = np.ctypeslib.as_array(buf.pdata, shape=(h * w * (bpp // 8),))

                if depth_format == "Coord3D_C16":
                    depth = arr.view(np.uint16).reshape(h, w)
                else:  # Coord3D_C16Y8 → 2 bytes depth + 1 byte intensity per pixel (packed)
                    # layout: [Zlo, Zhi, I] repeating — we only take Z
                    z = arr.reshape(h, w, 3)[:, :, :2].copy()
                    depth = (z[:, :, 1].astype(np.uint16) << 8) | z[:, :, 0].astype(np.uint16)

                device.requeue_buffer(buf)

                # Determine display range
                if args.near is not None and args.far is not None:
                    dmin, dmax = int(args.near), int(args.far)
                else:
                    lo, hi = robust_minmax(depth)
                    if ema_min is None:
                        ema_min, ema_max = lo, hi
                    else:
                        ema_min = int((1 - alpha) * ema_min + alpha * lo)
                        ema_max = int((1 - alpha) * ema_max + alpha * hi)
                    dmin, dmax = ema_min, ema_max

                heat = to_heatmap(depth, dmin, dmax, cmap)

                # FPS overlay
                if args.show_fps:
                    frame_count += 1
                    now = time.time()
                    dt = now - tick_last
                    if dt >= 0.5:
                        fps = frame_count / dt
                        fps_smoothed = (fps if fps_smoothed is None
                                        else 0.9 * fps_smoothed + 0.1 * fps)
                        tick_last = now
                        frame_count = 0
                    if fps_smoothed is not None:
                        cv2.putText(heat, f"{fps_smoothed:5.1f} FPS",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(args.win, heat)
                # Non-blocking key check
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC or q
                    break

    finally:
        # Cleanup
        try:
            system.destroy_device()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()