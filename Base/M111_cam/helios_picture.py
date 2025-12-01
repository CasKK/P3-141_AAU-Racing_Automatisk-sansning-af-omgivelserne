#!/usr/bin/env python3
"""
Live Helios/Helios2 depth heatmap with robust colormap scaling (Arena SDK, Jetson).
- Supports pixel formats: Coord3D_C16 (preferred), Coord3D_C16Y8 (fallback).
- Scaling modes:
    * fixed:    --near/--far in depth units (mm typical)
    * robust:   per-frame robust [pmin,pmax] with EMA smoothing (default)
    * global:   learn range on first N frames then freeze
    * percentile: custom pmin/pmax per frame
  + optional --gamma and --invert, colorbar, and FPS overlay.
"""
import argparse, time, signal, sys
import numpy as np, cv2
from arena_api.system import system
from arena_api.enums import PixelFormat

# ---------- CLI ----------
def cli():
    p = argparse.ArgumentParser("Helios Heatmap (scaled)")
    p.add_argument("--scale", choices=["fixed","robust","global","percentile"], default="robust",
                   help="Scaling strategy for mapping depth->colors")
    p.add_argument("--near", type=int, default=None, help="Near clamp (mm). Used if --scale=fixed")
    p.add_argument("--far",  type=int, default=None, help="Far clamp (mm). Used if --scale=fixed")
    p.add_argument("--pmin", type=float, default=1.0, help="Lower percentile for robust/percentile")
    p.add_argument("--pmax", type=float, default=99.0, help="Upper percentile for robust/percentile")
    p.add_argument("--ema",  type=float, default=0.2, help="EMA smoothing factor for robust range")
    p.add_argument("--warmup", type=int, default=30, help="Frames to learn global min/max")
    p.add_argument("--gamma", type=float, default=1.0, help="Gamma for non-linear contrast shaping")
    p.add_argument("--invert", action="store_true", help="Invert gradient (near->far flips colors)")
    p.add_argument("--colormap", default="turbo",
                   choices=["turbo","inferno","magma","plasma","viridis","jet"], help="OpenCV colormap")
    p.add_argument("--fps", type=float, default=None, help="Request acquisition FPS")
    p.add_argument("--buffers", type=int, default=8, help="Stream buffers")
    p.add_argument("--win", default="Helios Heatmap", help="Window title")
    p.add_argument("--show_fps", action="store_true", help="Draw FPS")
    p.add_argument("--colorbar", action="store_true", help="Draw colorbar")
    return p.parse_args()

COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
}

stop_flag = False
def _sigint(_s,_f):  # graceful Ctrl+C
    global stop_flag; stop_flag = True

# ---------- Scaling helpers ----------
def compute_range(depth_u16, mode, pmin, pmax, ema_state, warm_state):
    """
    Returns (dmin,dmax,ema_state,warm_state) for chosen mode.
    - depth zeros are ignored.
    """
    nz = depth_u16[depth_u16 > 0]
    if nz.size < 50:
        lo, hi = 0, int(depth_u16.max())
        return lo, max(hi, lo+1), ema_state, warm_state
    if mode == "percentile":
        lo = np.percentile(nz, pmin); hi = np.percentile(nz, pmax)
        if hi <= lo: hi = nz.max()
        return int(lo), int(hi), ema_state, warm_state

    if mode == "robust":
        lo = np.percentile(nz, pmin); hi = np.percentile(nz, pmax)
        if hi <= lo: hi = nz.max()
        if ema_state is None:
            ema_state = [int(lo), int(hi)]
        else:
            ema_state[0] = int((1 - args.ema)*ema_state[0] + args.ema*lo)
            ema_state[1] = int((1 - args.ema)*ema_state[1] + args.ema*hi)
        return ema_state[0], ema_state[1], ema_state, warm_state

    if mode == "global":
        # Accumulate global min/max during warmup, then freeze
        if not warm_state["frozen"]:
            warm_state["n"] += 1
            warm_state["lo"] = min(warm_state["lo"], int(np.percentile(nz, pmin)))
            warm_state["hi"] = max(warm_state["hi"], int(np.percentile(nz, pmax)))
            if warm_state["n"] >= warm_state["warmup"]:
                warm_state["frozen"] = True
        lo = warm_state["lo"]; hi = warm_state["hi"]
        return lo, max(hi, lo+1), ema_state, warm_state

    raise ValueError("Invalid mode")

def apply_colormap(depth_u16, dmin, dmax, cmap_id, gamma=1.0, invert=False):
    # Clip & normalize to 0..1 float
    d = depth_u16.astype(np.float32)
    d = np.clip(d, dmin, dmax)
    if dmax <= dmin: dmax = dmin + 1.0
    norm = (d - dmin) / (dmax - dmin)
    if invert: norm = 1.0 - norm
    if gamma != 1.0:
        norm = np.power(norm, 1.0 / max(1e-6, gamma))
    u8 = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(u8, cmap_id)
    color[depth_u16 == 0] = (0,0,0)  # invalid to black
    return color

def draw_colorbar(img, dmin, dmax, cmap_id):
    h, w = img.shape[:2]
    bar_h, bar_w = 20, min(300, w//3)
    bar = np.linspace(0, 255, bar_w, dtype=np.uint8)
    bar = np.repeat(bar[None,:], bar_h, axis=0)
    bar = cv2.applyColorMap(bar, cmap_id)
    # put text
    cv2.rectangle(img, (10, h-10-bar_h-30), (10+bar_w, h-10), (0,0,0), -1)
    img[h-10-bar_h:h-10, 10:10+bar_w] = bar
    cv2.putText(img, f"{dmin}", (10, h-12-bar_h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    txt = f"{dmax}"
    (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, txt, (10+bar_w-tw, h-12-bar_h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1,cv2.LINE_AA)

# ---------- Main ----------
def get_depth_from_buffer(buf, depth_format):
    """Return depth_u16 ndarray (H,W) for Coord3D_C16 or Coord3D_C16Y8."""
    h, w, bpp = buf.height, buf.width, buf.bits_per_pixel
    byte_count = h*w*(bpp//8)
    arr = np.ctypeslib.as_array(buf.pdata, shape=(byte_count,))
    if depth_format == "Coord3D_C16":
        depth = arr.view(np.uint16).reshape(h, w)    # 2 bytes per pixel
    else:
        # layout per pixel: [Zlo,Zhi, I] → take Z 16-bit little-endian
        px = arr.reshape(h, w, 3)
        depth = (px[:,:,1].astype(np.uint16) << 8) | px[:,:,0].astype(np.uint16)
    return depth

if __name__ == "__main__":
    args = cli()
    signal.signal(signal.SIGINT, _sigint)

    # Find Helios
    devices = []
    for _ in range(30):
        devices = system.create_device()
        if devices: break
        print("Waiting for Helios device...")
        time.sleep(1)
    if not devices:
        print("No LUCID device found. Connect Helios and retry."); sys.exit(1)
    dev = devices[0]
    # Maximize packet size if available (GigE throughput)
    try:
        ps = dev.nodemap["DeviceStreamChannelPacketSize"]; ps.value = ps.max
    except Exception: pass

    # Request FPS (optional)
    if args.fps is not None:
        try:
            dev.nodemap["AcquisitionFrameRateEnable"].value = True
            afr = dev.nodemap["AcquisitionFrameRate"]
            afr.value = max(min(args.fps, afr.max), afr.min)
        except Exception as ex:
            print(f"FPS not applied: {ex}")

    # Depth pixel format
    pf = dev.nodemap["PixelFormat"]
    depth_fmt = None
    try:
        pf.value = PixelFormat.Coord3D_C16; depth_fmt = "Coord3D_C16"
    except Exception:
        pf.value = PixelFormat.Coord3D_C16Y8; depth_fmt = "Coord3D_C16Y8"

    cmap_id = COLORMAPS[args.colormap]
    cv2.namedWindow(args.win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(args.win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ema_state = None
    warm_state = {"warmup": args.warmup, "n": 0, "lo": 1<<30, "hi": 0, "frozen": False}

    # If fixed scaling, validate inputs
    if args.scale == "fixed":
        if args.near is None or args.far is None or args.far <= args.near:
            print("For --scale fixed, provide valid --near <mm> and --far <mm> (far>near)."); sys.exit(2)

    # Stream loop
    t_last, frames, fps_smoothed = time.time(), 0, None
    try:
        with dev.start_stream(args.buffers):
            while not stop_flag:
                buf = dev.get_buffer()
                depth = get_depth_from_buffer(buf, depth_fmt)
                dev.requeue_buffer(buf)

                if args.scale == "fixed":
                    dmin, dmax = int(args.near), int(args.far)
                else:
                    dmin, dmax, ema_state, warm_state = compute_range(
                        depth, args.scale, args.pmin, args.pmax, ema_state, warm_state)

                img = apply_colormap(depth, dmin, dmax, cmap_id, gamma=args.gamma, invert=args.invert)

                if args.show_fps:
                    frames += 1
                    now = time.time()
                    if now - t_last >= 0.5:
                        fps = frames / (now - t_last)
                        fps_smoothed = fps if fps_smoothed is None else 0.9*fps_smoothed + 0.1*fps
                        frames, t_last = 0, now
                    if fps_smoothed is not None:
                        cv2.putText(img, f"{fps_smoothed:5.1f} FPS", (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)

                if args.colorbar: draw_colorbar(img, dmin, dmax, cmap_id)

                cv2.imshow(args.win, img)
                key = cv2.waitKey(1) & 0xFF
                
                if key in (27, ord('q')): #ESC eller q afslutter
                    break
                elif key == 32: #mellemrum gemmer billede
                    import os
                    os.makedirs("frames", exist_ok=True) #sørg for mappen findes
                    filename = f"frames/frame_{int(time.time()*1000)}.png" #unikt navn
                    cv2.imwrite(filename, img)
                    print(f"Billede gemt; {filename}")
    finally:
        try: system.destroy_device()
        except Exception: pass
        cv2.destroyAllWindows()