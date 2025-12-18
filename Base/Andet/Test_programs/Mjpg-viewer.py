
# mjpg_frame_viewer.py
# View a Motion-JPEG (MJPG) video one frame at a time.
# Keys: 'n' next, 'p' previous, 'r' restart, 's' save frame, 'q' quit.

import cv2
import sys
import os

def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def main():
    # Choose path: from argv if provided, otherwise hardcoded fallback
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = r"C:\Users\jacob\Desktop\videoer\output_1765470849.633834.avi"

    try:
        cap = open_video(path)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    # Read first frame
    frame = read_frame(cap)
    if frame is None:
        print("No frames found or unable to read from file.")
        sys.exit(1)

    cv2.namedWindow("MJPG Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MJPG Viewer", 960, 540)

    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # may be 0 for some containers
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_dir = "saved_frames"
    os.makedirs(save_dir, exist_ok=True)

    MAX_FRAME = 220  # limit

    while True:
        overlay = frame.copy()
        info = f"Frame {frame_idx+1}/{total if total>0 else '?'} | Limit: {MAX_FRAME}"
        cv2.putText(overlay, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("MJPG Viewer", overlay)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            if frame_idx + 1 >= MAX_FRAME:
                print(f"Reached frame limit ({MAX_FRAME}).")
                continue
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video.")
                break
            frame_idx += 1
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            frame_idx = 0
            if not ret:
                        print("Unable to restart video.")

            elif key == ord('s'):
                # Save current frame as PNG
                name = os.path.join(save_dir, f"frame_{frame_idx+1:06d}.png")
                cv2.imwrite(name, frame)
                print(f"Saved {name}")

            else:
                # Ignore other keys
                pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()