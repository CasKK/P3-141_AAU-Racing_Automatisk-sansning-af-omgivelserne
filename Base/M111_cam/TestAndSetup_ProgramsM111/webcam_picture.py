import cv2
import os
import re

# Mappe til at gemme billeder
output_picture = "captured_frames"
os.makedirs(output_picture, exist_ok=True)

frame_count = 0

# Åbn kamera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cap.isOpened():
    print("Kan ikke åbne kamera /dev/video0")
    exit(1)

print("Tryk 's' for at gemme et billede, 'q' for at lukke")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fejl: kunne ikke læse frame")
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Gem billedet
        filename = os.path.join(output_picture, f"frame_{frame_count:03d}.png")
        cv2.imwrite(filename, frame)
        print(f"Billede gemt: {filename}")
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
