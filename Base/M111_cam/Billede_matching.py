import cv2
import numpy as np
import matplotlib.pyplot as plt

#funktion til at sikre at begge billeder er samme orientering
def ensure_landscape(img):
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

#indlæs billeder
# img1 = cv2.imread("frames/frame_1764928187431.png")#reference Helios
# img2 = cv2.imread("captured_frames/frame_000.png") #skal transformeres Webcam

img1 = cv2.imread("/home/aaujetson67/Documents/P3-141_AAU-Racing_Automatisk-sansning-af-omgivelserne/Base/heatmap.jpg")  # Reference image (Helios)
img2 = cv2.imread("/home/aaujetson67/Documents/P3-141_AAU-Racing_Automatisk-sansning-af-omgivelserne/Base/captured_frames/Webcam_screenshot_10.12.2025.png")  # Image to align (Webcam)


#sørg for begge billeder er samme orientering
img1 = ensure_landscape(img1)
img2 = ensure_landscape(img2)

points_img1 = []
points_img2 = []
max_points = 12  # antal punkter

# --- Mouse callback funktion ---
def select_points(event, x, y, flags, param):
    img_name, points = param
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < max_points:
        points.append([x, y])
        cv2.circle(img_name, (x, y), 5, (0,0,255), -1)
        cv2.imshow("Select points", img_name)

# --- Vælg punkter på billede 1 ---
img1_copy = img1.copy()
cv2.namedWindow("Select points", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Select points", cv2.WND_PROP_FULLSCREEN, 1)  # Sætter vinduet til fuldskærm
cv2.imshow("Select points", img1_copy)
cv2.setMouseCallback("Select points", select_points, param=(img1_copy, points_img1))
print(f"Klik på {max_points} punkter i Billede 1")
cv2.waitKey(0)
cv2.destroyAllWindows()

#gem billede med de valgte punkter
for point in points_img1:
    cv2.circle(img1, tuple(point), 5, (0, 0, 255), -1)
cv2.imwrite("img1WithPoints.png", img1)

# --- Vælg punkter på billede 2 ---
img2_copy = img2.copy()
cv2.namedWindow("Select points", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Select points", cv2.WND_PROP_FULLSCREEN, 1)  # Sætter vinduet til fuldskærm
cv2.imshow("Select points", img2_copy)
cv2.setMouseCallback("Select points", select_points, param=(img2_copy, points_img2))
print(f"Klik på {max_points} tilsvarende punkter i Billede 2")
cv2.waitKey(0)
cv2.destroyAllWindows()

#gem billede med de valgte punkter
for point in points_img2:
    cv2.circle(img2, tuple(point), 5, (0, 0, 255), -1)
cv2.imwrite("img2WithPoints.png", img2)

# --- Konverter til numpy arrays ---
pts1 = np.array(points_img1, dtype=np.float32)
pts2 = np.array(points_img2, dtype=np.float32)

#cv2.homography finder projective transform
H, status = cv2.findHomography(pts2, pts1, method=cv2.RANSAC)
print("Transformation matrix H:\n", H)

#warp billede2 på billede1
height, width = img1.shape[:2]
warped_img2 = cv2.warpPerspective(img2, H, (width, height))
cv2.imwrite("WarpedImg2.png", warped_img2)

blended = cv2.addWeighted(img1, 0.5, warped_img2, 0.5, 0)

#vis resultatet
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Billede 1")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(warped_img2, cv2.COLOR_BGR2RGB))
plt.title("Warped billede 2")
plt.show()

#gem transformationsmatricen og gem det blendede billede
cv2.imwrite("blended_result.png", blended)
np.save('homography_matrix.txt', H)