import cv2
import numpy as np

# Load the reference image and the image to align
img2 = cv2.imread("/home/aaujetson67/Documents/P3-141_AAU-Racing_Automatisk-sansning-af-omgivelserne/Base/heatmap.jpg")  # Reference image (Helios)
img1 = cv2.imread("/home/aaujetson67/Documents/P3-141_AAU-Racing_Automatisk-sansning-af-omgivelserne/Base/Webcam_screenshot_10.12.2025.png")  # Image to align (Webcam)

# Resize img1 to match img2 if necessary (optional)
if img1.shape != img2.shape:
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

# Convert to grayscale for keypoint detection (ORB works in grayscale)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Match descriptors using Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract the matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute homography matrix
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Warp the perspective of img1 to align it with img2
aligned = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

# Optionally, you can display the matches for debugging purposes:
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matches", img_matches)

# Show the aligned image
cv2.imshow("Aligned Image", aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()
