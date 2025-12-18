import cv2

img = cv2.imread(fr"C:\Users\jacob\Downloads\thumbnail_IMG_2177.jpg", 0)
img = cv2.Canny(img, 100, 200)

cv2.imwrite("car_canny.png", img)
cv2.imshow("img", img)
cv2.waitKey()