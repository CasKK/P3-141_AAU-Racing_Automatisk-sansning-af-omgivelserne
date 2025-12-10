import cv2
import os
import math

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # Solidity
    hull_points = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull_points)
    solidity = area / hull_area if hull_area != 0 else 0

    # Circularity
    perimeter = cv2.arcLength(c, True)
    circularity = perimeter / (2 * math.sqrt(math.pi * area)) if area > 0 else 0

    # Aspect ratio
    rect = cv2.minAreaRect(c)
    (w_minArea, h_minArea) = rect[1]
    if h_minArea > w_minArea:
        aspect = h_minArea / w_minArea if w_minArea != 0 else 0
    else:
        aspect = w_minArea / h_minArea if h_minArea != 0 else 0

    # Compactness
    compactness = area / (h_minArea * w_minArea) if (h_minArea * w_minArea) != 0 else 0

    # Convexity Defects
    hull = cv2.convexHull(c, returnPoints=False)
    defects = cv2.convexityDefects(c, hull)
    num_defects = defects.shape[0] if defects is not None else 0

    return [solidity, circularity, aspect, compactness, num_defects]

def process_folder(folder_path, class_name, output_file="training_data.txt"):
    features_list = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg")):
            image_path = os.path.join(folder_path, filename)
            features = extract_features(image_path)
            if features:
                features_list.append(features)

    # Gem i txt i Python‑venlig form
    with open(output_file, "a") as f:  # append så du kan køre for flere klasser
        f.write(f"{class_name} = {features_list}\n\n")

# Eksempel: kør for to mapper
process_folder("blobsTop", "classTop", "training_data.txt")
process_folder("blobsBottom", "classBottom", "training_data.txt")
process_folder("blobsMix1", "classMix1", "training_data.txt")
process_folder("blobsMix2", "classMix2", "training_data.txt")


