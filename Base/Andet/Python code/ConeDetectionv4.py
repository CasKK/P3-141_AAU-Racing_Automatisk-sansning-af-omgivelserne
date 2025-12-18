
import numpy as np
import cv2
import time
import subprocess
import math
import glob
from py_helios_smooth_result import CreateDevice, HeliosRunning, HeliosEnd, set_nodes
import threading
kernel = np.ones([5,5], np.uint8)



camera_lock = threading.Lock()
# The setup function is where the camera is inititiated and the limits for color segmentation is defined in HSV
def Setup():
    def getCameraControl(device, name):
        result = subprocess.run(
            ["v4l2-ctl", f"--device={device}", "--get-ctrl", name],
            capture_output=True, text=True
        )
        return result.stdout.strip()

    def setCameraControl(device, control, value):
        cmd = ["v4l2-ctl", f"--device={device}", "--set-ctrl", f"{control}={value}"]
        subprocess.run(cmd, check=True)


    videoDevices = sorted(glob.glob("/dev/video*"))
    cam = None
    selectedDevice = None

    for dev in videoDevices:
        index = int(dev.replace("/dev/video", ""))
        print(f"Prøver {dev} ...")
        
        with camera_lock:
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if cap.isOpened():
            print(f"Leder efter kamera på: {dev}")
            cam = cap
            selectedDevice = dev
            break
        else:
            cap.release()

    if not cam:
        print("Ingen kamera fundet")
        exit(1)

    print("bruger kamera:", selectedDevice)
    
    # Læs gain i auto mode
    auto_gain = getCameraControl(selectedDevice, "gain")

    #manually setting the camera setings
    setCameraControl(selectedDevice, "auto_exposure",1)
    setCameraControl(selectedDevice, "exposure_time_absolute", 200)
    setCameraControl(selectedDevice, "gain", 10)#250)
    setCameraControl(selectedDevice, "focus_automatic_continuous",0)
    setCameraControl(selectedDevice, "focus_absolute", 0)
    setCameraControl(selectedDevice, "white_balance_automatic", 1)

    upperLimitBlue = np.array([123, 252, 150], np.uint8)
    lowerLimitBlue = np.array([103, 92, 5], np.uint8)
    upperLimitYellow = np.array([33,255,230], np.uint8)
    lowerLimitYellow = np.array([13,120,70], np.uint8)
    return cam, upperLimitBlue, lowerLimitBlue, upperLimitYellow, lowerLimitYellow



# The masking function is where all the preprocessing and masking is performed
# A gaussian blur is added to remove noise and smooth the frame
# The frame is then converted to HSV
# A mask is created using the limits from Setup() and then converted to a mask so only the colors in the limit is found
# Opening and closing morphology is performed on the masks to further remove noise
# Then adding a median blur to EVEN further decrease noise. This might not be necesarry 
def Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower):
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maskBlue = cv2.inRange(HSV, blueLower, blueUpper)
        maskYellow = cv2.inRange(HSV, yellowLower, yellowUpper)

        maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_OPEN, kernel, iterations=2)
        maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_CLOSE, kernel, iterations=2)
        maskBlue = cv2.medianBlur(maskBlue, 5)

        maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_OPEN, kernel, iterations=2)
        maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_CLOSE, kernel, iterations=2)
        maskYellow = cv2.medianBlur(maskYellow, 5)
        return HSV, maskBlue, maskYellow


# Find the contours from in the masks to find objects
# This is performed on each mask and separated so the colors of each contour is found
def FindContours(maskBlue, maskYellow):
    contoursBlue, _ = cv2.findContours(maskBlue,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxBlue = []
    contoursYellow, _ = cv2.findContours(maskYellow,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxYellow = []

    def FilterBbox(contours, bbox):
        #Gennemgår alle konturer fundet i et maskeret billede
        for c in contours:

            def CheckFeatures(c):

                # Check if the area is to small
                area = cv2.contourArea(c)
                if not area > 500:   # Er arealet mindre end 100 pixel
                    return False
                
                # Check solidity
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area != 0 else 0
                if not solidity > 0.8:
                    return False

                # Check circularity
                perimeter = cv2.arcLength(c, True)
                circularity = perimeter/(2*math.sqrt(math.pi*area))
                if not (1 < circularity < 50):
                    return False

                # Check aspect
                rect = cv2.minAreaRect(c)
                (w_minArea,h_minArea) = rect[1]
                if h_minArea > w_minArea:
                    aspect = h_minArea / w_minArea if w_minArea != 0 else 0
                else:
                    aspect = w_minArea / h_minArea if h_minArea != 0 else 0
                if not (0.7 < aspect < 3):
                    return False

                # Check compactness
                compactness = area / (h_minArea*w_minArea) if (h_minArea*w_minArea) != 0 else 0
                if not (0.3 < compactness < 0.9):
                    return False

                # Check if the bbox is to small
                x,y,w_bbox,h_bbox = cv2.boundingRect(c)
                if not (w_bbox > 10 and h_bbox > 10):
                    return False
                
                # Calculate a diagonal
                d = math.sqrt(h_minArea**2 + w_minArea**2)

                return (c, x, y, w_bbox, h_bbox, d)#[c]
            
            valid = CheckFeatures(c)
            if valid != False:
                bbox.append(valid)
                        
    FilterBbox(contoursBlue, bboxBlue)
    FilterBbox(contoursYellow, bboxYellow)
    # The Bboxes are then being merged in the MergeBbox() function to merge close bboxes
    
    return bboxBlue, bboxYellow




def EdgeDetection(HSV):

    def Morphology(combine):
        #morphology
        closing = cv2.morphologyEx(combine, cv2.MORPH_CLOSE, kernel, iterations=1) #closing small holes
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1) #removing small noise pixels outside the edges 

        #Converting the frames to binary pictures, making pixels >20 white and the rest black
        _, binary = cv2.threshold(opening, 20, 255, cv2.THRESH_BINARY)
        

        return binary
    
    def RemoveSmallObjectsWithContours(binary, min_area=300):
        #Finding all outer contours in the binary picture 
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Making an empty mask, the same size as the input picture 
        mask = np.zeros_like(binary)

        #Listing to save the large contours 
        keptContours = []

        #Gping through every contour 
        for c in contours:
            #Calculating the area of the contour 
            area = cv2.contourArea(c)
            #Only saving contourer, that is larger than min_area
            if area >= min_area:
                keptContours.append(c)

        #Draw all the saved contours as white objekts on the mask
        if keptContours:
                cv2.drawContours(mask, keptContours, -1, 255, thickness=cv2.FILLED)

        return mask, keptContours

    #splitting hsv pictures 
    v = HSV[:, :, 2]
    
    #using gaussian blur on v-channel
    frameBlur = cv2.GaussianBlur(v, (7,7), 0) 

    #making sobel edge detection with kernel (3,3) on v-channel from HSV
    sobelx = cv2.Sobel(frameBlur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(frameBlur, cv2.CV_64F, 0, 1, ksize=3)

    #calculating the gradient magnitude and thresholding it
    gradientMagnitude = cv2.magnitude(sobelx, sobely)
    _, gradientThresh = cv2.threshold(gradientMagnitude, 50, 255, cv2.THRESH_TOZERO)

    #converting gradient magnitude to uint8
    gradientMagnitude = cv2.convertScaleAbs(gradientThresh)

    #using blurred v-channel to make canny-edge detection
    edges = cv2.Canny(frameBlur, threshold1=50, threshold2=150)

    #combining sobel and canny edge detection
    combine = cv2.bitwise_or(gradientMagnitude, edges) 

    #Making binary picture with morphology
    binary = Morphology(combine)

    # Remove small objects fast with contour filter
    cleanedMask, keptContours = RemoveSmallObjectsWithContours(binary, min_area=250)

    #Finding contours in the binary edge picture and draw a frame 
        # edge_contours: listing with points for every contour
        # cv2.RETR_EXTERNAL: only outer contours, ingoring inner holes 
        # cv2.CHAIN_APPROX_SIMPLE: compress contourpoints along side a line to save memmory
    edgeContours, _ = cv2.findContours(combine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return combine, edgeContours, cleanedMask

def CompareWithHelios(contours, frame, depth):    
    newContours = []
    
    for contour in contours:
        c, x, y, w, h, d = contour
        mask = np.zeros_like(frame[:, :, 0])
        cv2.drawContours(mask, [c], -1, 255, -1)
        depthMask = depth[mask==1]
        hist, _ = np.histogram(depthMask, bins=120, range=(0,6000))

        peaks = []
        countSum, countNumber, countRange = 0, 0, 0

        for i, count in enumerate(hist):            
            
            minDist, maxDist = 0, 0

            if (count < 100):
                if countNumber > 0:
                    expan = 2
                    minDist = i - countRange - expan
                    maxDist = -1 + expan
                    peakMask = cv2.inRange(depth, minDist, maxDist)
                    mean = np.mean(depth[depthMask & (peakMask == 255)])
                    peaks.append((peakMask, mean, minDist, maxDist))
                    countSum, countNumber, countRange = 0, 0, 0
            else:
                countSum += i * count
                countNumber += count
                countRange += 1

        if len(peaks) == 1:
            _, mean, _, _ = peaks[0]
            newContours.append((c, x, y, w, h, d, mean))  # Append original contour and its mean
        elif len(peaks) > 1:
            for p in peaks:
                peakMask, mean, _, _ = p
                combinedMask = cv2.bitwise_and(depthMask, peakMask)
                cNew, _ = cv2.findContours(combinedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                xNew, yNew, wNew, hNew = cv2.boundingRect(cNew[0])
                dNew = math.sqrt(hNew**2 + wNew**2)
                newContours.append((cNew[0], xNew, yNew, wNew, hNew, dNew, mean))

    return newContours  

def VerifyConesWithEdges(bboxes, edges, threshold):
    verified = [] #Listing for the bboxes, that is being verified as valid 
    
    _, edgesBin = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)

    for box in bboxes:
        #Converting bounding box coordinates to a integer
        c, _, _, _, _, _ = box

        maskEdge = np.zeros_like(edgesBin)
        cv2.drawContours(maskEdge, [c], -1, (255), 4)
        
        overlap = cv2.bitwise_and(maskEdge, edgesBin)

        edgePixels = cv2.countNonZero(maskEdge)
        overlapPixels = cv2.countNonZero(overlap)

        if edgePixels > 0 and (overlapPixels / edgePixels >= threshold):
            verified.append(box)

    return verified

# The MergeBbox function is used to merge the base and top parts of the cones
# It has two helper functions CenterCalc(), where it calculates the centers of each bbox and its it to a list.
# MergeCenters then filters through these centers and checks if two centers are within the predefined pixel distance
# if they are a new center is calculated as the average of both ccenters, and a new bbox is defined.
def MergeBbox(bboxBlue, bboxYellow):
    def CenterCalc(bbox, centrum):
        for b in bbox:
            _,x,y,w,h,d = b
            cx = x+w/2
            cy = y+h/2
            centrum.append((cx,cy, w, h,x,y,d))
    def MergeCenters(centrum, merged, mergedCorner, mergedCenter):
        for c in centrum: # go through each item in the list of center locations
            found = False # Found is false at the start since the first center is not close to any of the others since there is none to compare to
            for i, m in enumerate(merged): # For each iteration i, and m in enumerate(merged) i is the iteration and m is the corresponding entry in merged
                if (abs(c[1]-m[1]) < c[6]*2) and (abs(c[0]-m[0]) < c[6]*0.5): # Checks to see if the center c is close enough to the entry m
                    merged[i] = ((c[0]+m[0])/2, (c[1]+m[1])/2, abs(c[1]-m[1])*1.5, abs(c[0]-m[0])*3) # If it is close enough a new center is calculated 
                    x1 = min(c[0]-c[2]/2, m[0]-m[2]/2) # The minimum and maximum values for the new center is created
                    y1 = min(c[1]-c[3]/2, m[1]-m[3]/2)
                    x2 = max(c[0]+c[2]/2, m[0]+m[2]/2)
                    y2 = max(c[1]+c[3]/2, m[1]+m[3]/2)
                    w_new = x2 - x1
                    h_new = y2 - y1 #This is so a new bbox can be created
                    mergedCorner[i] = (x1, y1, w_new, h_new) # Add the new bbox to the mergedCorner list, this is the upper left corner and is standard for  reating bboxes
                    found = True # set found to true
                    break # go out of the for loop so the next center will be assed 
            if not found: # If found != True it just adds the original positions
                merged.append((c[0],c[1],c[2],c[3])) #This is the center position
                mergedCorner.append((c[4],c[5],c[2],c[3])) #This is for the upper corner and correct bbox creation
                
                cx = c[4]+c[2]/2
                cy = c[5]+c[3]/2
                mergedCenter.append((cx,cy))

    centrumB, centrumY,  mergedBlue, mergedYellow, mergedBlueCorner, mergedYellowCorner, mergedCenterBlue, mergedCenterYellow = [], [], [], [], [], [], [], []
    CenterCalc(bboxBlue, centrumB)
    CenterCalc(bboxYellow, centrumY)
    MergeCenters(centrumB, mergedBlue, mergedBlueCorner, mergedCenterBlue)
    MergeCenters(centrumY, mergedYellow, mergedYellowCorner, mergedCenterYellow)
    
    return mergedBlueCorner, mergedYellowCorner, mergedBlue, mergedYellow, mergedCenterBlue, mergedCenterYellow

# This function draws the bounding boxes
def DrawBoundingBox(box, frame, color):
    x, y, w, h = box
    p1 = (int(x), int(y))
    p2 = (int(x+w), int(y+h))

    if  color == "Blue":
        label = "Blue Cone"
        frameColor = (255,0,0)

    elif color == "Yellow":
        label = "Yellow Cone"
        frameColor = (0,255,255)

    else:
        label = "Unknown"
        frameColor = (0,255,0)
    cv2.rectangle(frame, p1, p2, frameColor, 2, 1)
    cv2.putText(frame, label, (p1[0], p1[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, frameColor, 2, cv2.LINE_AA)

def DistToCenter (cones, depth):
    xyzToCones = []
    
    for cone in cones:
        x, y = cone
        x, y = int(x), int(y) 
        z = depth[y,x]
        z = int(z)
        xyzToCones.append((x, y, z))
    
    return xyzToCones




latestDistanceFrame = None

# This is where everything comes together 
def main():
    stopEvent = threading.Event()
    device ,scale_z, pixelFormat_initial, operating_mode_initial,  exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial = CreateDevice()


    def HeliosThread(device, scale_z):
        global latestDistanceFrame
        try:
            with device.start_stream(10):
                while not stopEvent.is_set():
                    heatmap, depth = HeliosRunning(device, scale_z)
                    if heatmap is None:
                        continue
                    latestDistanceFrame = (heatmap, depth)
        except Exception as e:
            print(f"Fejl i HeliosThread: {e}")

    global latestDistanceFrame
    latestDistanceFrame = None 
    
    t = threading.Thread(target=HeliosThread, args=(device, scale_z), daemon=True)
    t.start()

    # The setup() is loaded in
    cap, blueUpper, blueLower, yellowUpper, yellowLower = Setup()
    # Time is used to calculate FPS. (might not be necesarry to calculate for each frame, might be enough to display once a second)
    
    prev_time = time.time()


    while True:
        ret, frame = cap.read() # Get the frame from the camera
        current_time = time.time() #more fps
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        if not ret: # If there are is no camera
            print("Can't read camera, ending stream")
            break
        if latestDistanceFrame is None:
             print("Vent på data fra Helios")
        else:
             heatmap, depth = latestDistanceFrame
             cv2.imshow("Helios Heatmap", depth)

        # Get the masks created in Masking()
        HSV, maskBlue, maskYellow = Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower)
        # uses the masks in found Contours to find the locations of objects
        bboxesBlue, bboxesYellow = FindContours(maskBlue, maskYellow)

        #set_nodes(device.nodemap)
        
        bbBlue = CompareWithHelios(bboxesBlue, frame, depth)
        bbYellow = CompareWithHelios(bboxesYellow, frame, depth)

        # Checks if it has found any Blue and Yellow BBoxes, before trying to draw them
        
        combine, edgeContours, cleanedMask = EdgeDetection(HSV)

        # cv2.drawContours drawing all contours (-1) in red, (BGR: 0,0,255) with thickness 3
        frameEdges = frame.copy()
        cv2.drawContours(frameEdges, edgeContours, -1, (0, 0, 255), 3)

        #verifying edge-detection
        bboxesBlueVerified = VerifyConesWithEdges(bboxesBlue, combine, 0.1)
        bboxesYellowVerified = VerifyConesWithEdges(bboxesYellow, combine, 0.3)

        bboxBlue, bboxYellow, _, _, centerPointsBlue, centerPointsYellow = MergeBbox(bboxesBlueVerified , bboxesYellowVerified)

        positionBlue = DistToCenter(centerPointsBlue, depth)

        print(positionBlue)

        #draw the varified boxes and edges
        for boxb in bboxBlue:
            DrawBoundingBox(boxb, frameEdges, "Blue")
        for boxy in bboxYellow:
            DrawBoundingBox(boxy, frameEdges, "Yellow")
        for boxb in bboxesBlueVerified:
            cv2.putText(frameEdges, "Edge-verified", (int(boxb[1]), int(boxb[2]-20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        for boxy in bboxesYellowVerified:
            cv2.putText(frameEdges, "Edge-verified", (int(boxy[1]), int(boxy[2]-20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
         

        #cv2.imshow("frame", frame)
        cv2.imshow("frame Edges", frameEdges)
        #cv2.imshow("Frame with boxes and edges", maskBlue)
        # Show the combined mask and frame with bboxes
        #cv2.imshow("mask", mask)
        # prints FPS
        print(f"FPS: {fps}")
        if cv2.waitKey(1) == ord('q'):
            break
    
    stopEvent.set()
    t.join()

    cap.release()
    cv2.destroyAllWindows()

    time.sleep(0.1)

    try:
        HeliosEnd(device, pixelFormat_initial, operating_mode_initial,  exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial)
    except Exception as e:
        print("Fejl under HeliosEnd", e)

if __name__ == "__main__":
    main()
    