
import numpy as np
import cv2
import time
import subprocess
import math
import glob
from .helios_create_image import CreateDevice, HeliosRunning, HeliosEnd
import threading
kernel = np.ones([5,5], np.uint8)
debug = False

frame_width = 640
frame_height = 480
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(f"output_{time.time()}.avi", fourcc, 30.0, (frame_width, frame_height))

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
    setCameraControl(selectedDevice, "exposure_time_absolute", 250)
    setCameraControl(selectedDevice, "gain", 10)#250)
    setCameraControl(selectedDevice, "focus_automatic_continuous",0)
    setCameraControl(selectedDevice, "focus_absolute", 0)
    setCameraControl(selectedDevice, "white_balance_automatic", 1)

    upperLimitBlue = np.array([123, 252, 150], np.uint8)
    lowerLimitBlue = np.array([103, 92, 5], np.uint8)
    upperLimitYellow = np.array([33,255,230], np.uint8)
    lowerLimitYellow = np.array([13,120,30], np.uint8)

    bayesValues = {
        "topMean": [0.9644363764088177, 1.1378143182985014, 1.3961731590237239, 0.8197422981794419, 6.8076923076923075],
        "topVar": [0.0012718660489877433, 0.004174815174160363, 0.03272987798648929, 0.0059563763778370015, 19.693786982248522],
        "bottomMean": [0.9415715247980834, 1.1681854816697144, 1.2863062514179588, 0.811978889557676, 8.96],
        "bottomVar": [0.0017130435723415446, 0.009287333475935779, 0.021151436488789154, 0.005764079511957598, 15.7184],
        "mix1Mean": [0.9644363764088177, 1.1378143182985014, 1.3961731590237239, 0.8197422981794419, 6.8076923076923075],
        "mix1Var": [0.0012718660489877433, 0.004174815174160363, 0.03272987798648929, 0.0059563763778370015, 19.693786982248522],
        "mix2Mean": [0.9644363764088177, 1.1378143182985014, 1.3961731590237239, 0.8197422981794419, 6.8076923076923075],
        "mix2Var": [0.0012718660489877433, 0.004174815174160363, 0.03272987798648929, 0.0059563763778370015, 19.693786982248522]
    }
    
    return cam, upperLimitBlue, lowerLimitBlue, upperLimitYellow, lowerLimitYellow, bayesValues

def WarpFrame(frame, depth):

    
    matrix = np.array([[ 8.43114000e-01, -3.84176124e-02,  2.83616563e+01],
                        [-2.15365538e-02,  8.45311445e-01,  4.68435043e+01],
                          [-1.13793182e-04, -9.33131759e-05,  1.00000000e+00]])
    

    #warp billede2 på billede1
    height, width = depth.shape[:2]
    warpedFrame = cv2.warpPerspective(frame, matrix, (width, height))
    # cv2.imwrite("WarpedImg2.png", warped_img2)


    return warpedFrame



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
                if not 50 < area < 15000:
                    return False
                
                # Check solidity
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area != 0 else 0
                if not 0.7 <solidity < 1:
                    return False

                # Check circularity
                perimeter = cv2.arcLength(c, True)
                circularity = perimeter/(2*math.sqrt(math.pi*area))
                if not (1 < circularity < 2):
                    return False

                # Check aspect
                rect = cv2.minAreaRect(c)
                (w_minArea,h_minArea) = rect[1]
                if h_minArea > w_minArea:
                    aspect = h_minArea / w_minArea if w_minArea != 0 else 0
                else:
                    aspect = w_minArea / h_minArea if h_minArea != 0 else 0
                if not (1 < aspect < 1.9):
                    return False

                # Check compactness
                compactness = area / (h_minArea*w_minArea) if (h_minArea*w_minArea) != 0 else 0
                if not (0.5 < compactness < 0.95):
                    return False

                # Check if the bbox is to small
                x,y,w_bbox,h_bbox = cv2.boundingRect(c)
                if not (w_bbox > 10 and h_bbox > 10):
                    return False
                
                # Covexity defects
                hullC = cv2.convexHull(c, returnPoints=False)
                defects = cv2.convexityDefects(c, hullC)
                numDefects = defects.shape[0] if defects is not None else 0
                if not (1 < numDefects < 20):
                    return False

                # Calculate a diagonal
                d = math.sqrt(h_minArea**2 + w_minArea**2)

                return (c, x, y, w_bbox, h_bbox, d, (solidity, circularity, aspect, compactness, numDefects), area)
            
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



    #Finding contours in the binary edge picture and draw a frame 
        # edge_contours: listing with points for every contour
        # cv2.RETR_EXTERNAL: only outer contours, ingoring inner holes 
        # cv2.CHAIN_APPROX_SIMPLE: compress contourpoints along side a line to save memmory
    edgeContours, _ = cv2.findContours(combine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return combine, edgeContours

def VerifyConesWithEdges(bboxes, edges, threshold):
    verified = [] #Listing for the bboxes, that is being verified as valid 
    
    _, edgesBin = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)

    for box in bboxes:
        #Converting bounding box coordinates to a integer
        c, _, _, _, _, _, _, _ = box

        maskEdge = np.zeros_like(edgesBin)
        cv2.drawContours(maskEdge, [c], -1, (255), 4)
        
        overlap = cv2.bitwise_and(maskEdge, edgesBin)

        edgePixels = cv2.countNonZero(maskEdge)
        overlapPixels = cv2.countNonZero(overlap)

        if edgePixels > 0 and (overlapPixels / edgePixels >= threshold):
            verified.append(box)

    return verified

def ShapeClassification(bayesValues, bbox):

    def gaussianLogProb(x, mu, var):
        return -0.5*math.log(2*math.pi*var) - ((x-mu)**2)/(2*var)

    def classLogScore(x, mu, var, prior):
        log_probs = [gaussianLogProb(xi, mui, vari) for xi, mui, vari in zip(x, mu, var)]
        return sum(log_probs) + math.log(prior)
    
    classified = []

    for b in bbox:
        c, x, y, w, h, d, features, a = b

        priorTop = 0.4
        priorBottom = 0.4
        priorMix1 = 0.05
        priorMix2 = 0.1

        logTop = classLogScore(features, bayesValues["topMean"], bayesValues["topVar"], priorTop)
        logBottom = classLogScore(features, bayesValues["bottomMean"], bayesValues["bottomVar"], priorBottom)
        logMix1 = classLogScore(features, bayesValues["mix1Mean"], bayesValues["mix1Var"], priorMix1)
        logMix2 = classLogScore(features, bayesValues["mix2Mean"], bayesValues["mix2Var"], priorMix2)

        compare = {
            0: logTop, 1: logBottom, 2: logMix1, 3: logMix2
        }

        predicted = max(compare, key=compare.get)
      
        classified.append((c, x, y, w, h, d, predicted, a))

    return classified



def CompareWithHelios(contours, frame, depth):   
    '''
    Function to check if a BLOB is seperated by distance and therefore should be to different objekts.
    '''

    def Scale(contour, x, y, w, h, scale):
        cx = x + w / 2
        cy = y + h / 2

        #   https://nvsabhilash.me/2019/06/29/Scaling-and-Rotating-contours.html?utm_source=chatgpt.com
        scaledContour = contour.copy().astype(np.float32)
        scaledContour[:, 0, 0] = cx + (scaledContour[:, 0, 0] - cx) * scale
        scaledContour[:, 0, 1] = cy + (scaledContour[:, 0, 1] - cy) * scale
        scaledContour = scaledContour.astype(np.int32)

        return scaledContour, cx, cy

    newContours = []
    mixCone = []

    depth = cv2.medianBlur(depth, 5)


    # For-loop to go through all contours in a list to
    for contour in contours:
        c, x, y, w, h, d, t, a = contour
        
        if t == 0 or t == 1 or t == 2  :# or t == 3:

            scaledContour, cx, cy = Scale(c, x, y, w, h, 0.05)

            # Lav en maske ud fra den skalerede kontur
            mask = np.zeros_like(frame[:, :, 0])
            cv2.drawContours(mask, [scaledContour], -1, 255, thickness=cv2.FILLED)  # fyld konturen

            # Calculates mean of depth
            mean = cv2.mean(depth, mask=mask)[0]

            if t == 2:

                mixCone.append((cx, cy, mean))
            else:
                newContours.append((x, y, w, h, d, mean, a))
        
        if t == 3:
            scaledContour, _, _ = Scale(c, x, y, w, h, 0.8)

            # Makes a mask that from the current contour     
            mask = np.zeros_like(frame[:, :, 0])
            cv2.drawContours(mask, [scaledContour], -1, 255, thickness=cv2.FILLED)

            
            # Use the mask of the on the depthmap to isolate the area, using median filter to remove noise, flattening it to a 1D-list
            # with only every 3 pixel. Then sorting it
            depthValues = depth[mask==255]
            depthValues = depthValues.flatten()[::3]
            depthValues.sort()

            peaks = []
            currentPeak = []

            # Making a threshold relative to the size of the BLOB
            threshold = np.count_nonzero(mask)*0.1


            # For-loop that goes through the list and make an nev list if the value between to elements is higher than 150.
            # Each new current list is a peak, and will be appended to the peak list if there is a minimum of pixel in it
            for i, value in enumerate(depthValues):
                if i == 0:
                    currentPeak.append(int(value)) 
                else:
                    if (value - depthValues[i-1]) < 150:
                        currentPeak.append(int(value))
                    else:
                        if (len(currentPeak) > threshold):
                            if (max(currentPeak) - min(currentPeak)) < 200:
                                peaks.append(currentPeak)
                        currentPeak = []
                        currentPeak.append(int(value))

            # If there is only one peak, then it is its own BLOB, the mean depth value will be calculated and the original contour is appended to newContours
            if len(peaks)==1:
                mean = int( sum(peaks[0]) / len(peaks[0]) )
                newContours.append((x, y, w, h, d, mean, a))

            # If there are more it will finde new contour for each peak.
            elif len(peaks) > 1:
                for p in peaks:
                    mean = int( sum(p) / len(p) )
                    innerMask = cv2.inRange(depth, p[0],p[-1])
                    innerMask = cv2.morphologyEx(innerMask, cv2.MORPH_OPEN, kernel, iterations=2)
                    innerMask = cv2.morphologyEx(innerMask, cv2.MORPH_CLOSE, kernel, iterations=2)
                    combinedMask = cv2.bitwise_and(mask, innerMask)
                    cNew, _ = cv2.findContours(combinedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(cNew) > 0:
                        for c in cNew:
                            area = cv2.contourArea(c)
                            if area > threshold:
                                xNew, yNew, wNew, hNew = cv2.boundingRect(c)
                                dNew = math.sqrt(hNew**2 + wNew**2)
                                newContours.append((xNew, yNew, wNew, hNew, dNew, mean, a))

    return newContours, mixCone  

# The MergeBbox function is used to merge the base and top parts of the cones
# It has two helper functions CenterCalc(), where it calculates the centers of each bbox and its it to a list.
# MergeCenters then filters through these centers and checks if two centers are within the predefined pixel distance
# if they are a new center is calculated as the average of both ccenters, and a new bbox is defined.

def MergeBbox(bboxBlue, bboxYellow):
    def CenterCalc(bbox, centrum):
        for b in bbox:
            x,y,w,h,diameter,depth, a = b
            cx = x+w/2
            cy = y+h/2
            centrum.append((cx, cy, w, h, x, y, diameter, depth, a))
    def MergeCenters(centrum, merged, mergedCorner, mergedCenter):
        centrum.sort(key=lambda c: c[7])
        for c in centrum: # go through each item in the list of center locations
            cxC, cyC, wC, hC, xC, yC, diameterC, depthC, aC = c
            found = False # Found is false at the start since the first center is not close to any of the others since there is none to compare to
            for i, m in enumerate(merged): # For each iteration i, and m in enumerate(merged) i is the iteration and m is the corresponding entry in merged
                cxM, cyM, wM, hM, depthM, aM = m
                if aC > aM:
                    a = aC
                elif aC < aM:
                    a = aM
                else:
                    a = aM
                if (abs(cyC-cyM) < diameterC*2) and (abs(cxC-cxM) < diameterC*1.5) and (abs(depthC-depthM) < 900): # Checks to see if the center c is close enough to the entry m
                    merged[i] = ((cxC+cxM)/2, (cyC+cyM)/2, abs(cyC-cyM)*1.5, abs(cxC-cxM)*3, abs((depthC+depthM)/2), a) # If it is close enough a new center is calculated
                    x1 = min(cxC-wC/2, cxM-wM/2) # The minimum and maximum values for the new center is created
                    y1 = min(cyC-hC/2, cyM-hM/2)
                    x2 = max(cxC+wC/2, cxM+wM/2)
                    y2 = max(cyC+hC/2, cyM+hM/2)
                    w_new = x2 - x1
                    h_new = y2 - y1 #This is so a new bbox can be created
                    mergedCorner[i] = (x1, y1, w_new, h_new, depthC, a) # Add the new bbox to the mergedCorner list, this is the upper left corner and is standard for  reating bboxes
                    found = True # set found to true
                    break # go out of the for loop so the next center will be assed
            if not found: # If found != True it just adds the original positions
                merged.append((cxC,cyC,wC,hC,depthC, aC)) #This is the center position
                mergedCorner.append((xC,yC,wC,hC,depthC, aC)) #This is for the upper corner and correct bbox creation
               
                cx = cxC+wC/2
                cy = cyC+hC/2
                mergedCenter.append((cx,cy,depthC))

    centrumB, centrumY,  mergedBlue, mergedYellow, mergedBlueCorner, mergedYellowCorner, mergedCenterBlue, mergedCenterYellow = [], [], [], [], [], [], [], []
    CenterCalc(bboxBlue, centrumB)
    CenterCalc(bboxYellow, centrumY)
    MergeCenters(centrumB, mergedBlue, mergedBlueCorner, mergedCenterBlue)
    MergeCenters(centrumY, mergedYellow, mergedYellowCorner, mergedCenterYellow)
   
    return mergedBlueCorner, mergedYellowCorner, mergedBlue, mergedYellow, mergedCenterBlue, mergedCenterYellow

# This function draws the bounding boxes
def DrawBoundingBox(box, frame, color, depth):
    x, y, w, h, d, _ = box
    p1 = (int(x), int(y))
    p2 = (int(x+w), int(y+h))
    # x, y = int(x), int(y)
    # z = depth[y,x]

    if d != 0:
        if  color == "Blue":
            label = f"{d:.0f}" if d is not None else 0 #"Blue Cone"
            frameColor = (255,0,0)

        elif color == "Yellow":
            label = f"{d:.0f}" if d is not None else 0 #"Yellow Cone"
            frameColor = (0,255,255)

        else:
            label = "Unknown"
            frameColor = (0,255,0)
        cv2.rectangle(frame, p1, p2, frameColor, 2, 1)
        cv2.putText(frame, label, (p1[0], p1[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, frameColor, 2, cv2.LINE_AA)

def DistToCenter (conesBlue1, conesBlue2, conesYellow1, conesYellow2, depth):
    blueArray, yellowArray, combinedArray = [], [], []
    
    def CalcZ(cones1, cones2, array, combined):
        for cone in cones1:
            x, y, z = cone          
            x, y, z = int(x), int(y), int(z)
            if z != 0:
                array.append((x, y, z))
            #print(f"Dist: {z}")
        for cone in cones2:
            x, y, z = cone
            x, y = int(x), int(y), int(z)
            if z != 0:
                array.append((x, y, z))
            #print(f"Dist: {z}")
        combined.append(array)

    CalcZ(conesBlue1, conesBlue2, blueArray, combinedArray)
    CalcZ(conesYellow1, conesYellow2, yellowArray, combinedArray)
    
    return combinedArray

latestDistanceFrame = None

# This is where everything comes together 
def run(output_queue):
# def main():
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

    
    t = threading.Thread(target=HeliosThread, args=(device, scale_z), daemon=True)
    t.start()

    # The setup() is loaded in
    cap, blueUpper, blueLower, yellowUpper, yellowLower, bayesValues = Setup()
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
            #  print(f"dybde: {np.asanyarray(depth).shape}")
             #cv2.imshow("Helios Heatmap", depth)

        frame = WarpFrame(frame, depth)

        # Get the masks created in Masking()
        HSV, maskBlue, maskYellow = Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower)
        # uses the masks in found Contours to find the locations of objects
        bboxesBlue, bboxesYellow = FindContours(maskBlue, maskYellow)


        # Checks if it has found any Blue and Yellow BBoxes, before trying to draw them
        
        combine, edgeContours = EdgeDetection(HSV)

        # cv2.drawContours drawing all contours (-1) in red, (BGR: 0,0,255) with thickness 3
        frameEdges = frame.copy()
        if debug == True:
            cv2.drawContours(frameEdges, edgeContours, -1, (0, 0, 255), 3)

        # Verifying edge-detection
        bboxesBlueVerified = VerifyConesWithEdges(bboxesBlue, combine, 0.1)
        bboxesYellowVerified = VerifyConesWithEdges(bboxesYellow, combine, 0.3)

        # Shape Classify
        bboxesBlueClassified = ShapeClassification(bayesValues, bboxesBlueVerified)
        bboxesYellowClassified = ShapeClassification(bayesValues, bboxesYellowVerified)

        

        # Depth
        blobBlue, mixBlue = CompareWithHelios(bboxesBlueClassified, frame, depth)
        blobYellow, mixYellow = CompareWithHelios(bboxesYellowClassified, frame, depth)

        bboxBlue, bboxYellow, _, _, centerPointsBlue, centerPointsYellow = MergeBbox(blobBlue , blobYellow)

        
        blended = cv2.addWeighted(np.float32(heatmap), 0.2, np.float32(frame), 0.2, 0)


        # Normalize the result to make sure it's in the range [0, 255]
        blended_float = np.clip(blended, 0, 255)  # This will clamp any values outside [0, 255]

        # Convert the result back to uint8 (for display or saving)
        blended = np.uint8(blended_float)

        positions = DistToCenter(centerPointsBlue, mixBlue, centerPointsYellow, mixYellow, depth)
        print(f"ConePos: {positions}")


        output_queue.put(positions)

        #draw the varified boxes and edges
        for boxb in bboxBlue:
            DrawBoundingBox(boxb, frameEdges, "Blue", depth)
        for boxy in bboxYellow:
            DrawBoundingBox(boxy, frameEdges, "Yellow", depth)
        # for boxb in bboxesBlueVerified:
        #     cv2.putText(frameEdges, "Edge-verified", (int(boxb[1]), int(boxb[2]-20)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        # for boxy in bboxesYellowVerified:
        #     cv2.putText(frameEdges, "Edge-verified", (int(boxy[1]), int(boxy[2]-20)), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
         
        # with open("depthTest.txt", "a") as f:
        #     f.write(f"{depth}\n")

        with open("fps.txt", "a") as f:
            f.write(f"{fps}\n")
        out.write(frameEdges)
        #cv2.imshow("frame", frame)
        #cv2.imshow("frame Edges", frameEdges)
        # print(f"frame: {np.asanyarray(frame).shape}")
        #cv2.imshow("depth", frame)
        #cv2.imshow("depth", blended)


        #cv2.imshow("Frame with boxes and edges", maskBlue)
        # Show the combined mask and frame with bboxes
        #cv2.imshow("mask", mask)
        # prints FPS
        # print(f"FPS: {fps}")
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
        import traceback
        traceback.print_exc()


# if __name__ == "__main__":
#     main()
    