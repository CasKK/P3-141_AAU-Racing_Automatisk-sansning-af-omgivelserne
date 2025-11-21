#AAU Robotics ROB3 GR.141 Project 
# Cone detection algorithm



#WHAT NEEDS TO BE IMPLEMENTED AND IMPORVED
# Better noise reduction and/or filtering, it still locates too much noise, especially from blue color
# Center positions needs to be exported as a vector with space for distance
# improve merge function so there is not a hard limit on what the distance is, it might interfere with other cones in the distance
# 



import numpy as np
import cv2
import time
kernel = np.ones([5,5], np.uint8)

# The setup function is where the camera is inititiated and the limits for color segmentation is defined in HSV
def Setup():
    cam = cv2.VideoCapture(0)
    upperLimitBlue = np.array([123, 252, 150], np.uint8)
    lowerLimitBlue = np.array([103, 92, 5], np.uint8)
    upperLimitYellow = np.array([33,255,255], np.uint8)
    lowerLimitYellow = np.array([13,120,120], np.uint8)
    return cam, upperLimitBlue, lowerLimitBlue, upperLimitYellow, lowerLimitYellow

def sobel_edge_detection(frame):
    #splitting hsv pictures 
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    
    frame_blur = cv2.GaussianBlur(v, (7,7), 0) 

    #tager v kanalen og laver en sobeledge detection med kernel 3,3
    sobelx = cv2.Sobel(frame_blur, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(frame_blur, cv2.CV_16S, 0, 1, ksize=3)

    #konverter til float32
    sobelx = sobelx.astype(np.float32)
    sobely = sobely.astype(np.float32)

    #beregner gradient magnituden og thresholder den
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    _, gradient_thresh = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_TOZERO)

    #converterer til uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_thresh)

    return gradient_magnitude, frame_blur

def canny_edge_detection(frame_blur):
    #laver blur på v kanalen og bruger til canny edge detection
    edges = cv2.Canny(frame_blur, threshold1=50, threshold2=150)

    return edges

def morphology(combine):
    #morphology
    closing = cv2.morphologyEx(combine, cv2.MORPH_CLOSE, kernel, iterations=1) #lukker små huller
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1) #fjern små støjpixels udenfor kanterne
    # Konverter billedet til binært: pixels >20 bliver hvide og resten bliver sort
    _, binary = cv2.threshold(opening, 20, 255, cv2.THRESH_BINARY)
     
    return binary

def remove_small_objects_with_contours(binary, min_area=300):
    #Find alle ydre konturer i det binære billede
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Opret en tom maske med samme størrelse som inputbilledet
    mask = np.zeros_like(binary)

    #Liste til at gemme konturer, der er store nok
    kept_contours = []

    #Gennemgå hver kontur
    for c in contours:
        #Beregn konturens areal
        area = cv2.contourArea(c)
        #Gem kun konturer, der er større end min_area
        if area >= min_area:
            kept_contours.append(c)

    #Tegn alle gemte konturer som fyldte hvide objekter på masken
    if kept_contours:
            cv2.drawContours(mask, kept_contours, -1, 255, thickness=cv2.FILLED)

    return mask, kept_contours


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
        return maskBlue, maskYellow

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
            #Beregn arealet af konturen i pixelsd
            area = cv2.contourArea(c)
            if area > 100: #100 pixel^2 - skal være større end 100 pixels
                #Bregen den kovekse skal af kontiren (mindste konvekse fporm der omslutter konturen)
                hull = cv2.convexHull(c)
                #Arealet af den kovekse skal
                hull_area = cv2.contourArea(hull)
                
                #Solidity = hvor "fyldt" konturen er i forholf til dens konvekse skal
                #Hvis solidity er tæt på 1, er kontoren tæt på at være konveks og solid
                solidity = area / hull_area if hull_area != 0 else 0
              
                if solidity > 0.8: #The convex hull is at least 80% filled
                    #Beregn den mindste rektangel (bbox), der omslutter konturen
                    x,y,w,h = cv2.boundingRect(c)

                    if w > 10 and h > 10: #Filter: bbox skal være mindst 10x10 pixels
                        bbox.append((x,y,w,h)) # Add the object to the bbox list
                        
    FilterBbox(contoursBlue, bboxBlue)
    FilterBbox(contoursYellow, bboxYellow)
    # The Bboxes are then being merged in the MergeBbox() function to merge close bboxes
    bboxBlue, bboxYellow, _, _ = MergeBbox(bboxBlue, bboxYellow)
    return bboxBlue, bboxYellow

# The MergeBbox function is used to merge the base and top parts of the cones
# It has two helper functions CenterCalc(), where it calculates the centers of each bbox and its it to a list.
# MergeCenters then filters through these centers and checks if two centers are within the predefined pixel distance
# if they are a new center is calculated as the average of both ccenters, and a new bbox is defined.
def MergeBbox(bboxBlue, bboxYellow):
    def CenterCalc(bbox, centrum):
        for b in bbox:
            x,y,w,h = b
            cx = x+w/2
            cy = y+h/2
            centrum.append((cx,cy, w, h,x,y))
    def MergeCenters(centrum, merged, mergedCorner):
        yDist = 250 
        xDist = 50
        for c in centrum: # go through each item in the list of center locations
            found = False # Found is false at the start since the first center is not close to any of the others since there is none to compare to
            for i, m in enumerate(merged): # For each iteration i, and m in enumerate(merged) i is the iteration and m is the corresponding entry in merged
                if (abs(c[1]-m[1]) < yDist) and (abs(c[0]-m[0]) < xDist): # Checks to see if the center c is close enough to the entry m
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

    centrumB = []
    centrumY = []
    mergedBlue = []
    mergedYellow = []
    mergedBlueCorner = []
    mergedYellowCorner = []
    CenterCalc(bboxBlue, centrumB)
    CenterCalc(bboxYellow, centrumY)
    MergeCenters(centrumB, mergedBlue, mergedBlueCorner)
    MergeCenters(centrumY, mergedYellow, mergedYellowCorner)

    return mergedBlueCorner, mergedYellowCorner, mergedBlue, mergedYellow

# This function draws the bounding boxes
def DrawBoundingBox(box, frame, color):
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))

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

def verify_cones_with_edges(bboxes, edges, threshold=0.30):
    verified = [] #Liste til de bboxes, der bliver verificeret som gyldige
    
    for box in bboxes:
        #Konverterer bpunding box-koordinater til heltal
        x, y, w, h = map(int, box)

        #Udtræk regionen af interesse (ROI) fra edge-billedet
        #Det vil sige kund et område hvor bbox'en ligger
        roi = edges[y:y+h, x:x+w]
        
        #Tæl hvor mange pixels i ROI der faktsik kanter (ikke-nul pixels)
        edge_pixels = cv2.countNonZero(roi)
        
        #Beregn det totale antal pixels i ROI
        total_pixels = w * h

        #Hvis andelen af edge-pixels i ROI overstiger threshold
        if total_pixels > 0 and (edge_pixels / total_pixels >= threshold):
            #Tilføj bbox til listen over verificerede objekter
            verified.append(box)
    return verified
    

# This is where everything comes together 
def main():
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
        
        gradient_magnitude, v = sobel_edge_detection(frame)
        edges = canny_edge_detection(v)

        #combinere sobel og canny edge qdetection
        combine = cv2.bitwise_or(gradient_magnitude, edges)

    
    
        #Lav binært bilelde med morphology
        binary = morphology(combine)

        # Remove små objekter hurtigt med contour-filter (erstatter connectedComponents)
        cleaned_mask, kept_contours = remove_small_objects_with_contours(binary, min_area=250)


        # Get the masks created in Masking()
        maskBlue, maskYellow = Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower)
        # uses the masks in found Contours to find the locations of objects
        bboxesBlue, bboxesYellow = FindContours(maskBlue, maskYellow)
        # Checks if it has found any Blue and Yellow BBoxes, before trying to draw them
        
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, edge_contours, -1, (0, 0, 255), 3)

        
        #verificer ,ed edge-detection
        bboxesBlue_verified = verify_cones_with_edges(bboxesBlue, combine)
        bboxesYellow_verified = verify_cones_with_edges(bboxesYellow, combine)

        #tegne de verificerede
        for boxb in bboxesBlue:
            DrawBoundingBox(boxb, frame, "Blue")
        for boxy in bboxesYellow:
            DrawBoundingBox(boxy, frame, "Yellow")
        for boxb in bboxesBlue_verified:
            cv2.putText(frame, "Edge-verified", (int(boxb[0]), int(boxb[1]-20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        for boxy in bboxesYellow_verified:
            cv2.putText(frame, "Edge-verified", (int(boxy[0]), int(boxy[1]-20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        
       
        cv2.imshow("Frame with boxes and edges", cleaned_mask)
        # Show the combined mask and frame with bboxes
        #cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)
        # prints FPS
        print(f"FPS: {fps}")
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()