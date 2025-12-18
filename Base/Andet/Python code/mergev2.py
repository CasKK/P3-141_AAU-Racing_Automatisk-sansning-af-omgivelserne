def MergeBbox(bboxBlue, bboxYellow):
    def CenterCalc(bbox, centrum):
        for b in bbox:
            x,y,w,h,diameter,depth = b
            cx = x+w/2
            cy = y+h/2
            centrum.append((cx, cy, w, h, x, y, diameter, depth))
    def MergeCenters(centrum, merged, mergedCorner, mergedCenter):
        centrum.sort(key=lambda c: c[7])
        for c in centrum: # go through each item in the list of center locations
            cxC, cyC, wC, hC, xC, yC, diameterC, depthC = c
            found = False # Found is false at the start since the first center is not close to any of the others since there is none to compare to
            for i, m in enumerate(merged): # For each iteration i, and m in enumerate(merged) i is the iteration and m is the corresponding entry in merged
                cxM, cyM, wM, hM, depthM = m
                if (abs(cyC-cyM) < diameterC*3) and (abs(cxC-cxM) < diameterC*1.5) and (abs(depthC-depthM) < 500): # Checks to see if the center c is close enough to the entry m
                    merged[i] = ((cxC+cxM)/2, (cyC+cyM)/2, abs(cyC-cyM)*1.5, abs(cxC-cxM)*3, abs((depthC+depthM)/2)) # If it is close enough a new center is calculated
                    x1 = min(cxC-wC/2, cxM-wM/2) # The minimum and maximum values for the new center is created
                    y1 = min(cyC-hC/2, cyM-hM/2)
                    x2 = max(cxC+wC/2, cxM+wM/2)
                    y2 = max(cyC+hC/2, cyM+hM/2)
                    w_new = x2 - x1
                    h_new = y2 - y1 #This is so a new bbox can be created
                    mergedCorner[i] = (x1, y1, w_new, h_new, depthC) # Add the new bbox to the mergedCorner list, this is the upper left corner and is standard for  reating bboxes
                    found = True # set found to true
                    break # go out of the for loop so the next center will be assed
            if not found: # If found != True it just adds the original positions
                merged.append((cxC,cyC,wC,hC,depthC)) #This is the center position
                mergedCorner.append((xC,yC,wC,hC,depthC)) #This is for the upper corner and correct bbox creation
               
                cx = cxC+wC/2
                cy = cyC+hC/2
                mergedCenter.append((cx,cy,depthC))

    centrumB, centrumY,  mergedBlue, mergedYellow, mergedBlueCorner, mergedYellowCorner, mergedCenterBlue, mergedCenterYellow = [], [], [], [], [], [], [], []
    CenterCalc(bboxBlue, centrumB)
    CenterCalc(bboxYellow, centrumY)
    MergeCenters(centrumB, mergedBlue, mergedBlueCorner, mergedCenterBlue)
    MergeCenters(centrumY, mergedYellow, mergedYellowCorner, mergedCenterYellow)
   
    return mergedBlueCorner, mergedYellowCorner, mergedBlue, mergedYellow, mergedCenterBlue, mergedCenterYellow
