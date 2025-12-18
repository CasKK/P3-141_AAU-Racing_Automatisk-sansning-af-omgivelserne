import cv2
import os
import time
import threading
import subprocess
import glob
from helios_create_image import CreateDevice, HeliosRunning, HeliosEnd
import numpy as np
from threading import Lock

kernel = np.ones([5,5], np.uint8)
latestDistanceFrame = None
latestDistanceFrameLock = Lock()
camera_lock = threading.Lock()

# Setup kameraet og farvesegmentering
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

    print("Bruger kamera:", selectedDevice)
    
    # Indstil kameraet
    auto_gain = getCameraControl(selectedDevice, "gain")
    setCameraControl(selectedDevice, "auto_exposure", 1)
    setCameraControl(selectedDevice, "exposure_time_absolute", 200)
    setCameraControl(selectedDevice, "gain", 10)  # Justér efter behov
    setCameraControl(selectedDevice, "focus_automatic_continuous", 0)
    setCameraControl(selectedDevice, "focus_absolute", 0)
    setCameraControl(selectedDevice, "white_balance_automatic", 1)

    upperLimitBlue = np.array([123, 252, 150], np.uint8)
    lowerLimitBlue = np.array([103, 92, 5], np.uint8)
    upperLimitYellow = np.array([35,255,255], np.uint8)
    lowerLimitYellow = np.array([13,100,30], np.uint8)
    return cam, upperLimitBlue, lowerLimitBlue, upperLimitYellow, lowerLimitYellow

def WarpFrame(frame, depth):
    matrix4 = np.array([[ 8.43114000e-01, -3.84176124e-02,  2.83616563e+01],
                        [-2.15365538e-02,  8.45311445e-01,  4.68435043e+01],
                        [-1.13793182e-04, -9.33131759e-05,  1.00000000e+00]])
    
    height, width = depth.shape[:2]
    warpedFrame = cv2.warpPerspective(frame, matrix4, (width, height))
    return warpedFrame

# Maskering og billedebehandling
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

# Funktion til at vælge et område (ROI) på masken
def select_roi_on_mask(mask):
    roi = cv2.selectROI("Select Region on Mask", mask, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region on Mask")
    return roi

# Funktion til at gemme billeder i den rigtige mappe og med sekventielt navn
def save_image_in_class_folder(image, class_key, output_dirs):
    folder = output_dirs[class_key]
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_list = os.listdir(folder)
    file_numbers = [int(f.split('.')[0]) for f in file_list if f.split('.')[0].isdigit()]
    next_num = max(file_numbers, default=0) + 1

    image_filename = os.path.join(folder, f"{class_key}_{next_num}.jpg")
    cv2.imwrite(image_filename, image)
    print(f"Billede gemt som {image_filename}")

# Denne funktion håndterer tasterne 'a', 'b', 'c', 'd' og gemmer et billede fra masken
def capture_and_save_image_from_mask(maskBlue, maskYellow, output_dirs, key):
    if key in [ord('a'), ord('b'), ord('c'), ord('d')]:
        if key == ord('a'):
            roi = select_roi_on_mask(maskBlue)
            selected_mask = maskBlue
        elif key == ord('b'):
            roi = select_roi_on_mask(maskYellow)
            selected_mask = maskYellow
        else:
            return

        if roi != (0, 0, 0, 0):
            x, y, w, h = roi
            cropped_image = selected_mask[y:y+h, x:x+w]
            save_image_in_class_folder(cropped_image, chr(key), output_dirs)

# Hovedfunktionen hvor alt samles
def main():
    stopEvent = threading.Event()
    device, scale_z, pixelFormat_initial, operating_mode_initial, exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial = CreateDevice()

    def HeliosThread(device, scale_z):
        global latestDistanceFrame
        try:
            with device.start_stream(10):
                while not stopEvent.is_set():
                    heatmap, depth = HeliosRunning(device, scale_z)
                    if heatmap is None:
                        continue
                    with latestDistanceFrameLock:
                        latestDistanceFrame = (heatmap, depth)
        except Exception as e:
            print(f"Fejl i HeliosThread: {e}")

    t = threading.Thread(target=HeliosThread, args=(device, scale_z), daemon=True)
    t.start()

    # Setup kamera
    cap, blueUpper, blueLower, yellowUpper, yellowLower = Setup()

    prev_time = time.time()

    output_dirs = {'a': 'captured_images', 'b': 'captured_images', 'c': 'captured_images', 'd': 'captured_images'}

    while True:
        ret, frame = cap.read()
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        if not ret:
            print("Can't read camera, ending stream")
            break
        if latestDistanceFrame is None:
             print("Vent på data fra Helios")
        else:
             _, depth = latestDistanceFrame
             cv2.imshow("Helios depthmap", depth)

        frame = WarpFrame(frame, depth)

        # Maskering
        HSV, maskBlue, maskYellow = Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower)
        cv2.imshow("maskBlue", maskBlue)
        cv2.imshow("maskYellow", maskYellow)

        # Kald funktion til at gemme billede
        capture_and_save_image_from_mask(maskBlue, maskYellow, output_dirs, "a")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    stopEvent.set()
    t.join()

    cap.release()
    cv2.destroyAllWindows()

    time.sleep(0.1)

    try:
        HeliosEnd(device, pixelFormat_initial, operating_mode_initial, exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial)
    except Exception as e:
        print("Fejl under HeliosEnd", e)

if __name__ == "__main__":
    main()


# import cv2
# import os
# import time
# import threading
# import subprocess
# import glob
# from helios_create_image import CreateDevice, HeliosRunning, HeliosEnd
# import numpy as np
# from threading import Lock

# kernel = np.ones([5,5], np.uint8)
# latestDistanceFrame = None
# latestDistanceFrameLock = Lock()
# camera_lock = threading.Lock()

# # Setup kameraet og farvesegmentering
# def Setup():
#     def getCameraControl(device, name):
#         result = subprocess.run(
#             ["v4l2-ctl", f"--device={device}", "--get-ctrl", name],
#             capture_output=True, text=True
#         )
#         return result.stdout.strip()

#     def setCameraControl(device, control, value):
#         cmd = ["v4l2-ctl", f"--device={device}", "--set-ctrl", f"{control}={value}"]
#         subprocess.run(cmd, check=True)

#     videoDevices = sorted(glob.glob("/dev/video*"))
#     cam = None
#     selectedDevice = None

#     for dev in videoDevices:
#         index = int(dev.replace("/dev/video", ""))
#         print(f"Prøver {dev} ...")
        
#         with camera_lock:
#             cap = cv2.VideoCapture(index)
#             cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

#         if cap.isOpened():
#             print(f"Leder efter kamera på: {dev}")
#             cam = cap
#             selectedDevice = dev
#             break
#         else:
#             cap.release()

#     if not cam:
#         print("Ingen kamera fundet")
#         exit(1)

#     print("Bruger kamera:", selectedDevice)
    
#     # Indstil kameraet
#     auto_gain = getCameraControl(selectedDevice, "gain")
#     setCameraControl(selectedDevice, "auto_exposure", 1)
#     setCameraControl(selectedDevice, "exposure_time_absolute", 200)
#     setCameraControl(selectedDevice, "gain", 10)  # Justér efter behov
#     setCameraControl(selectedDevice, "focus_automatic_continuous", 0)
#     setCameraControl(selectedDevice, "focus_absolute", 0)
#     setCameraControl(selectedDevice, "white_balance_automatic", 1)

#     upperLimitBlue = np.array([123, 252, 150], np.uint8)
#     lowerLimitBlue = np.array([103, 92, 5], np.uint8)
#     upperLimitYellow = np.array([35,255,255], np.uint8)
#     lowerLimitYellow = np.array([13,100,30], np.uint8)
#     return cam, upperLimitBlue, lowerLimitBlue, upperLimitYellow, lowerLimitYellow

# def WarpFrame(frame, depth):
   
#     matrix4 = np.array([[ 8.43114000e-01, -3.84176124e-02,  2.83616563e+01],
#                         [-2.15365538e-02,  8.45311445e-01,  4.68435043e+01],
#                           [-1.13793182e-04, -9.33131759e-05,  1.00000000e+00]])
    
#     #warp billede2 på billede1
#     height, width = depth.shape[:2]
#     warpedFrame = cv2.warpPerspective(frame, matrix4, (width, height))
#     # cv2.imwrite("WarpedImg2.png", warped_img2)


#     return warpedFrame

# # Maskering og billedebehandling
# def Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower):
#     frame = cv2.GaussianBlur(frame, (5, 5), 0)
#     HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     maskBlue = cv2.inRange(HSV, blueLower, blueUpper)
#     maskYellow = cv2.inRange(HSV, yellowLower, yellowUpper)

#     maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_OPEN, kernel, iterations=2)
#     maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_CLOSE, kernel, iterations=2)
#     maskBlue = cv2.medianBlur(maskBlue, 5)

#     maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_OPEN, kernel, iterations=2)
#     maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_CLOSE, kernel, iterations=2)
#     maskYellow = cv2.medianBlur(maskYellow, 5)
#     return HSV, maskBlue, maskYellow

# # Funktion til at vælge et område (ROI) på masken
# def select_roi_on_mask(mask):
#     """
#     Vælger et område (ROI) på masken via OpenCVs selectROI-funktion.
#     """
#     roi = cv2.selectROI("Select Region on Mask", mask, fromCenter=False, showCrosshair=True)
#     cv2.destroyWindow("Select Region on Mask")
#     return roi

# # Funktion til at gemme billeder i den rigtige mappe og med sekventielt navn
# def save_image_in_class_folder(image, class_key, output_dirs):
#     """
#     Gemmer et billede i den relevante mappe og giver det et sekventielt navn.
#     """
#     folder = output_dirs[class_key]  # Find den mappe baseret på tasten
#     if not os.path.exists(folder):
#         os.makedirs(folder)

#     # Find det næste ledige billede i den mappe
#     file_list = os.listdir(folder)
#     file_numbers = [int(f.split('.')[0]) for f in file_list if f.split('.')[0].isdigit()]
#     next_num = max(file_numbers, default=0) + 1

#     # Gem billede med det næste nummer
#     image_filename = os.path.join(folder, f"{next_num}.jpg")
#     cv2.imwrite(image_filename, image)
#     print(f"Billede gemt som {image_filename}")

# # Denne funktion håndterer tasterne 'a', 'b', 'c', 'd' og gemmer et billede fra masken
# def capture_and_save_image_from_mask(maskBlue, maskYellow, output_dirs, key):
#     """
#     Denne funktion gemmer et billede ud fra maskBlue eller maskYellow, når en tast trykkes.
#     - maskBlue og maskYellow: de masker, du vil vælge et område fra.
#     - output_dirs: mappen til at gemme billeder.
#     - key: den tast der blev trykket (a, b, c, d).
#     """
#     # Håndter tasterne 'a', 'b', 'c', 'd'
#     if key in [ord('a'), ord('b'), ord('c'), ord('d')]:  # Hvis en af de ønskede taster trykkes
#         # Markér et område (ROI) på den valgte maske
#         if key == ord('a'):
#             roi = select_roi_on_mask(maskBlue)  # Vælg ROI på masken
#             selected_mask = maskBlue
#         elif key == ord('b'):
#             roi = select_roi_on_mask(maskYellow)  # Vælg ROI på masken
#             selected_mask = maskYellow
#         else:
#             return  # Hvis en ukendt tast trykkes, gør ingenting

#         if roi != (0, 0, 0, 0):  # Hvis der er valgt et område
#             x, y, w, h = roi
#             cropped_image = selected_mask[y:y+h, x:x+w]  # Skær masken ud i det valgte område

#             # Gem billedet i den rigtige mappe
#             save_image_in_class_folder(cropped_image, chr(key), output_dirs)

# # Hovedfunktionen hvor alt samles
# def main():
#     stopEvent = threading.Event()
#     device, scale_z, pixelFormat_initial, operating_mode_initial, exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial = CreateDevice()

#     def HeliosThread(device, scale_z):
#         global latestDistanceFrame
#         try:
#             with device.start_stream(10):
#                 while not stopEvent.is_set():
#                     heatmap, depth = HeliosRunning(device, scale_z)
#                     if heatmap is None:
#                         continue
#                     with latestDistanceFrameLock:
#                         latestDistanceFrame = (heatmap, depth)
#         except Exception as e:
#             print(f"Fejl i HeliosThread: {e}")

#     t = threading.Thread(target=HeliosThread, args=(device, scale_z), daemon=True)
#     t.start()

#     # Setup kamera
#     cap, blueUpper, blueLower, yellowUpper, yellowLower = Setup()

#     prev_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         current_time = time.time()
#         fps = 1 / (current_time - prev_time)
#         prev_time = current_time
#         if not ret:
#             print("Can't read camera, ending stream")
#             break
#         if latestDistanceFrame is None:
#              print("Vent på data fra Helios")
#         else:
#              _, depth = latestDistanceFrame
#              cv2.imshow("Helios depthmap", depth)

#         frame = WarpFrame(frame, depth)

#         # Maskering
#         HSV, maskBlue, maskYellow = Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower)
#         cv2.imshow("maskBlue", maskBlue)
#         cv2.imshow("maskYellow", maskYellow)
#         cv2.imshow("maskYellow", frame)

#         capture_and_save_image_from_mask(maskBlue, maskYellow, output_dirs, "a")

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#     stopEvent.set()
#     t.join()

#     cap.release()
#     cv2.destroyAllWindows()

#     time.sleep(0.1)

#     try:
#         HeliosEnd(device, pixelFormat_initial, operating_mode_initial, exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial)
#     except Exception as e:
#         print("Fejl under HeliosEnd", e)

# if __name__ == "__main__":
#     main()






# # import cv2
# # import os
# # import time
# # import threading
# # import subprocess
# # import glob
# # import numpy as np
# # from threading import Lock

# # kernel = np.ones([5,5], np.uint8)
# # latestDistanceFrame = None
# # latestDistanceFrameLock = Lock()
# # camera_lock = threading.Lock()

# # # Setup kameraet og farvesegmentering
# # def Setup():
# #     def getCameraControl(device, name):
# #         result = subprocess.run(
# #             ["v4l2-ctl", f"--device={device}", "--get-ctrl", name],
# #             capture_output=True, text=True
# #         )
# #         return result.stdout.strip()

# #     def setCameraControl(device, control, value):
# #         cmd = ["v4l2-ctl", f"--device={device}", "--set-ctrl", f"{control}={value}"]
# #         subprocess.run(cmd, check=True)

# #     videoDevices = sorted(glob.glob("/dev/video*"))
# #     cam = None
# #     selectedDevice = None

# #     for dev in videoDevices:
# #         index = int(dev.replace("/dev/video", ""))
# #         print(f"Prøver {dev} ...")
        
# #         with camera_lock:
# #             cap = cv2.VideoCapture(index)
# #             cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# #         if cap.isOpened():
# #             print(f"Leder efter kamera på: {dev}")
# #             cam = cap
# #             selectedDevice = dev
# #             break
# #         else:
# #             cap.release()

# #     if not cam:
# #         print("Ingen kamera fundet")
# #         exit(1)

# #     print("Bruger kamera:", selectedDevice)
    
# #     # Indstil kameraet
# #     auto_gain = getCameraControl(selectedDevice, "gain")
# #     setCameraControl(selectedDevice, "auto_exposure", 1)
# #     setCameraControl(selectedDevice, "exposure_time_absolute", 200)
# #     setCameraControl(selectedDevice, "gain", 10)  # Justér efter behov
# #     setCameraControl(selectedDevice, "focus_automatic_continuous", 0)
# #     setCameraControl(selectedDevice, "focus_absolute", 0)
# #     setCameraControl(selectedDevice, "white_balance_automatic", 1)

# #     upperLimitBlue = np.array([123, 252, 150], np.uint8)
# #     lowerLimitBlue = np.array([103, 92, 5], np.uint8)
# #     upperLimitYellow = np.array([35,255,255], np.uint8)
# #     lowerLimitYellow = np.array([13,100,30], np.uint8)
# #     return cam, upperLimitBlue, lowerLimitBlue, upperLimitYellow, lowerLimitYellow

# # # Maskering og billedebehandling
# # def Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower):
# #     frame = cv2.GaussianBlur(frame, (5, 5), 0)
# #     HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# #     maskBlue = cv2.inRange(HSV, blueLower, blueUpper)
# #     maskYellow = cv2.inRange(HSV, yellowLower, yellowUpper)

# #     maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_OPEN, kernel, iterations=2)
# #     maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_CLOSE, kernel, iterations=2)
# #     maskBlue = cv2.medianBlur(maskBlue, 5)

# #     maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_OPEN, kernel, iterations=2)
# #     maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_CLOSE, kernel, iterations=2)
# #     maskYellow = cv2.medianBlur(maskYellow, 5)
# #     return HSV, maskBlue, maskYellow

# # # Funktion til at gemme billeder sekventielt
# # def capture_binary_image_from_video(output_dirs=None, threshold_value=127):
# #     if output_dirs is None:
# #         output_dirs = {'a': 'classA', 'b': 'classB', 'c': 'classC', 'd': 'classD'}
    
# #     for folder in output_dirs.values():
# #         if not os.path.exists(folder):
# #             os.makedirs(folder)

# #     # Start video capture
# #     cap = cv2.VideoCapture(0)  # Skift til din ønskede videokilde, hvis nødvendigt
# #     if not cap.isOpened():
# #         print("Kunne ikke åbne kameraet.")
# #         return

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             print("Fejl ved indlæsning af billede.")
# #             break

# #         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         _, binary_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

# #         cv2.imshow('Binary Video Feed', binary_frame)

# #         key = cv2.waitKey(1) & 0xFF

# #         # Hvis 'q' trykkes, afslut programmet
# #         if key == ord('q'):
# #             break

# #         # Håndter tasterne 'a', 'b', 'c', 'd' for at gemme billeder
# #         if key in [ord('a'), ord('b'), ord('c'), ord('d')]:
# #             folder = output_dirs[chr(key)]  # Mapperne bestemmes af 'a', 'b', 'c', 'd'

# #             # Find det næste ledige billede i den mappe
# #             file_list = os.listdir(folder)
# #             file_numbers = [int(f.split('.')[0]) for f in file_list if f.split('.')[0].isdigit()]
# #             next_num = max(file_numbers, default=0) + 1

# #             image_filename = os.path.join(folder, f"{next_num}.jpg")
# #             cv2.imwrite(image_filename, binary_frame)
# #             print(f"Billede gemt som {image_filename}")

# #     cap.release()
# #     cv2.destroyAllWindows()

# # # Hovedfunktionen hvor alt samles
# # def main():
# #     stopEvent = threading.Event()
# #     device, scale_z, pixelFormat_initial, operating_mode_initial, exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial = CreateDevice()

# #     def HeliosThread(device, scale_z):
# #         global latestDistanceFrame
# #         try:
# #             with device.start_stream(10):
# #                 while not stopEvent.is_set():
# #                     heatmap, depth = HeliosRunning(device, scale_z)
# #                     if heatmap is None:
# #                         continue
# #                     with latestDistanceFrameLock:
# #                         latestDistanceFrame = (heatmap, depth)
# #         except Exception as e:
# #             print(f"Fejl i HeliosThread: {e}")

# #     t = threading.Thread(target=HeliosThread, args=(device, scale_z), daemon=True)
# #     t.start()

# #     # Setup kamera
# #     cap, blueUpper, blueLower, yellowUpper, yellowLower = Setup()

# #     prev_time = time.time()

# #     while True:
# #         ret, frame = cap.read()
# #         current_time = time.time()
# #         fps = 1 / (current_time - prev_time)
# #         prev_time = current_time
# #         if not ret:
# #             print("Can't read camera, ending stream")
# #             break
# #         if latestDistanceFrame is None:
# #              print("Vent på data fra Helios")
# #         else:
# #              _, depth = latestDistanceFrame
# #              cv2.imshow("Helios depthmap", depth)

# #         frame = WarpFrame(frame, depth)

# #         # Maskering
# #         HSV, maskBlue, maskYellow = Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower)
# #         cv2.imshow("maskBlue", maskBlue)
# #         cv2.imshow("maskYellow", maskYellow)

# #         # Kald billedtagning funktionen
# #         capture_binary_image_from_video()

# #         key = cv2.waitKey(1) & 0xFF
# #         if key == ord('q'):
# #             break

# #     stopEvent.set()
# #     t.join()

# #     cap.release()
# #     cv2.destroyAllWindows()

# #     time.sleep(0.1)

# #     try:
# #         HeliosEnd(device, pixelFormat_initial, operating_mode_initial, exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial)
# #     except Exception as e:
# #         print("Fejl under HeliosEnd", e)

# # if __name__ == "__main__":
# #     main()




# # # import cv2
# # # import os



# # # import numpy as np
# # # import cv2
# # # import time
# # # import subprocess
# # # import math
# # # import glob
# # # from helios_create_image import CreateDevice, HeliosRunning, HeliosEnd
# # # import threading
# # # from threading import Lock
# # # kernel = np.ones([5,5], np.uint8)
# # # latestDistanceFrame = None 
# # # latestDistanceFrameLock = Lock()


# # # camera_lock = threading.Lock()
# # # # The setup function is where the camera is inititiated and the limits for color segmentation is defined in HSV
# # # def Setup():
# # #     def getCameraControl(device, name):
# # #         result = subprocess.run(
# # #             ["v4l2-ctl", f"--device={device}", "--get-ctrl", name],
# # #             capture_output=True, text=True
# # #         )
# # #         return result.stdout.strip()

# # #     def setCameraControl(device, control, value):
# # #         cmd = ["v4l2-ctl", f"--device={device}", "--set-ctrl", f"{control}={value}"]
# # #         subprocess.run(cmd, check=True)


# # #     videoDevices = sorted(glob.glob("/dev/video*"))
# # #     cam = None
# # #     selectedDevice = None

# # #     for dev in videoDevices:
# # #         index = int(dev.replace("/dev/video", ""))
# # #         print(f"Prøver {dev} ...")
        
# # #         with camera_lock:
# # #             cap = cv2.VideoCapture(index)
# # #             cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# # #         if cap.isOpened():
# # #             print(f"Leder efter kamera på: {dev}")
# # #             cam = cap
# # #             selectedDevice = dev
# # #             break
# # #         else:
# # #             cap.release()

# # #     if not cam:
# # #         print("Ingen kamera fundet")
# # #         exit(1)

# # #     print("bruger kamera:", selectedDevice)
    
# # #     # Læs gain i auto mode
# # #     auto_gain = getCameraControl(selectedDevice, "gain")

# # #     #manually setting the camera setings
# # #     setCameraControl(selectedDevice, "auto_exposure",1)
# # #     setCameraControl(selectedDevice, "exposure_time_absolute", 200)
# # #     setCameraControl(selectedDevice, "gain", 10)#250)
# # #     setCameraControl(selectedDevice, "focus_automatic_continuous",0)
# # #     setCameraControl(selectedDevice, "focus_absolute", 0)
# # #     setCameraControl(selectedDevice, "white_balance_automatic", 1)

# # #     upperLimitBlue = np.array([123, 252, 150], np.uint8)
# # #     lowerLimitBlue = np.array([103, 92, 5], np.uint8)
# # #     upperLimitYellow = np.array([35,255,255], np.uint8)
# # #     lowerLimitYellow = np.array([13,100,30], np.uint8)
# # #     return cam, upperLimitBlue, lowerLimitBlue, upperLimitYellow, lowerLimitYellow

# # # def WarpFrame(frame, depth):
   
# # #     matrix4 = np.array([[ 8.43114000e-01, -3.84176124e-02,  2.83616563e+01],
# # #                         [-2.15365538e-02,  8.45311445e-01,  4.68435043e+01],
# # #                           [-1.13793182e-04, -9.33131759e-05,  1.00000000e+00]])
    
# # #     #warp billede2 på billede1
# # #     height, width = depth.shape[:2]
# # #     warpedFrame = cv2.warpPerspective(frame, matrix4, (width, height))
# # #     # cv2.imwrite("WarpedImg2.png", warped_img2)


# # #     return warpedFrame

# # # # The masking function is where all the preprocessing and masking is performed
# # # # A gaussian blur is added to remove noise and smooth the frame
# # # # The frame is then converted to HSV
# # # # A mask is created using the limits from Setup() and then converted to a mask so only the colors in the limit is found
# # # # Opening and closing morphology is performed on the masks to further remove noise
# # # # Then adding a median blur to EVEN further decrease noise. This might not be necesarry 
# # # def Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower):
# # #         frame = cv2.GaussianBlur(frame, (5, 5), 0)
# # #         HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# # #         maskBlue = cv2.inRange(HSV, blueLower, blueUpper)
# # #         maskYellow = cv2.inRange(HSV, yellowLower, yellowUpper)

# # #         maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_OPEN, kernel, iterations=2)
# # #         maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_CLOSE, kernel, iterations=2)
# # #         maskBlue = cv2.medianBlur(maskBlue, 5)

# # #         maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_OPEN, kernel, iterations=2)
# # #         maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_CLOSE, kernel, iterations=2)
# # #         maskYellow = cv2.medianBlur(maskYellow, 5)
# # #         return HSV, maskBlue, maskYellow



# # # def capture_binary_image_from_video(video_source=0, output_dirs=None, threshold_value=127):
# # #     # Start video capture (0 betyder at tage kameraet, kan ændres til en videofil sti)
# # #     cap = cv2.VideoCapture(video_source)

# # #     if not cap.isOpened():
# # #         print("Kunne ikke åbne videoen.")
# # #         return

# # #     if output_dirs is None:
# # #         output_dirs = {'a': 'classA', 'b': 'classB', 'c': 'classC', 'd': 'classD'}
    
# # #     # Skab mapper, hvis de ikke eksisterer
# # #     for folder in output_dirs.values():
# # #         if not os.path.exists(folder):
# # #             os.makedirs(folder)

# # #     # Indstil at vise video-feedet
# # #     while True:
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             print("Fejl ved indlæsning af billede.")
# # #             break

# # #         # Konverter farvebillede til gråtone
# # #         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # #         # Anvend tærskeloperation for at få et binært billede (sort-hvid)
# # #         _, binary_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

# # #         # Vis det binære billede
# # #         cv2.imshow('Binary Video Feed', binary_frame)

# # #         # Venter på tastetryk
# # #         key = cv2.waitKey(1) & 0xFF

# # #         # Hvis 'q' trykkes, afslut programmet
# # #         if key == ord('q'):
# # #             break

# # #         # Håndter tasterne 'a', 'b', 'c', 'd'
# # #         if key == ord('a') or key == ord('b') or key == ord('c') or key == ord('d'):
# # #             folder = output_dirs[chr(key)]  # Mapperne bestemmes af 'a', 'b', 'c', 'd'

# # #             # Find det næste ledige billede i den mappe
# # #             file_list = os.listdir(folder)
# # #             file_numbers = [int(f.split('.')[0]) for f in file_list if f.split('.')[0].isdigit()]
# # #             next_num = max(file_numbers, default=0) + 1

# # #             # Gem billede med det næste nummer i den valgte mappe
# # #             image_filename = os.path.join(folder, f"{next_num}.jpg")
# # #             cv2.imwrite(image_filename, binary_frame)
# # #             print(f"Billede gemt som {image_filename}")

# # #     # Frigiv videoobjektet og luk alle vinduer
# # #     cap.release()
# # #     cv2.destroyAllWindows()


# # # latestDistanceFrame = None

# # # # This is where everything comes together 
# # # def main():
# # #     stopEvent = threading.Event()
# # #     device ,scale_z, pixelFormat_initial, operating_mode_initial,  exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial = CreateDevice()



# # #     def HeliosThread(device, scale_z):
# # #         global latestDistanceFrame
# # #         try:
# # #             with device.start_stream(10):
# # #                 while not stopEvent.is_set():
# # #                     heatmap, depth = HeliosRunning(device, scale_z)
# # #                     if heatmap is None:
# # #                         continue
# # #                     with latestDistanceFrameLock:
# # #                         latestDistanceFrame = (heatmap, depth)
# # #         except Exception as e:
# # #             print(f"Fejl i HeliosThread: {e}")
# # #             #device ,scale_z, pixelFormat_initial, operating_mode_initial,  exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial = CreateDevice()



    
# # #     t = threading.Thread(target=HeliosThread, args=(device, scale_z), daemon=True)
# # #     t.start()

# # #     # The setup() is loaded in
# # #     cap, blueUpper, blueLower, yellowUpper, yellowLower = Setup()
# # #     # Time is used to calculate FPS. (might not be necesarry to calculate for each frame, might be enough to display once a second)
    
# # #     prev_time = time.time()


# # #     while True:
# # #         ret, frame = cap.read() # Get the frame from the camera
# # #         current_time = time.time() #more fps
# # #         fps = 1 / (current_time - prev_time)
# # #         prev_time = current_time
# # #         if not ret: # If there are is no camera
# # #             print("Can't read camera, ending stream")
# # #             break
# # #         if latestDistanceFrame is None:
# # #              print("Vent på data fra Helios")
# # #         else:
# # #              _, depth = latestDistanceFrame
# # #              cv2.imshow("Helios depthmap", depth)

# # #         frame = WarpFrame(frame, depth)

# # #         # Get the masks created in Masking()
# # #         HSV, maskBlue, maskYellow = Masking(frame, blueUpper, blueLower, yellowUpper, yellowLower)
# # #         # uses the masks in found Contours to find the locations of objects

# # #         capture_binary_image_from_video() 

# # #         #cv2.imshow("frame", frame)
# # #         cv2.imshow("maskBlue", maskBlue)
# # #         cv2.imshow("maskYellow", maskYellow)

# # #         #cv2.imshow("Frame with boxes and edges", maskBlue)
# # #         # Show the combined mask and frame with bboxes
# # #         #cv2.imshow("mask", mask)
# # #         # prints FPS
# # #         #print(f"FPS: {fps}")
# # #         #if cv2.waitKey(1) == ord('q'):
# # #             #break
# # #         key = cv2.waitKey(1) & 0xFF
# # #         if key == ord('q'):
# # #             break
# # #         elif key == ord('s'):
# # #             # Gem billedet som et unikt billede med tid som filnavn
# # #             timestamp = time.strftime("%Y%m%d_%H%M%S")
# # #             image_filename = f"captured_imageB_{timestamp}.jpg"
# # #             cv2.imwrite(image_filename, maskBlue)
# # #             print(f"Billede gemt som {image_filename}")


# # #     stopEvent.set()
# # #     t.join()

# # #     cap.release()
# # #     cv2.destroyAllWindows()

# # #     time.sleep(0.1)

# # #     try:
# # #         HeliosEnd(device, pixelFormat_initial, operating_mode_initial,  exposure_time_initial, conversion_gain_initial, image_accumulation_initial, spatial_filter_initial, confidence_threshold_initial)
# # #     except Exception as e:
# # #         print("Fejl under HeliosEnd", e)

# # # if __name__ == "__main__":
# # #     main()
    


