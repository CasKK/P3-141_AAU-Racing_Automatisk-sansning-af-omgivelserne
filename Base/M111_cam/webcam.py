import cv2
import subprocess

def setCameraControl(device, control, value):
    cmd = ["v4l2-ctl", f"--device={device}", "--set-ctrl", f"{control}={value}"]
    subprocess.run(cmd, check=True)

def getCameraControl(name):
    result = subprocess.run(
        ["v4l2-ctl", "--get-ctrl", name],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# Prøv alle video noder 0-3
for index in range(4):
    device_path = f"/dev/video{index}"
    print(f"Prøver /dev/video{device_path} ...")
    cap = cv2.VideoCapture(index)
    
    # Tving MJPEG, da mange USB-kameraer på Jetson kræver det
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    if cap.isOpened():
        print(f"Kamera fundet på /dev/video{index}")
        selected_device = device_path
        break
    else:
        cap.release()
        cap = None

else:
    print("Ingen kameraer fundet på /dev/video0-3")
    exit(1)


# Læs gain i auto mode
auto_gain = getCameraControl("gain")

# Læs frames og vis
while True:
    setCameraControl(selected_device, "auto_exposure",1)
    setCameraControl(selected_device, "exposure_time_absolute", 200)
    setCameraControl(selected_device, "gain", 10)#250)
    setCameraControl(selected_device, "focus_automatic_continuous",0)
    setCameraControl(selected_device, "focus_absolute", 20)
    setCameraControl(selected_device, "white_balance_automatic", 1)

    ret, frame = cap.read()
    if not ret:
        print("Fejl: kunne ikke læse frame")
        break

    cv2.imshow("Webcam", frame)

    # Tryk 'q' for at lukke
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
