import serial
import time
from multiprocessing import Queue

def run(read_queue, write_queue):
    ser = serial.Serial('com5', 115200, timeout=0.1)
    time.sleep(2)  # allow Arduino reset

    
    lastTime = 0
    encoderCounts = 0
    degrees = 0

    while True:
        # --- READ ---
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            if "," in line:
                try:
                    angle_str, count_str = line.split(',')
                    angle = float(angle_str)
                    count = float(count_str)
                    count = int(count)
                    read_queue.put((angle, count))
                    #print("Angle:", angle, "Count:", count)
                except ValueError:
                    print("Bad line:", line)
            else:
                print("skrewed line:", line)
        
        # if lastTime + 0.01 < time.time():
        #     read_queue.put((degrees, encoderCounts))
        #     degrees += 0.1
        #     encoderCounts += 10
        #     lastTime = time.time()

        # --- WRITE ---
        while not write_queue.empty():
            message = write_queue.get()
            #ser.write((message + '\n').encode('utf-8'))
            print(f"messageOutToArduino: {message}")

        time.sleep(0.001)