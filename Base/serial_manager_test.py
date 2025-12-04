import serial
import time
from multiprocessing import Queue

def run(read_queue, write_queue):
    #ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
    time.sleep(2)  # allow Arduino reset

    lastTime = 0
    encoderCounts = 0
    degrees = 0

    while True:
        # --- READ ---
        # if ser.in_waiting:
        #     try:
        #         line = ser.readline().decode('utf-8', errors='ignore').strip()
        #         if line:
        #             read_queue.put(line)
        #     except:
        #         pass
        if lastTime + 0.01 < time.time():
            read_queue.put((degrees, encoderCounts))
            degrees += 0.1
            encoderCounts += 10
            lastTime = time.time()

        # --- WRITE ---
        while not write_queue.empty():
            message = write_queue.get()
            #ser.write((message + '\n').encode('utf-8'))
            print(f"messageOutToArduino: {message}")

        time.sleep(0.001)