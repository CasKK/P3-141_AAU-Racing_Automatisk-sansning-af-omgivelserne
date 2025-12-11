import serial
import time
from multiprocessing import Queue
import csv
import json
import datetime

zeroFound = False

def run(read_queue, write_queue):
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
    time.sleep(9)  # allow Arduino reset
    with open("seriallog", "a", newline="") as f:
        writer = csv.writer(f)
        while True:
            # --- READ ---
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                if "," in line:
                    try:
                        angle_str, count_str = line.split(',')
                        angle = float(angle_str) 
                        count = float(count_str)
                        if count < 1:
                            zeroFound = True
                        if zeroFound:
                            if read_queue.qsize() >= 5:
                                try:
                                    read_queue.get_nowait()  # remove oldest item
                                except:
                                    pass
                            read_queue.put((angle, count))
                            print("Angle:", angle, "Count:", count)
                            timestamp = time.time()
                            writer.writerow([timestamp, angle, count])
                            f.flush()
                    except ValueError:
                        print("Bad line:", line)
                else:
                    print("skrewed line:", line)

            # --- WRITE ---
            while not write_queue.empty():
                message = write_queue.get()
                print(f"messageOutToArduino: {message}")
                message = bytes([message])
                ser.write(message)

            time.sleep(0.001)