import serial
import time
from multiprocessing import Queue
import csv
import json
import datetime


def run(read_queue, write_queue):
    ser = serial.Serial('com5', 115200, timeout=0.1)
    time.sleep(5)  # allow Arduino reset
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
                # ser.write((message + '\n').encode('utf-8'))
                print(f"messageOutToArduino: {message}")

            time.sleep(0.001)