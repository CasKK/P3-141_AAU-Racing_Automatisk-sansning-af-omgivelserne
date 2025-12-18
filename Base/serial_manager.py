import serial
import time
from multiprocessing import Queue
import csv
import json
import datetime


def run(read_queue, write_queue): #Run Function for Serial Manager
    
    zeroFound = False
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1) # Open the serial port (Linux))
    time.sleep(9)  # allow Arduino reset
    with open("seriallog", "a", newline="") as f: # Open the serial log file for appending
        writer = csv.writer(f) #Initialize CSV writer
        while True: # Read and write loop
            # --- READ ---
            line = ser.readline().decode(errors='ignore').strip() # Read a line from the serial port
            if line: # Parse the line if it's not empty
                if "," in line:
                    try:
                        angle_str, count_str = line.split(',')
                        angle = float(angle_str) 
                        count = float(count_str)
                        if count < 1: # Account for initial zero readings to prevent logging invalid data
                            zeroFound = True
                        if zeroFound:
                            if read_queue.qsize() >= 5: # Limit queue size to prevent falling behind
                                try:
                                    read_queue.get_nowait()  # remove oldest item
                                except:
                                    pass
                            read_queue.put((angle, count)) # Put parsed data into read queue
                            print("Angle:", angle, "Count:", count)
                            timestamp = time.time()
                            writer.writerow([timestamp, angle, count])
                            f.flush()
                    except ValueError:
                        print("Bad line:", line)
                else:
                    print("skrewed line:", line)

            # --- WRITE ---
            while not write_queue.empty(): # Send all messages in the write queue
                message = write_queue.get()
                print(f"messageOutToArduino: {message}")
                message = bytes([message]) # Convert to bytes
                ser.write(message)

            time.sleep(0.001) # Small delay to prevent CPU overload