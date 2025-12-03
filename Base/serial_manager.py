import serial
import time
from multiprocessing import Queue

def serial_manager(read_queue, write_queue):
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
    time.sleep(2)  # allow Arduino reset

    while True:
        # --- READ ---
        if ser.in_waiting:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    read_queue.put(line)
            except:
                pass

        # --- WRITE ---
        while not write_queue.empty():
            message = write_queue.get()
            ser.write((message + '\n').encode('utf-8'))

        time.sleep(0.001)