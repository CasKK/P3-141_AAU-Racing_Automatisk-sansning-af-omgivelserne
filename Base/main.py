from multiprocessing import Process, Queue
import time
from M111_cam import ConeDetectionv5
from M113_position import Coordinates_test3
from M121_logic import M121_logic
import serial_manager

if __name__ == "__main__":

    q_m111_to_m113 = Queue()
    q_m113_to_m121 = Queue()
    q_serial_read = Queue() # Serial que til modtagelse "kun på m113"
    q_serial_write = Queue() # Serial que til afsendelse "kun på m121"
    
    m111 = Process(target=ConeDetectionv5.run, args=(q_m111_to_m113,))
    m113 = Process(target=Coordinates_test3.run, args=(q_m111_to_m113, q_m113_to_m121, q_serial_read)) #
    m121 = Process(target=M121_logic.run, args=(q_m113_to_m121, q_serial_write)) #
    serial = Process(target=serial_manager.run, args=(q_serial_read, q_serial_write))

    m111.start()
    m113.start()
    m121.start()
    serial.start()

    print("CTRL + C = kEYboArDInTerRuPt")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("kIlLiNg!") 
        m111.terminate()
        m113.terminate()
        m121.terminate()
        serial.terminate()
