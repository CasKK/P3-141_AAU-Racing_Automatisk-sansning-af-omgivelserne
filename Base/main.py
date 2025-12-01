from multiprocessing import Process, Queue
import time
from M111_cam import ConeDetectionv3
from M113_position import Coordinates
from M121_logic import M121_logic

if __name__ == "__main__":

    q_m111_to_m113 = Queue()
    q_m113_to_m121 = Queue()
    
    m111 = Process(target=ConeDetectionv3.run, args=(q_m111_to_m113,))
    m113 = Process(target=Coordinates.run, args=(q_m111_to_m113, q_m113_to_m121))
    m121 = Process(target=M121_logic.run, args=(q_m113_to_m121,))

    m111.start()
    m113.start()
    m121.start()

    print("CTRL + C = kEYboArDInTerRuPt")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("kIlLiNg!")
        m111.terminate()
        m113.terminate()
        m121.terminate()
