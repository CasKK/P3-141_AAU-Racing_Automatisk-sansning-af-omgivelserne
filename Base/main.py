from multiprocessing import Process, Queue
import time
#from M111_perception import M111_perception
from MR113_Position import Coordinates
from M121_logic import M121_logic

if __name__ == "__main__":

    q_m113_to_m121 = Queue()
    
    m113 = Process(target=Coordinates.run, args=(q_m113_to_m121,))
    m121 = Process(target=M121_logic.run, args=(q_m113_to_m121,))

    m113.start()
    m121.start()

    print("CTRL + C = kEYboArDInTerRuPt")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("kIlLiNg!")
        m113.terminate()
        m121.terminate()
