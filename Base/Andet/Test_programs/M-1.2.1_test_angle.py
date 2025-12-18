
import serial
import time
import threading
import queue

PORT = 'COM10'
BAUD = 115200
WRITE_INTERVAL = 0.05  # sekunder mellem skrivninger (tuning efter behov)

updates = queue.Queue()
stop_evt = threading.Event()

def input_thread():
    print("Skriv en værdi mellem 0 og 180 og tryk Enter. (q/quit for at stoppe)")
    while not stop_evt.is_set():
        try:
            line = input()
        except EOFError:
            break
        except KeyboardInterrupt:
            stop_evt.set()
            break

        line = line.strip()
        if not line:
            continue

        if line.lower() in {"q", "quit", "exit", "stop"}:
            stop_evt.set()
            break

        try:
            val = int(line)
            if 0 <= val <= 180:
                updates.put(val)
            else:
                print("Ugyldig værdi: skal være 0–180.")
        except ValueError:
            print(f"Kunne ikke parse heltal: '{line}'")

def parse_line(line: str):
    """Forventet format: 'angle,count' → returnér (angle, count) eller None."""
    if "," not in line:
        print(f"skrewed line: {line}")
        return None
    try:
        angle_str, count_str = line.split(',', 1)
        angle = float(angle_str)
        count = float(count_str)
        return angle, count
    except ValueError:
        print(f"Bad line: {line}")
        return None

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except serial.SerialException as e:
        print(f"Kunne ikke åbne {PORT}: {e}")
        return

    # Start input-tråden
    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    # Startværdi
    current_value = 90
    zeroFound = False
    timestamp = None
    last_write = 0.0

    print(f"Starter med {current_value}. Skriv ny værdi når som helst.")

    try:
        while not stop_evt.is_set():
            # Hent nye brugerinput uden at blokere
            while not updates.empty():
                current_value = updates.get_nowait()

            # Skriv til Arduino med passende cadence
            now = time.time()
            if now - last_write >= WRITE_INTERVAL:
                ser.write(bytes([current_value]))   # 0–180 er ok som én byte
                print(f"\rmessageOutToArduino: {current_value:<3}", end="")
                last_write = now

            # Læs linje fra Arduino
            raw = ser.readline()
            if raw:
                line = raw.decode(errors='ignore').strip()
                parsed = parse_line(line)
                if parsed is not None:
                    angle, count = parsed
                    if count < 1 and not zeroFound:
                        zeroFound = True
                        timestamp = time.time()
                        print(f"\nAngle: {angle}  Count: {count}  @ {time.strftime('%H:%M:%S')}")

            time.sleep(0.005)  # små pauser for CPU-venlighed

    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        t.join(timeout=1.0)
        try:
            ser.flush()
        except Exception:
            pass
        ser.close()
        print("\nStopper pænt...")

if __name__ == "__main__":
    main()
