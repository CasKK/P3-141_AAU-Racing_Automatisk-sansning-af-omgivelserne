import serial 

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)


while True:
    message = int(input())
    print(f"messageOutToArduino: {message}")
    message = bytes([message])
    ser.write(message)
