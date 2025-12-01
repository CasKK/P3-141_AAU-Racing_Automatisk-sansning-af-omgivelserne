from smbus2 import SMBus
import time

# I2C-adresser
ICM_ADDR = 0x69  # ICM20600
AK_ADDR = 0x0C   # AK09918

bus = SMBus(1)  # /dev/i2c-1

# --- ICM20600 registre ---
ICM_WHO_AM_I = 0x75
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
PWR_MGMT_1 = 0x6B
USER_CTRL = 0x6A
I2C_MST_CTRL = 0x24
I2C_SLV0_ADDR = 0x25
I2C_SLV0_REG = 0x26
I2C_SLV0_CTRL = 0x27

# --- AK09918 registre ---
AK_WHO_AM_I = 0x00
AK_ST1 = 0x10
AK_HXL = 0x11
AK_CNTL2 = 0x31
AK_MODE_CONTINUOUS_100HZ = 0x08

# Helper til 16-bit signed
def read_word_i2c(addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    val = (high << 8) | low
    if val >= 0x8000:
        val = -((65535 - val) + 1)
    return val

# --- Init ICM20600 ---
bus.write_byte_data(ICM_ADDR, PWR_MGMT_1, 0x00)  # Wake up
time.sleep(0.1)

print("ICM WHO_AM_I:", hex(bus.read_byte_data(ICM_ADDR, ICM_WHO_AM_I)))
print("AK09918 WHO_AM_I (pre-reset):", hex(bus.read_byte_data(AK_ADDR, AK_WHO_AM_I)))

# --- Init AK09918 ---
bus.write_byte_data(AK_ADDR, AK_CNTL2, 0x01)  # Reset
time.sleep(0.1)
bus.write_byte_data(AK_ADDR, AK_CNTL2, AK_MODE_CONTINUOUS_100HZ)  # Continuous measurement

time.sleep(0.1)
print("AK09918 initialized.")

# --- Læs acc, gyro og mag ---
def read_accel():
    x = read_word_i2c(ICM_ADDR, ACCEL_XOUT_H)
    y = read_word_i2c(ICM_ADDR, ACCEL_XOUT_H + 2)
    z = read_word_i2c(ICM_ADDR, ACCEL_XOUT_H + 4)
    return (x, y, z)

def read_gyro():
    x = read_word_i2c(ICM_ADDR, GYRO_XOUT_H)
    y = read_word_i2c(ICM_ADDR, GYRO_XOUT_H + 2)
    z = read_word_i2c(ICM_ADDR, GYRO_XOUT_H + 4)
    return (x, y, z)

def read_mag():
    # Vent på data ready
    while not (bus.read_byte_data(AK_ADDR, AK_ST1) & 0x01):
        pass
    x = read_word_i2c(AK_ADDR, AK_HXL)
    y = read_word_i2c(AK_ADDR, AK_HXL + 2)
    z = read_word_i2c(AK_ADDR, AK_HXL + 4)
    return (x, y, z)

# --- Hovedloop ---
try:
    while True:
        acc = read_accel()
        gyro = read_gyro()
        mag = read_mag()
        print(f"Accel: {acc}, Gyro: {gyro}, Mag: {mag}")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Stopper...")
