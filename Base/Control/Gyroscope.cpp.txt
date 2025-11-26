#include "ICM20600.h"
#include <Wire.h>
#include <math.h>
unsigned long t_prev = 0;
float angle = 0.0f;
float gyro_offset = 0.0f;


ICM20600 icm20600(true);

void setup() {
    // join I2C bus (I2Cdev library doesn't do this automatically)
    Wire.begin();
    Wire.setClock(400000);
    icm20600.initialize();
    delay(10);
    Serial.begin(115200);
    
    // Kalibrér gyro offset
    Serial.println("Calibrating gyro...");
    long sum = 0;
    for (int i = 0; i < 1000; i++) {
        sum += icm20600.getGyroscopeZ();
        delay(2);
    }
    gyro_offset = sum / 1000.0f;
    Serial.print("Gyro offset: "); Serial.println(gyro_offset);
    t_prev = micros();
}
void loop() {
    float totalAngle = 0.0f;
    unsigned long t_now = 0;
    unsigned long time = micros();
    for(int i = 0; i <= 15; i++){
        t_now = micros();       
        float dt = (t_now - t_prev)/ 1e6;
        t_prev = t_now;
        float gyroZ = icm20600.getGyroscopeZ() - gyro_offset; 
        angle += gyroZ * dt; 
        totalAngle += angle;
    }
    Serial.print("   time: ");
    Serial.println(micros() - time);
    Serial.print("Angle : ");
    Serial.print(M_PI/180 * (totalAngle/15), 2);
}

