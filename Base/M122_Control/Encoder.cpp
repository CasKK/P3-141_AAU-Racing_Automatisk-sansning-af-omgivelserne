
#include <Arduino.h>
#include <Servo.h>
#include "ICM20600.h"
#include <Wire.h>
#include <math.h>

// Encoder and Wheel Parameters

#define ENCODER_PIN 2
#define PULSES_PER_REV 100       // Counting both edges (50 holes × 2)

const float WHEEL_CIRCUMFERENCE = 0.5776f;  // meters

unsigned long lastCalcTime = 0;
unsigned long lastPulseTime = 0;

volatile long encoderCount = 0;
long lastEncoderCount = 0;

float rpm = 0;
float speed_mps = 0;

Servo esc;

// Gyro variables
unsigned long t_prev = 0;
float angle = 0.0f;
float gyro_offset = 0.0f;


ICM20600 icm20600(true);


// -----------------------------------------------------------
// Interrupt Service Routine
// -----------------------------------------------------------
void encoderISR() {
  unsigned long now = micros();
  if (now - lastPulseTime > 1000) { // debounce: ignore pulses <1ms apart
    encoderCount++;
    lastPulseTime = now;
  }
}

// -----------------------------------------------------------
// Proportional Speed Controller (returns microseconds)
// -----------------------------------------------------------
int P_control(float desired_speed, float current_rpm) {
  const float Kp = 0.2;           // Reduced gain for stability
  const float deadband = 3.0;     // RPM deadband to avoid chasing noise
  const int min_us = 1250;        // Spin start point
  const int max_us = 2000;

  static int lastOutput = min_us + 50;

  float desired_rpm = (desired_speed * 60.0f) / WHEEL_CIRCUMFERENCE;
  float error = desired_rpm - current_rpm;

  if (abs(error) < deadband) {
    return lastOutput;
  }

  float throttle_delta = Kp * error; // No extra scaling

  int output = lastOutput + throttle_delta;

  output = constrain(output, min_us, max_us);
  lastOutput = output;

  Serial.print(">P_control Output:");
  Serial.println(output);
  return output;
}

// -----------------------------------------------------------
// Setup
// -----------------------------------------------------------
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


  pinMode(ENCODER_PIN, INPUT);

  esc.attach(11, 1000, 2000);

  delay(500);

  // ESC calibration
  esc.writeMicroseconds(2000);
  delay(1500);
  esc.writeMicroseconds(1000);
  delay(1500);
  delay(2000);

  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN), encoderISR, CHANGE);
}

// -----------------------------------------------------------
// Main Loop
// -----------------------------------------------------------
void loop() {
  float desiredSpeed = 0.8f;  // m/s

  if (millis() - lastCalcTime >= 100) {
    noInterrupts();
    long snapshot = encoderCount;
    interrupts();

    long countDiff = snapshot - lastEncoderCount;
    lastEncoderCount = snapshot;

    unsigned long now_ms = millis();
    unsigned long dt_ms = now_ms - lastCalcTime;

    float newRPM = ((float)countDiff / PULSES_PER_REV) * (60000.0f / dt_ms);

    float alpha = 0.3;
    rpm = (alpha * newRPM) + (1 - alpha) * rpm;

    speed_mps = (rpm * WHEEL_CIRCUMFERENCE) / 60.0f;

    lastCalcTime = now_ms;

    // Update throttle only at RPM refresh
    int throttle_us = P_control(desiredSpeed, rpm);
    esc.writeMicroseconds(throttle_us);

    // Debug output
    Serial.print(">encoderCount:");
    Serial.println(snapshot);
    Serial.print(">Rounds:");
    Serial.println((float)snapshot / PULSES_PER_REV);
    Serial.print(">RPM:");
    Serial.println(rpm);
    Serial.print(">m/s:");
    Serial.println(speed_mps);
  }

  // float totalAngle = 0.0f;
  // unsigned long t_now = 0;
  // unsigned long time = micros();
  // for(int i = 0; i <= 15; i++){
  //     t_now = micros();       
  //     float dt = (t_now - t_prev)/ 1e6;
  //     t_prev = t_now;
  //     float gyroZ = icm20600.getGyroscopeZ() - gyro_offset; 
  //     angle += gyroZ * dt; 
  //     totalAngle += angle;
  // }
  // Serial.print(">time:");
  // Serial.println(micros() - time);
  // Serial.print(">Angle:");
  // Serial.println(M_PI/180 * (totalAngle/15), 2);
}