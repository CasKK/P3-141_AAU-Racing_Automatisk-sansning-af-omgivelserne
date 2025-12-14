#include <Arduino.h>
#include <Servo.h>
#include "ICM20600.h"
#include <Wire.h>
#include <math.h>

// Encoder and Wheel Parameters

#define ENCODER_PIN 2
#define PULSES_PER_REV 100       // Counting both edges (50 holes × 2)

const float WHEEL_CIRCUMFERENCE = 0.5776f;  // meters
const bool debug = false;
unsigned long lastCalcTime = 0;
unsigned long lastPulseTime = 0;

volatile long encoderCount = 0;
long lastEncoderCount = 0;
int val = 90;
float rpm = 0.0f;
float speed_mps = 0.0f;

Servo esc;
Servo steering;

// Gyro variables
unsigned long t_prev = 0;
float angle = 0.0f;
float gyro_offset = 0.0f;
unsigned long time;
bool newData = false;
float wheelAngle;
float wheelAngleTemp;
float desiredSpeed = 0.4f;  // m/s
unsigned long lastTime = 0;
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
  const float Kp = 0.2f;           // Reduced gain for stability
  const float deadband = 3.0f;     // RPM deadband to avoid chasing noise
  const int min_us = 1600;        // Spin start point
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
  if (debug == true){  
    Serial.print(">P_control Output:");
    Serial.println(output);
  }

  return output;
}



void ReadSerial() {
    if (Serial.available() > 0 && newData == false) {
        wheelAngleTemp = Serial.read();   // 0–255 direkte
        wheelAngleTemp = constrain(wheelAngleTemp, 0, 180);
        newData = true;
    }
}



// -----------------------------------------------------------
// Setup
// -----------------------------------------------------------
void setup() {
  Wire.begin();
  Wire.setClock(400000);
  icm20600.initialize();
  delay(10);
  Serial.begin(115200);
    
   // Kalibrér gyro offset
  //Serial.println("Calibrating gyro...");
  int sum = 0;
  for (int i = 0; i < 1000; i++) {
    sum += icm20600.getGyroscopeZ();
    delay(2);
  }
  gyro_offset = sum / 1000.0f;
  //Serial.print("Gyro offset: ");
  //Serial.println(gyro_offset);
  t_prev = micros();

  pinMode(ENCODER_PIN, INPUT);

  esc.attach(11, 1000, 2000);
  steering.attach(10);
  steering.write(val);
  delay(500);
  //Serial.println("Arduino Is ready :)");
  delay(10000);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN), encoderISR, CHANGE);
}

float MotorToServo(float x) {
  float x2 = x * x;
  float x3 = x2 * x;
  float servoAngle =  0.00123378679436315 * x3 - 0.34492174790361900 * x2 + 34.23770218227510000 * x - 1095.74990424557000000;
  int servoAngleInt = (int)round(servoAngle);
  return constrain(servoAngleInt, 0, 180);
}

// -----------------------------------------------------------
// Main Loop
// -----------------------------------------------------------
const float GYRO_SENS = 16.4f;
float Gyroscope(){
  unsigned long t_now = micros();
  float dt = (t_now - t_prev) * 1e-6f;
  t_prev = t_now;

  float gyroRawZ = icm20600.getGyroscopeZ() - gyro_offset;
  float gyroZ_dps = gyroRawZ / GYRO_SENS;   // convert to deg/s

  angle += gyroZ_dps * dt;  
  if (debug == true){
    Serial.print(">time:");
    Serial.println(micros() - time);
    Serial.print(">Angle:");
    Serial.println(angle);
  }

  return angle;
}

void Encoder(){
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
    // throttle_us = 1000; // Neutral for testing
    esc.writeMicroseconds(throttle_us);

    if (debug == true){
      //Debug output
      Serial.print(">encoderCount:");
      Serial.println(snapshot);
      Serial.print(">Rounds:");
      Serial.println((float)snapshot / PULSES_PER_REV);
      Serial.print(">RPM:");
      Serial.println(rpm);
      Serial.print(">m/s:");
      Serial.println(speed_mps);
    }
  }
}

void ServoMotor(){
  ReadSerial();
  if (newData){
    if (wheelAngleTemp > 10){
      wheelAngle = wheelAngleTemp;
      if (wheelAngle >= 0 && wheelAngle <= 180) {             // sanity check
        steering.write(MotorToServo(wheelAngle));
        if (debug == true){
          Serial.print("Servo angle set to: ");
          Serial.println(MotorToServo(wheelAngle));
        }
      }
    } 
    newData = false;
  }
}

void loop() {
  time = micros();
  ServoMotor();
  Encoder();
  float angleValue = Gyroscope();
  if (time - lastTime >= 33000){
    Serial.print(angleValue);
    Serial.print(", ");
    Serial.println(encoderCount);
    lastTime = time;
  }
}