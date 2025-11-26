
#include <Arduino.h>
#include <Servo.h>

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
  Serial.begin(115200);
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
}

// int ENCODER_A = 3;

// volatile int encoder_Value = 0;
// int lastEncoderValue = 0;
// float encoderDiff = 0;
// float RPM_1 = 0;
// float RPM_2 = 0;
// float RPM_3 = 0;
// float RPM_4 = 0;
// float RPM = 0;
// float meters_per_second = 0;
// unsigned long lastTime = 0;
// float wheel_circumference = 0.5776; // in m
// float encoderTime = 0;
// float integral = 0;

// Servo esc;

// bool ENCODER_A_STATE;

// void encoderAFunc();

// void setup() {
//     Serial.begin(115200);
//     pinMode(ENCODER_A, INPUT);
//     esc.attach(9);       // Attach ESC to pin 9
//     delay(500);

//     // Step 1: Full throttle
//     esc.write(180);      
//     delay(1500);

//     // Step 2: Zero throttle
//     esc.write(0);        
//     delay(1500);    
//     delay(2000);
//     attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoderAFunc, CHANGE);
// }

// float P_control(float desired_speed, float RPM) {
//   float K_p = 0.2; // Proportional gain
//   float error = ((desired_speed*60)/wheel_circumference) - RPM;
//   float control_signal = K_p * error;
//   Serial.print(">Control Signal:");
//   Serial.println(control_signal);
//   return control_signal;
// }

// void loop() {
//     Serial.print(">encoderValue:");
//     Serial.println(encoder_Value);
//     Serial.print(">Rounds:");
//     Serial.println(encoder_Value/100);
//     esc.write(120); // Neutral signal

//     float desired_speed = 1; // Desired speed in m/s
//     float control_signal = P_control(desired_speed, RPM)+90;
//     if (control_signal > 180.0){
//       control_signal = 180.0;
//     } 
//     if (control_signal < 90.0){
//       control_signal = 90.0;
//     }
//     esc.write(control_signal);

//     if (millis() - lastTime >= 50) {
//         encoderDiff = (float)encoder_Value - (float)lastEncoderValue;
//         // Serial.print("  encoderDiff = ");
//         // Serial.print(encoderDiff);
//         // Serial.print("  lastEncodeValue = ");
//         // Serial.print(lastEncoderValue);
//         RPM_1 = (encoderDiff / 100) * 20 * 60;
//         RPM_4 = RPM_3;
//         RPM_3 = RPM_2;
//         RPM_2 = RPM_1;
//         RPM = (RPM_1 + RPM_2 + RPM_3 + RPM_4) / 4.0;

//         meters_per_second = (RPM * wheel_circumference) / 60;
//         lastEncoderValue = encoder_Value;
//         lastTime = millis();
//     }

//     Serial.print(">RPM:");
//     Serial.println(RPM);
//     Serial.print(">m/s:");
//     Serial.println(meters_per_second);
// }

// void encoderAFunc() {
//   if (micros() - encoderTime > 500) {
//     encoder_Value += 1;
//   }
//   encoderTime = micros();
// }
