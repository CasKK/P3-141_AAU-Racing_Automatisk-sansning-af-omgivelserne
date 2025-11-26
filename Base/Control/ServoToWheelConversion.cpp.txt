#include <Arduino.h>
#include <Servo.h>
#include <math.h>

Servo myservo;  // create Servo object to control a servo

float val = 90;    // initial servo angle

void setup() {
  myservo.attach(6);  // attach the servo to pin A0
  Serial.begin(115200);
  myservo.write(val);   // set initial position
  Serial.println("Servo control ready. Enter angle (0-180):");
}

float MotorToServo(float x) {
  float x2 = x * x;
  float x3 = x2 * x;
  float servoAngle =  0.00123378679436315 * x3 - 0.34492174790361900 * x2 + 34.23770218227510000 * x - 1095.74990424557000000;
  return constrain(servoAngle, 0, 180);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // read until newline
    input.trim();                                 // remove whitespace
    float angle = input.toFloat();                // convert to float
    if (angle >= 0 && angle <= 180) {             // sanity check
      val = angle;
      myservo.write(MotorToServo(val));
      Serial.print("Servo angle set to: ");
      Serial.println(MotorToServo(val));
    } else {
      Serial.println("Invalid angle! Enter value between 0 and 180.");
    }
  }
}
