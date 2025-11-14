#include <Arduino.h>

int ENCODER_A = 9;

int encoder_Value = 0;


void encoderAFunc();

void setup() {
  Serial.begin(115200);
  pinMode(ENCODER_A, INPUT);

  attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoderAFunc, CHANGE);
}

void loop() {
  Serial.print(">encoderValue:");
  Serial.println(encoder_Value);
  delay(1);
}

void encoderAFunc() {
  encoder_Value += 1;
}
