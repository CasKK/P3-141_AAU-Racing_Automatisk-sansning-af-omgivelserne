#include <Arduino.h>
#include <Servo.h>

Servo myServo;  // create a servo object




int ENCODER_A = 2;

unsigned long encoderTime = 0;

long encoder_Value = 0;


void encoderAFunc();

void setup() {
  Serial.begin(115200);
  pinMode(ENCODER_A, INPUT);
  myServo.attach(6);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoderAFunc, RISING);
}

void loop() {
  Serial.print("  >DigitalValue:");
  Serial.print(digitalRead(ENCODER_A));
  Serial.print("  >EncoderValue:");
  Serial.println(encoder_Value);
  myServo.write(135);
  delay(1);
}

void encoderAFunc() {
  if (micros() - encoderTime > 500) {
    encoder_Value += 1;
  }
  encoderTime = micros();
}
