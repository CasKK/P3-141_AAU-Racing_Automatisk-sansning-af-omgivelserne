#include <Arduino.h>
#include <Servo.h>

Servo myServo;  // create a servo object
Servo myServo1;  // create a servo object




int ENCODER_A = 2;

unsigned long encoderTime = 0;

long encoder_Value = 0;


void encoderAFunc();

void setup() {
  Serial.begin(115200);
  pinMode(ENCODER_A, INPUT);
  myServo.attach(11, 1000, 2000);
  myServo1.attach(10);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoderAFunc, RISING);
}

void loop() {
  // Serial.print("  >DigitalValue:");
  Serial.print(digitalRead(ENCODER_A));
  Serial.print(",");
  //Serial.print("  >EncoderValue:");
  Serial.println(encoder_Value);
  myServo.writeMicroseconds(1750);
  myServo1.write(90);
  delay(1);
  while (encoder_Value > 2000){
    myServo.writeMicroseconds(1750);
    Serial.print(digitalRead(ENCODER_A));
    Serial.print(",");
    //Serial.print("  >EncoderValue:");
    Serial.println(encoder_Value);
    delay(1000);
  }
}

void encoderAFunc() {
  if (micros() - encoderTime > 1000) {
    encoder_Value += 1;
  }
  encoderTime = micros();
}
