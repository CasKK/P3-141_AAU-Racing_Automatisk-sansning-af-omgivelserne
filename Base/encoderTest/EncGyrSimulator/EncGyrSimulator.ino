
float someVal = 0;
float someOtherVal = 0;

void setup() {
  Serial.begin(115200);
  delay(7000);
}

void loop() {
  Serial.print(someVal);
  Serial.print(",");
  Serial.print(someOtherVal);
  Serial.print("\n");
  someVal += 0.02;
  someOtherVal += 2;
  delay(67);
}
