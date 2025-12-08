
float someVal = 0;
float someOtherVal = 0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  Serial.print(someVal);
  Serial.print(",");
  Serial.print(someOtherVal);
  Serial.print("\n");
  someVal += 0.001;
  someOtherVal += 0.1;
  delay(10);
}
