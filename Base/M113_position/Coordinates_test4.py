import csv
import json
import time

fil = "m113log"

frames = []

with open(fil) as f:
    reader = csv.reader(f)

    for row in reader:
        timestamp = float(row[0])
        blue = json.loads(row[1])
        yellow = json.loads(row[2])

        frames.append({
            "timestamp": timestamp,
            "blue": blue,
            "yellow": yellow
        })

def run(output_queue):
    for frame in frames:
        blue = frame["blue"]
        yellow = frame["yellow"]
        output_queue.put((blue, yellow))
        time.sleep(0.2)

