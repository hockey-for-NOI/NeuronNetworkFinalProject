import csv
import numpy as np
import random

INPUT_FILE_NAME = 't_price.csv'
OUTPUT_FILE_NAME = 't_price_filtered.csv'
csv_reader = csv.reader(open(INPUT_FILE_NAME))

data = []
for row in csv_reader:
	for d in row:
		data.append(float(d))

output = []
output.append(float(data[0]))

datalen = len(data)

for i in range(1, datalen):
	output.append(float(0.05 * data[i] + 0.95 * output[i - 1]))

csv_writer = csv.writer(open(OUTPUT_FILE_NAME, 'w'))
csv_writer.writerow(output)
