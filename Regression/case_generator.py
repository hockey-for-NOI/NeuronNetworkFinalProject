import csv
import numpy as np
import random

INPUT_FILE_NAME = 't_price_filtered.csv'
csv_reader = csv.reader(open(INPUT_FILE_NAME))
OUTPUT_FILE_NAME = 't_price_cut.csv'
LABEL_FILE_NAME = 't_label.csv'

flag = False
data = []
sample = []
for row in csv_reader:
	data.append(row)
data = np.array(data)
data = data.reshape([-1, 16201])
print data.shape
data = data[:, 260: -241]

for i in range(1211):
	for j in range(155):
		sample.append(data[i, j * 100: j * 100 + 300])

sample = np.array(sample)
csv_writer = csv.writer(open(OUTPUT_FILE_NAME, 'w'))
csv_writer_2 = csv.writer(open(LABEL_FILE_NAME, 'w'))
for i in range(sample.shape[0]):
	csv_writer.writerow(list(sample[i]))
print sample.shape

for i in range(sample.shape[0]):
	if (i < 50000):
		csv_writer_2.writerow([float(sorted(sample[i])[int(300 * 0.40) - 1])])
		continue
	if (i < 100000):
		csv_writer_2.writerow([float(sorted(sample[i])[int(300 * 0.50) - 1])])
	else:
		csv_writer_2.writerow([float(sorted(sample[i])[int(300 * 0.60) - 1])])