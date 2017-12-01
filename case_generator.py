import csv
import numpy as np

INPUT_FILE_NAME = 't_price.csv'
csv_reader = csv.reader(open(INPUT_FILE_NAME))
OUTPUT_FILE_NAME = 't_price_cut.csv'
LABEL_FILE_NAME = 't_label.csv'

flag = False
data = []
sample = []
for row in csv_reader:
	if flag == False:
		flag = True
		continue
	data.append(row)
data = np.array(data)
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

print sample[0]
print sorted(sample[0])
print sorted(sample[0])[80 - 1]
for i in range(sample.shape[0]):
	csv_writer_2.writerow([float(sorted(sample[i])[int(200 * 0.4) - 1])])