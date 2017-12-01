import numpy as np
import pandas as pd
import csv

OUTPUT_FILE_NAME = 't_price.csv'
csv_writer = csv.writer(open(OUTPUT_FILE_NAME, 'w'))

askMatrix = []
def load(INPUT_FILE_NAME):
	csv_reader = csv.reader(open(INPUT_FILE_NAME))
	flag = False
	for row in csv_reader:
		if (flag == False):
			flag = True
			pricerow = []
			continue
		if len(row) == 0:
			askMatrix.append(pricerow)
			print len(askMatrix)
			pricerow = []
		else:
			pricerow.append(float(row[5]))
	askMatrix.append(pricerow)

load('t_1509.01.csv')
load('t_1512.01.csv')
load('t_1603.01.csv')
load('t_1606.01.csv')
load('t_1609.01.csv')
load('t_1612.01.csv')
load('t_1603.01.csv')
load('t_1706.01.csv')
load('t_1709.01.csv')
load('t_1712.01.csv')
csv_writer.writerow('')
for row in askMatrix:
	csv_writer.writerow(row)
x = np.array(askMatrix)
print x.shape