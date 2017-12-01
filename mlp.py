import numpy as np
import tensorflow as tf
import csv
import random

DATA_FILE_NAME = 't_price_cut.csv'
LABEL_FILE_NAME = 't_label.csv'

data = []
label = []

data_reader = csv.reader(open(DATA_FILE_NAME))
label_reader = csv.reader(open(LABEL_FILE_NAME))

for row in data_reader:
	data.append(row)

for row in label_reader:
	label.append(row)

data = np.array(data)
data = data.astype(float)
label = np.array(label)
label = label.astype(float)

label -= np.mean(data[:150000, :], axis = (0, 1))
data -= np.mean(data[:150000, :], axis = (0, 1))
label /= np.std(data[:150000, :], axis = (0, 1))
data /= np.std(data[:150000, :], axis = (0, 1))

train_data = data[:150000, :]
train_label = label[:150000, :]
test_data = data[150000:, :]
test_label = label[150000:, :]

batch_size = 100
start_idx = 0
max_idx = 15000

def get_train_batch(batch_size = 100):
	global start_idx
	batch_data = []
	batch_label = []
	for i in range(batch_size):
		idx = random.randint(0, max_idx)
		batch_data.append(train_data[idx])
		batch_label.append(train_label[idx])
	batch_data = np.array(batch_data)
	batch_label = np.array(batch_label)
	return batch_data, batch_label

def weight_variable(shape):
	initial = tf.truncated_normal(shape = shape, stddev = 0.001)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.0, shape = shape)
	return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape = [None, 300])
y_input = tf.placeholder(tf.float32, shape = [None, 1])

W_conv1 = weight_variable([300, 200])
b_conv1 = bias_variable([200])

x_conv1 = tf.nn.relu(tf.matmul(x, W_conv1) + b_conv1)

W_conv2 = weight_variable([200, 1])
b_conv2 = bias_variable([1])

x_output = tf.matmul(x_conv1, W_conv2) + b_conv2

loss = tf.reduce_mean((y_input - x_output) * (y_input - x_output))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

cnt = 0
with tf.Session() as sess:
	while(True):
		sess.run(tf.global_variables_initializer())
		nxt_batch = get_train_batch()
		sess.run(train_step, feed_dict = {x: nxt_batch[0], y_input: nxt_batch[1]})
		print sess.run(loss, feed_dict = {x: nxt_batch[0], y_input: nxt_batch[1]})
		cnt += 1

		if (cnt % 100) == 0:
			print sess.run(loss, feed_dict = {x: test_data, y_input: test_label})