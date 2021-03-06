# -*- coding: utf-8 -*-

import sys

import tensorflow as tf
import numpy as np
import os
import time
from model import Model
from load_data import load_mnist_2d

tf.app.flags.DEFINE_integer("batch_size", 100, "batch size for training")
tf.app.flags.DEFINE_integer("num_epochs", 20, "number of epochs")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "drop out rate")
tf.app.flags.DEFINE_string("data_dir", "./data", "data dir")
tf.app.flags.DEFINE_string("train_dir_pref", "./train", "training dir prefix")
tf.app.flags.DEFINE_integer("inference_version", 0, "the version for inferencing")
FLAGS = tf.app.flags.FLAGS


def shuffle(X, y, shuffle_parts):  # Shuffle the X and y
    chunk_size = len(X) // shuffle_parts
    shuffled_range = range(chunk_size)

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(list(shuffled_range))
        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
            y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
        y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

    return X, y


def train_epoch(model, sess, X, y): # Training Process
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(X) and ed <= len(X):
        X_batch, y_batch = X[st:ed], y[st:ed]
        feed = {model.x_: X_batch, model.y_: y_batch, model.keep_prob: FLAGS.keep_prob}
        loss_, acc_, _ = sess.run([model.loss, model.acc, model.train_op], feed)
        loss += loss_
        acc += acc_
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    acc /= times
    return acc, loss


def valid_epoch(model, sess, X, y):  # Valid Process
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(X) and ed < len(X):
        X_batch, y_batch = X[st:ed], y[st:ed]
        feed = {model.x_: X_batch, model.y_: y_batch, model.keep_prob: 1.0}
        loss_, acc_ = sess.run([model.loss, model.acc], feed)
        loss += loss_
        acc += acc_
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    acc /= times
    return acc, loss


def inference(model, sess, X):  # Test Process
    return sess.run([model.pred], {model.x_: X})[0]


def work(n_split):

    for i_split in range(1, n_split - 1):
        with tf.Session() as sess:
            X_train, X_test, y_train, y_test = load_mnist_2d(FLAGS.data_dir, n_split)
            X_test1, y_test1 = X_train[i_split - 1], y_train[i_split - 1]
            X_test2, y_test2 = X_train[i_split + 1], y_train[i_split + 1]
            X_train, y_train = X_train[i_split], y_train[i_split]
            mlp_model = Model(True)
            
            model_pref = "MLP_{}_{}".format(i_split, n_split)
            train_dir = "{}_{}_{}".format(FLAGS.train_dir_pref, i_split, n_split)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)

            if tf.train.get_checkpoint_state(train_dir):
                mlp_model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            else:
                tf.global_variables_initializer().run()

            pre_losses = [1e18] * 3
            for epoch in range(FLAGS.num_epochs):
                start_time = time.time()
                train_acc, train_loss = train_epoch(mlp_model, sess, X_train, y_train)  # Complete the training process
                X_train, y_train = shuffle(X_train, y_train, 1)

                test_acc1, test_loss1 = valid_epoch(mlp_model, sess, X_test1, y_test1)  # Complete the test process
                test_acc2, test_loss2 = valid_epoch(mlp_model, sess, X_test2, y_test2)  # Complete the test process
                mlp_model.saver.save(sess, '%s/checkpoint' % train_dir, global_step=mlp_model.global_step)
                    
                with open(model_pref + ".csv", "a") as f:
                    f.writelines([",".join([str(train_loss), str(train_acc), str(test_acc1), str(test_acc2), "\n"])])

                epoch_time = time.time() - start_time
                print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.num_epochs) + " took " + str(epoch_time) + "s")
                print("  learning rate:                 " + str(mlp_model.learning_rate.eval()))
                print("  training loss:                 " + str(train_loss))
                print("  test1 loss:                     " + str(test_loss1))
                print("  test1 accuracy:                 " + str(test_acc1))
                print("  test2 loss:                     " + str(test_loss2))
                print("  test2 accuracy:                 " + str(test_acc2))

                if train_loss > max(pre_losses):  # Learning rate decay
                    sess.run(mlp_model.learning_rate_decay_op)
                pre_losses = pre_losses[1:] + [train_loss]

if __name__ == "__main__":
    work(int(sys.argv[1]))
