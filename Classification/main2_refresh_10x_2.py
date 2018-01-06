# -*- coding: utf-8 -*-

import sys

import tensorflow as tf
import numpy as np
import os
import time
from model_refresh_10x import Model_refresh_10x
from load_data import load_mnist_2d

tf.app.flags.DEFINE_integer("batch_size", 100, "batch size for training")
tf.app.flags.DEFINE_integer("num_epochs", 20, "number of epochs")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "drop out rate")
tf.app.flags.DEFINE_string("data_dir", "./data", "data dir")
tf.app.flags.DEFINE_string("train_dir_pref", "./main2/train", "training dir prefix")
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
        y_rearr = np.vectorize(lambda x: np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])[x])(y_batch)
        feed = {model.x_: X_batch, model.y_: y_batch, model.y_rearr_: y_rearr, model.keep_prob: FLAGS.keep_prob}
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

    X_train_all, X_test_all, y_train_all, y_test_all = load_mnist_2d(FLAGS.data_dir, n_split)

    mlp_model = Model_refresh_10x(True)

    model_pref = "MLP_std_{}split_refresh_10x_2".format(n_split)
    train_dir = "{}_std_{}split_refresh_10x_2".format(FLAGS.train_dir_pref, n_split)

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    with tf.Session() as sess:

        if tf.train.get_checkpoint_state(train_dir):
            mlp_model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
        else:
            tf.global_variables_initializer().run()

        for i_split in range(n_split):
            X_train, y_train = X_train_all[i_split], y_train_all[i_split]
            
            for i in mlp_model.refresh:
                i.initializer.run()
            
            pre_losses = [1e18] * 3
            for epoch in range(FLAGS.num_epochs):
                start_time = time.time()
                X_train, y_train = shuffle(X_train, y_train, 1)
                train_acc, train_loss = train_epoch(mlp_model, sess, X_train, y_train)  # Complete the training process

                mlp_model.saver.save(sess, '%s/checkpoint' % train_dir, global_step=mlp_model.global_step)
                
                with open(model_pref + ".csv", "a") as f:
                    f.writelines([",".join([str(train_loss), str(train_acc), "\n"])])

                epoch_time = time.time() - start_time
                print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.num_epochs) + " took " + str(epoch_time) + "s")
                print("  learning rate:                 " + str(mlp_model.learning_rate.eval()))
                print("  training loss:                 " + str(train_loss))

                if train_loss > max(pre_losses):  # Learning rate decay
                    sess.run(mlp_model.learning_rate_decay_op)
                pre_losses = pre_losses[1:] + [train_loss]

if __name__ == "__main__":
    work(int(sys.argv[1]))
