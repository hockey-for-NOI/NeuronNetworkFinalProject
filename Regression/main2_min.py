# -*- coding: utf-8 -*-

import sys

import tensorflow as tf
import numpy as np
import os
import time
from model_extra import Model
from load_data_with_min import load_data

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


def train_epoch(model, sess, X, y, e): # Training Process
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(X) and ed <= len(X):
        X_batch, y_batch = X[st:ed], y[st:ed]
        feed = {model.x_: X_batch, model.y_: y_batch, model.z_: e, model.keep_prob: FLAGS.keep_prob, model.use_z: 1.0}
        loss_, _ = sess.run([model.loss, model.train_op], feed)
        loss += loss_
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    return loss


def valid_epoch(model, sess, X, y):  # Valid Process
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(X) and ed < len(X):
        X_batch, y_batch = X[st:ed], y[st:ed]
        feed = {model.x_: X_batch, model.y_: y_batch, model.z_: y_batch, model.keep_prob: 1.0, model.use_z: 0.0}
        loss_, = sess.run([model.loss], feed)
        loss += loss_
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    return loss


def inference(model, sess, X):  # Test Process
    return sess.run([model.pred], {model.x_: X})[0]


def work(n_split):

    X_train_all, X_test_all, y_train_all, y_test_all, extra_train = load_data(n_split)

    mlp_model = Model(True)

    model_pref = "MLP_std_{}split_min".format(n_split)
    train_dir = "{}_std_{}split_min".format(FLAGS.train_dir_pref, n_split)

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    with tf.Session() as sess:

        if tf.train.get_checkpoint_state(train_dir):
            mlp_model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
        else:
            tf.global_variables_initializer().run()

        for i_split in range(n_split):
            X_train, y_train, e_train = X_train_all[i_split], y_train_all[i_split], extra_train[i_split]
            
            pre_losses = [1e18] * 3
            for epoch in range(FLAGS.num_epochs):
                start_time = time.time()
                train_loss = train_epoch(mlp_model, sess, X_train, y_train, e_train)  # Complete the training process
                test_loss = valid_epoch(mlp_model, sess, X_test_all, y_test_all)

                mlp_model.saver.save(sess, '%s/checkpoint' % train_dir, global_step=mlp_model.global_step)
                
                with open(model_pref + ".csv", "a") as f:
                    f.writelines([",".join([str(train_loss), str(test_loss), "\n"])])

                epoch_time = time.time() - start_time
                print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.num_epochs) + " took " + str(epoch_time) + "s")
                print("  learning rate:                 " + str(mlp_model.learning_rate.eval()))
                print("  training loss:                 " + str(train_loss))
                print("  testing loss:                 " + str(test_loss))

                if train_loss > max(pre_losses):  # Learning rate decay
                    sess.run(mlp_model.learning_rate_decay_op)
                pre_losses = pre_losses[1:] + [train_loss]

if __name__ == "__main__":
    work(int(sys.argv[1]))
