# -*- coding: utf-8 -*-

import tensorflow as tf


class Model_refresh_10x:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.y_rearr_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        #logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers
        _ = self.x_
        _ = tf.matmul(_, weight_variable([784, 200])) + bias_variable([200])
        _ = tf.nn.relu(_)
        _ = tf.contrib.layers.batch_norm(_, is_training = is_train)
        _rearr = tf.matmul(_, weight_variable([200, 2])) + bias_variable([2])
        refresh_w = weight_variable([200, 10])
        refresh_b = bias_variable([10])
        self.refresh = [refresh_w, refresh_b]
        _ = tf.matmul(_, refresh_w) + refresh_b
        logits = _
        

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits)) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_rearr_, logits=_rearr)) * 10
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)  # Calculate the prediction result
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

