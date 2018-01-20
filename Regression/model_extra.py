# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.01,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 300])
        self.y_ = tf.placeholder(tf.float32, [None])
        self.z_ = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.use_z = tf.placeholder(tf.float32)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        #logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers
        _ = self.x_
        _ = tf.matmul(_, weight_variable([300, 200])) + bias_variable([200])
        _ = tf.nn.relu(_)
        _ = tf.contrib.layers.batch_norm(_, is_training = is_train)
        _ = tf.matmul(_, weight_variable([200, 100])) + bias_variable([100])
        _ = tf.nn.relu(_)
        _ = tf.contrib.layers.batch_norm(_, is_training = is_train)
        _z = tf.matmul(_, weight_variable([100, 1])) + bias_variable([1])
        _ = tf.matmul(_, weight_variable([100, 1])) + bias_variable([1])
        
        self.loss = tf.reduce_mean((self.y_ - _) * (self.y_ - _)) + \
                tf.reduce_mean((self.z_ - _z) * (self.z_ - _z)) * self.use_z

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

