import tensorflow as tf
import numpy as np

import sys
sys.path.append("../")

from utils import path_definition as m_pd

class M_Model():
    def __init__(self, input_img_size, batch_size):
        self.input_size = input_img_size
        self.batch_size = batch_size

        self.label_loss = 0

        self.gt_labels = None

        self.merged_summary = None

        self.classify_result = None


    def build_model(self, input_image):
        result_conv1 = tf.layers.conv2d(inputs=input_image,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 strides=[1, 1],
                                 padding="same",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name="result_conv1")
        result_pool1 = tf.layers.conv2d(inputs=result_conv1,
                                 filters=128,
                                 kernel_size=[2, 2],
                                 strides=[2, 2],
                                 padding="same",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name="result_pool1")

        result_conv2 = tf.layers.conv2d(inputs=result_pool1,
                                 filters=128,
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding="same",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name="result_conv2")

        result_pool2 = tf.layers.conv2d(inputs=result_conv2,
                                 filters=64,
                                 kernel_size=[2, 2],
                                 strides=[2, 2],
                                 padding="same",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name="result_pool2")

        result_conv3 = tf.layers.conv2d(inputs=result_pool2,
                                 filters=64,
                                 kernel_size=[1, 1],
                                 strides=[1, 1],
                                 padding="same",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name="result_conv3")

        result_conv3_flatten = tf.reshape(result_conv3, [-1, 64 * 8 * 8])
        result_fc1 = tf.layers.dense(inputs=result_conv3_flatten, units=512, activation=tf.nn.relu)
        result_fc2 = tf.layers.dense(inputs=result_fc1, units=10)

        self.classify_result = result_fc2

    def build_loss(self, gt_labels, lr, lr_decay_rate, lr_decay_step):

        self.gt_labels = gt_labels

        self.label_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.gt_labels, logits=self.classify_result)

        self.result_index = tf.argmax(self.classify_result, 1)
        self.gt_index = tf.argmax(self.gt_labels, 1)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.gt_labels, 1), tf.argmax(self.classify_result, 1)), tf.float32))
        self.accurate_num = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.gt_labels, 1), tf.argmax(self.classify_result, 1)), tf.float32))

        # tf.summary.scalar("affine loss", self.affine_loss)
        tf.summary.scalar("label loss", self.label_loss)
        tf.summary.scalar("accuracy", self.accuracy)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        # self.lr = tf.train.exponential_decay(lr,
                                             # global_step=self.global_step,
                                             # decay_rate=lr_decay_rate,
                                             # decay_steps=lr_decay_step)

        # tf.summary.scalar("learning rate", self.lr)
        # self.train_op = tf.contrib.layers.optimize_loss(loss=self.label_loss,
                                                        # global_step=self.global_step,
                                                        # learning_rate=self.lr,
                                                        # optimizer="Adam")

        optimizer = tf.train.GradientDescentOptimizer(0.05)
        self.train_op = optimizer.minimize(
                loss = self.label_loss,
                global_step = self.global_step)

        self.merged_summary = tf.summary.merge_all()
