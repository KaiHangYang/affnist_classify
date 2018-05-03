import tensorflow as tf
import numpy as np

import sys
sys.path.append("../")

from utils import path_definition as m_pd

class M_Model():
    def __init__(self, input_img_size, batch_size):
        self.input_size = input_img_size
        self.batch_size = batch_size

        self.affine_loss = 0
        self.label_loss = 0

        self.input_images = None
        self.gt_labels = None
        self.gt_imgs_centered = None

        self.merged_summary = None

        self.affine_result = None
        self.classify_result = None


    def build_model(self, input_image):
        # Two part: one is get the centered img
        #           another is classification

        # train the affine process
        # affine_conv1 = tf.layers.conv2d(inputs=input_image,
                                 # filters=64,
                                 # kernel_size=[5, 5],
                                 # strides=[1, 1],
                                 # padding="same",
                                 # activation=tf.nn.relu,
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 # name="affine_conv1")

        # affine_conv2 = tf.layers.conv2d(inputs=affine_conv1,
                                 # filters=64,
                                 # kernel_size=[5, 5],
                                 # strides=[1, 1],
                                 # padding="same",
                                 # activation=tf.nn.relu,
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 # name="affine_conv2")

        # affine_conv3 = tf.layers.conv2d(inputs=affine_conv2,
                                 # filters=128,
                                 # kernel_size=[3, 3],
                                 # strides=[1, 1],
                                 # padding="same",
                                 # activation=tf.nn.relu,
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 # name="affine_conv3")

        # affine_conv4 = tf.layers.conv2d(inputs=affine_conv3,
                                 # filters=128,
                                 # kernel_size=[3, 3],
                                 # strides=[1, 1],
                                 # padding="same",
                                 # activation=tf.nn.relu,
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 # name="affine_conv4")

        # affine_conv5 = tf.layers.conv2d(inputs=affine_conv4,
                                 # filters=64,
                                 # kernel_size=[3, 3],
                                 # strides=[1, 1],
                                 # padding="same",
                                 # activation=tf.nn.relu,
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 # name="affine_conv5")

        # self.affine_result = tf.layers.conv2d(inputs=affine_conv5,
                                 # filters=1,
                                 # kernel_size=[1, 1],
                                 # strides=[1, 1],
                                 # padding="same",
                                 # activation=None,
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 # name="affine_result")

        # result_conv1 = tf.layers.conv2d(inputs=self.affine_result,
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

        result_fc1 = tf.contrib.layers.fully_connected(inputs=result_conv3,
                                 num_outputs=512,
                                 scope="result_fc1")

        result_fc2 = tf.contrib.layers.fully_connected(inputs=result_fc1,
                                 num_outputs=10,
                                 scope="result_fc2")

        self.classify_result = tf.nn.softmax(result_fc2)

    def build_loss(self, gt_imgs_centered, gt_labels, lr, lr_decay_rate, lr_decay_step):

        self.gt_imgs_centered = gt_imgs_centered
        self.gt_labels = gt_labels

        # reduce_mean divide the total loss with batch_size
        self.label_loss = tf.reduce_mean(-tf.reduce_sum(self.gt_labels * tf.log(self.classify_result), reduction_indices=[1]))
        # self.affine_loss = tf.nn.l2_loss(gt_imgs_centered - self.affine_result, name="affine_l2_loss") / self.batch_size

        self.accuracy = tf.reduce_mean(tf.cast(tf.argmax(self.gt_labels, 1) == tf.argmax(self.classify_result, 1), tf.float32))

        # tf.summary.scalar("affine loss", self.affine_loss)
        tf.summary.scalar("label loss", self.label_loss)
        tf.summary.scalar("accuracy", self.accuracy)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(lr,
                                             global_step=self.global_step,
                                             decay_rate=lr_decay_rate,
                                             decay_steps=lr_decay_step)

        tf.summary.scalar("learn rate", self.lr)

        self.train_op = tf.contrib.layers.optimize_loss(loss=self.label_loss,
                                                        global_step=self.global_step,
                                                        learning_rate=self.lr,
                                                        optimizer="Adam")

        self.merged_summary = tf.summary.merge_all()

