#Only use GPU:0
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import sys

sys.path.append("../")

from nets import network_structure
from utils import tfrecord_reader as tfr_reader
from utils import path_definition as m_pd
from utils import data_preprocess as m_preprocessing

import numpy as np
import cv2
import time
import math
import random
import os

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

####################### Set the valid data path ########################
if not os.path.isdir(m_pd.test_data_dir):
    print("Test data directory is not valid!")
    quit()
valid_data_files = [os.path.join(m_pd.test_data_dir, i) for i in os.listdir(m_pd.test_data_dir)]

################################ Set the parameter #################################
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool("restore_trained_model",
        default_value=True,
        docstring="Do I need to restore the trained model?")

tf.app.flags.DEFINE_bool("reset_global_step",
        default_value=m_pd.reset_global_step,
        docstring="Do I need to reset the global_step variable?")

tf.app.flags.DEFINE_string("prev_trained_model",
        default_value=m_pd.trained_model_path,
        docstring="The prev trained model path")

tf.app.flags.DEFINE_integer('batch_size',
        default_value=32,
        docstring='Batch size.')

tf.app.flags.DEFINE_integer('input_img_size',
        default_value=m_pd.input_img_size,
        docstring='Input image size.')

##################### learning rate #######################
tf.app.flags.DEFINE_float('lr',
        default_value=m_pd.learning_rate,
        docstring='The learning_rate')
tf.app.flags.DEFINE_float('lr_decay_rate',
        default_value=m_pd.learning_decay_rate,
        docstring='The learning rate decay rate.')
tf.app.flags.DEFINE_integer('lr_decay_step',
        default_value=m_pd.learning_decay_step,
        docstring='The learing rate decay step')

tf.app.flags.DEFINE_integer('valid_iter',
        default_value=10000,
        docstring='Valid iteration')

tf.app.flags.DEFINE_string('valid_log_dir',
        default_value=os.path.join(m_pd.valid_log_dir, 'valid'),
        docstring='The log directory.')

tf.app.flags.DEFINE_string('log_file_name',
        default_value="log",
        docstring='The log file prefix')

def main(argv):
    batch_img_valid, batch_labels_valid = tfr_reader.read_batch(valid_data_files,
            FLAGS.input_img_size, batch_size = FLAGS.batch_size, is_shuffle = False, reader_name = "valid")

    input_image = tf.placeholder(dtype=tf.float32,
            shape=(None, FLAGS.input_img_size, FLAGS.input_img_size, 1),
            name='input_image')

    # input_image_centered = tf.placeholder(dtype=tf.float32,
            # shape=(None, FLAGS.input_img_size, FLAGS.input_img_size, 1),
            # name="input_image_centered")

    input_labels = tf.placeholder(dtype=tf.float32,
            shape=(None, 10),
            name="input_labels")

    model = network_structure.M_Model(FLAGS.input_img_size, FLAGS.batch_size)
    model.build_model(input_image)
    model.build_loss(input_labels, FLAGS.lr, FLAGS.lr_decay_rate, FLAGS.lr_decay_step)

    # then train
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # create the log writer

        valid_writer = tf.summary.FileWriter(logdir=FLAGS.valid_log_dir, graph=sess.graph,
                filename_suffix=FLAGS.log_file_name)

        # create the weight saver
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        # reload the model
        if FLAGS.restore_trained_model:
            if os.path.exists(FLAGS.prev_trained_model+".index"):
                print("#######################Restored all weights ###########################")
                saver.restore(sess, FLAGS.prev_trained_model)
            else:
                print("The prev model is not existing!")
                quit()

        global_step = 0
        total_img_num = 0
        total_accutate_num = 0
        while True:
            total_img_num += FLAGS.batch_size

            # batch_img_np, batch_img_centered_np, batch_labels_np = sess.run([batch_img_valid, batch_img_centered_valid, batch_labels_valid])
            batch_img_np, batch_labels_np = sess.run([batch_img_valid, batch_labels_valid])

            batch_img_np = batch_img_np.astype(np.float32)
            # batch_img_centered_np = batch_img_centered_np.astype(np.float32)

            batch_labels_np = batch_labels_np.astype(np.int32)
            batch_one_hot_labels_np = np.zeros([batch_labels_np.shape[0], 10], np.float32)

            # preprocess train data
            for img_num in range(batch_img_np.shape[0]):
                # batch_img_np[img_num], batch_img_centered_np[img_num] = m_preprocessing.preprocess(batch_img_np[img_num].copy(), batch_img_centered_np[img_num].copy())
                batch_img_np[img_num]  = m_preprocessing.preprocess(batch_img_np[img_num].copy())
                batch_one_hot_labels_np[img_num][batch_labels_np[img_num]] = 1.0

            label_loss_np, \
            accurate_num_np, \
            summary = sess.run([
                model.label_loss,
                model.accurate_num,
                model.merged_summary,
                ], feed_dict={
                    input_image:batch_img_np,
                    # input_image_centered:batch_img_centered_np,
                    input_labels:batch_one_hot_labels_np
                    })

            valid_writer.add_summary(summary, global_step)

            total_accutate_num += accurate_num_np

            print("##========={:} Iter {:>6d} ============##".format("Valid", global_step))

            print('Label Loss: {:>.3f}\n'.format(label_loss_np))
            print('Current Accuracy: {:>.3f}\n\n'.format(1.0 * total_accutate_num / total_img_num))
            print("Total %d, correct %d" % (total_accutate_num, total_img_num))

            if global_step == FLAGS.valid_iter:
                break

            global_step += 1

        coord.request_stop()
        coord.join(threads)

    print("Train done.")

if __name__ == "__main__":
    tf.app.run()
