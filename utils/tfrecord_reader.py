import tensorflow as tf
import numpy as np
import os

class PathError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value) + ": file is not existing!"


def read_and_decode(tfr_queue, img_size):
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           # 'centered_image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([1], tf.float32),
                                       })

    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [img_size, img_size, 1])

    # centered_img = tf.decode_raw(features['centered_image'], tf.uint8)
    # centered_img = tf.reshape(centered_img, [img_size, img_size, 1])

    label = features['label'][0]

    # return [img], [centered_img], [label]
    return [img], [label]

def read_batch(tfr_paths, img_size, batch_size=4, is_shuffle=True, reader_name="train", num_epochs=None):

    for i in range(len(tfr_paths)):
        if not os.path.exists(tfr_paths[i]):
            raise PathError(tfr_paths[i])

    with tf.name_scope('Batch_Inputs'):
        # Traing
        tfr_queue = tf.train.string_input_producer(tfr_paths, num_epochs=num_epochs, shuffle=is_shuffle)
        data_list = [read_and_decode(tfr_queue, img_size) for _ in range(1 * len(tfr_paths))]

        if is_shuffle:
            batch_images, batch_labels = tf.train.shuffle_batch_join(data_list,
            # batch_images, batch_centered_images, batch_labels = tf.train.shuffle_batch_join(data_list,
                                                                    batch_size=batch_size,
                                                                    capacity=300,
                                                                    min_after_dequeue=80,
                                                                    enqueue_many=True,
                                                                    name='%s_data_reader' % reader_name)
        else:
            batch_images, batch_labels = tf.train.batch_join(data_list,
            # batch_images, batch_centered_images, batch_labels = tf.train.batch_join(data_list,
                                                             batch_size=batch_size,
                                                             capacity=300,
                                                             enqueue_many=True,
                                                             name='%s_data_reader' % reader_name)
    return batch_images, batch_labels
    # return batch_images, batch_centered_images, batch_labels
