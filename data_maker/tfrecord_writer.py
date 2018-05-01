import numpy as np
import tensorflow as tf
import scipy.io as sio
import os
import cv2

class affNISTReader():
    def __init__(self, base_dir):
        self.image_datas = []
        self.label_datas = []
        self.dataset_num = 0
        if os.path.isdir(base_dir):
            mat_files = os.listdir(base_dir)
            for mat_file in mat_files:
                self.dataset_num += 1
                mat_data = sio.loadmat(os.path.join(base_dir, mat_file))["affNISTdata"][0, 0]
                self.image_datas.append(mat_data["image"])
                self.label_datas.append(mat_data["label_int"])
        else:
            print("Base dir is not valid!")
            quit()

    def writeTFRecord(self, target_dir):
        tf_train_writer = tf.python.io.TFRecordWriter(os.path.join(target_dir, "train.tfrecord"))
        tf_valid_writer = tf.python.io.TFRecordWriter(os.path.join(target_dir, "valid.tfrecord"))

        cur_writer = tf_train_writer

        for i in range(self.dataset_num):

            if i >= 24:
                cur_writer = tf_valid_writer

            imgs = self.image_datas[i]
            labels = self.label_datas[i]

            img_num = imgs.shape[1]
            for j in range(img_num):
                cur_label = labels[0, j]
                cur_img = cv2.resize(np.reshape(imgs[:, j], [40, 40]), [32, 32])

                example = tf.train.Example(features = tf.train.Features(
                    feature={
                        "image": tf.train.Feature(bytes_list = tf.train.BytesList(value = [cur_img.tobytes()])),
                        "label:" tf.train.Feature(float_list = tf.train.FloatList(value = [cur_label]))
                        }
                    ))
                cur_writer.write(example.SerializeToString())

    def showData(self):
        print(self.dataset_num)
        for i in range(self.dataset_num):
            imgs = self.image_datas[i]
            labels = self.label_datas[i]

            img_num = imgs.shape[1]

            for j in range(img_num):
                print(labels[0, j])
                img = cv2.resize(np.reshape(imgs[:, j], [40, 40]), (32, 32))
                cv2.imshow("img", img)
                cv2.waitKey()


if __name__ == "__main__":
    reader = affNISTReader("/Users/kaihang/Downloads/training_and_validation_batches/")
    reader.writeTFRecord("/Users/kaihang/Downloads/")
