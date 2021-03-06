import numpy as np
import tensorflow as tf
import scipy.io as sio
import os
import cv2

class affNISTReader():
    def __init__(self, base_dir, centered_dir, img_size=40):
        self.image_centered = []
        self.label_centered = []
        self.image_datas = []
        self.label_datas = []
        self.dataset_num = 0
        self.img_size = img_size
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

        # if os.path.isdir(centered_dir):
            # mat_files = os.listdir(centered_dir)
            # for mat_file in mat_files:
                # mat_data = sio.loadmat(os.path.join(centered_dir, mat_file))["affNISTdata"][0, 0]
                # self.image_centered = mat_data["image"]
                # self.label_centered = mat_data["label_int"]
                # break
        # else:
            # print("Centered dir is not valid!")
            # quit()

    def writeTFRecord(self, target_dir):
        tf_train_writer = tf.python_io.TFRecordWriter(os.path.join(target_dir, "train.tfrecord"))
        # tf_valid_writer = tf.python_io.TFRecordWriter(os.path.join(target_dir, "valid.tfrecord"))

        cur_writer = tf_train_writer

        for i in range(self.dataset_num):

            # if i >= 24:
                # cur_writer = tf_valid_writer

            imgs = self.image_datas[i]
            labels = self.label_datas[i]

            img_num = imgs.shape[1]
            for j in range(img_num):
                cur_label = labels[0, j]
                cur_img = cv2.resize(np.reshape(imgs[:, j], [self.img_size, self.img_size]), (32, 32))

                # centered_label = self.label_centered[0, j]
                # centered_img = cv2.resize(np.reshape(self.image_centered[:, j].copy(), [self.img_size, self.img_size]), (32, 32))

                # assert(cur_label == centered_label)

                example = tf.train.Example(features = tf.train.Features(
                    feature={
                        "image": tf.train.Feature(bytes_list = tf.train.BytesList(value = [cur_img.tobytes()])),
                        # "centered_image": tf.train.Feature(bytes_list = tf.train.BytesList(value=[centered_img.tobytes()])),
                        "label": tf.train.Feature(float_list = tf.train.FloatList(value = np.array([cur_label], dtype=np.float32)))
                        }
                    ))
                cur_writer.write(example.SerializeToString())

    def showData(self):
        print(self.dataset_num)

        img_num = 60000

        for j in range(img_num):

            for i in range(self.dataset_num):
                imgs = self.image_datas[i]
                labels = self.label_datas[i]

                print(labels[0, j])
                img = cv2.resize(np.reshape(imgs[:, j], [self.img_size, self.img_size]), (32, 32))
                cv2.imshow("img", img)
                cv2.waitKey()


if __name__ == "__main__":
    reader = affNISTReader("/home/kaihang/DataSet_2/affNIST/test_batches/", "/home/kaihang/DataSet_2/affNIST/training_and_validation_batcher_centered", 40)
    reader.writeTFRecord("/home/kaihang/DataSet/")
