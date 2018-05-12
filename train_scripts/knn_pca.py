import numpy as np
from sklearn import neighbors
from sklearn.decomposition import PCA
import scipy.io as sio
import time
import gc


if __name__ == "__main__":
    # load training set range(0, 30)
    n_components = 128

    train_set_num = 24
    test_set_num = 32

    train_images = []
    train_features = []
    train_labels = []
    test_images = []
    test_labels = []

    start = time.clock()
    for num in range(1, train_set_num + 1):
        mat_data = sio.loadmat("/home/kaihang/DataSet_2/affNIST/training_and_validation_batches/%d.mat" % num)
        train_images.append(np.transpose(mat_data["affNISTdata"][0, 0]["image"]).copy())
        train_labels.append(np.transpose(mat_data["affNISTdata"][0, 0]["label_int"]).copy().flatten())
        del mat_data

    gc.collect()

    for num in range(1, test_set_num + 1):
        mat_data = sio.loadmat("/home/kaihang/DataSet_2/affNIST/test_batches/%d.mat" % num)
        test_images.append(np.transpose(mat_data["affNISTdata"][0, 0]["image"]).copy())
        test_labels.append(np.transpose(mat_data["affNISTdata"][0, 0]["label_int"]).copy().flatten())
        del mat_data

    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)

    gc.collect()

    end = time.clock()
    print("Load time %fs" % (end - start))


    start = time.clock()
    _pca = PCA(n_components=n_components, copy=False, whiten=True, svd_solver="randomized")
    _pca.fit(test_images)

    end = time.clock()
    print("PCA time %fs" % (end - start))

    start = time.clock()
    for num in range(train_set_num):
        train_features.append(_pca.transform(train_images[num]))

    del train_images
    gc.collect()

    train_features = np.concatenate(np.array(train_features))
    train_labels = np.concatenate(np.array(train_labels))

    test_features = _pca.transform(test_images)

    end = time.clock()
    print("PCA transform time %fs" % (end - start))

    del test_images
    gc.collect()

    start = time.clock()
    _knn = neighbors.KNeighborsClassifier()
    _knn.fit(train_features, train_labels)
    end = time.clock()
    print("KNN fitting time %fs" % (end - start))
    gc.collect()

    total_frame_sum = test_features.shape[0]
    correct_frame_sum = 0

    with open("../logs/knn_pca_1.log", "w") as f:
        for frame_num in range(total_frame_sum):
            predict_result = _knn.predict([test_features[frame_num]])[0]
            correct_frame_sum += int(test_labels[frame_num] == predict_result)
            f.write(str((predict_result, test_labels[frame_num])) + "\n")
            f.write("Currently correct rate: %f( %d / %d / %d)\n" % (correct_frame_sum * 1.0 / (frame_num + 1), correct_frame_sum, frame_num + 1, total_frame_sum))

        f.write("Total frame sum: %d, correct frame num: %d" % (total_frame_sum, correct_frame_sum))

