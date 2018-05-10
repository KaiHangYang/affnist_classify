import numpy as np
from sklearn import neighbors
from sklearn.decomposition import PCA
import scipy.io as sio
import time
import gc


if __name__ == "__main__":
    # load training set range(0, 30)
    train_set_num = 30
    test_set_num = 32
    train_images = []
    train_features = []
    train_labels = []
    train_pca = []
    knn_arr = []

    test_images = []
    test_labels = []

    start = time.clock()
    for num in range(1, train_set_num + 1):
        mat_data = sio.loadmat("/home/kaihang/DataSet_2/affNIST/training_and_validation_batches/%d.mat" % num)
        train_images.append(np.transpose(mat_data["affNISTdata"][0, 0]["image"]).copy())
        train_labels.append(np.transpose(mat_data["affNISTdata"][0, 0]["label_int"]).copy().flatten())
        del mat_data

    end = time.clock()
    print("Load time %fs" % (end - start))

    # force the garbage collector
    gc.collect()

    start = time.clock()
    for num in range(train_set_num):
        print("Currently process data set %d: " % num)
        train_pca.append(PCA(n_components=40, copy=False, whiten=True, svd_solver='randomized').fit(train_images[num]))
        train_features.append(train_pca[num].transform(train_images[num]))

    del train_images
    gc.collect()

    end = time.clock()
    print("Transform time %fs" % (end - start))

    for num in range(train_set_num):
        knn_arr.append(neighbors.KNeighborsClassifier())
        knn_arr[num].fit(train_features[num], train_labels[num])

    del train_features
    del train_labels

    gc.collect()

    for num in range(1, test_set_num + 1):
        mat_data = sio.loadmat("/home/kaihang/DataSet_2/affNIST/test_batches/%d.mat" % num)
        test_images.append(np.transpose(mat_data["affNISTdata"][0, 0]["image"]).copy())
        test_labels.append(np.transpose(mat_data["affNISTdata"][0, 0]["label_int"]).copy().flatten())
        del mat_data

    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)

    print(test_images.shape, test_labels.shape)

    # vote for the result
    total_frame_sum = test_images.shape[0]
    correct_frame_sum = 0

    for frame_num in range(total_frame_sum):
        vote_arr = []
        for pca_num in range(train_set_num):
            test_feature = train_pca[pca_num].transform([test_images[frame_num]])[0]
            vote_arr.append(knn_arr[pca_num].predict([test_feature])[0])
        correct_frame_sum += int(test_labels[frame_num] == np.argmax(np.bincount(vote_arr)))
        print(vote_arr, test_labels[frame_num])
        print("Currently correct rate: %f" % (correct_frame_sum * 1.0 / (frame_num + 1)))

    print("Total frame sum: %d, correct frame num: %d" % (total_frame_sum, correct_frame_sum))

