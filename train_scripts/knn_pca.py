import numpy as np
from sklearn import neighbors
from sklearn.decomposition import PCA
import scipy.io as sio
import time
import gc


if __name__ == "__main__":
    # load training set range(0, 30)
    voter_num = 10
    voter_size = 3
    n_components = 128

    voter_features = []
    voter_labels = []
    voter_pca = []
    voter_knn_arr = []

    train_set_num = 24
    test_set_num = 32

    train_images = []
    train_labels = []
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

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # force the garbage collector
    gc.collect()

    start = time.clock()

    selected_arr = range(train_set_num)
    for num in range(voter_num):
        print("Currently process voter num: %d" % num)
        np.random.shuffle(selected_arr)

        concat_images = np.concatenate(train_images[selected_arr[0:voter_size]])

        voter_pca.append(PCA(n_components=n_components, copy=False, whiten=True, svd_solver='randomized').fit(concat_images))
        voter_features.append(voter_pca[num].transform(concat_images))
        voter_labels.append(np.concatenate(train_labels[selected_arr[0:voter_size]]))

    del train_images
    del train_labels
    gc.collect()

    end = time.clock()
    print("Transform time %fs" % (end - start))

    for num in range(voter_num):
        print("Current fitting voter num: %d" % num)
        voter_knn_arr.append(neighbors.KNeighborsClassifier())
        voter_knn_arr[num].fit(voter_features[num], voter_labels[num])

    del voter_features
    del voter_labels
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

    with open("../logs/knn_pca.log", "w") as f:
        for frame_num in range(total_frame_sum):
            vote_arr = []
            for pca_num in range(voter_num):
                test_feature = voter_pca[pca_num].transform([test_images[frame_num]])[0]
                vote_arr.append(voter_knn_arr[pca_num].predict([test_feature])[0])
            correct_frame_sum += int(test_labels[frame_num] == np.argmax(np.bincount(vote_arr)))
            f.write(str((vote_arr, test_labels[frame_num])) + "\n")
            f.write("Currently correct rate: %f( %d / %d / %d)\n" % (correct_frame_sum * 1.0 / (frame_num + 1), correct_frame_sum, frame_num + 1, total_frame_sum))

        f.write("Total frame sum: %d, correct frame num: %d" % (total_frame_sum, correct_frame_sum))

