# This is the path definition file.
use_depthwise = False

######################### Train Data Path ########################
train_data_dir = "/home/kaihang/DataSet/affnist_tfrecord/train/"
valid_data_dir = "/home/kaihang/DataSet/affnist_tfrecord/valid/"
test_data_dir = "/home/kaihang/DataSet/affnist_tfrecord/test/"

######################### Train parameter ########################
trained_model_path = "/home/kaihang/Projects/affnist_classify/models/models"+("_dw" if use_depthwise else "")+"/trained_model-600000"
log_dir = "/home/kaihang/Projects/affnist_classify/logs/logs"+("_dw" if use_depthwise else "")+"/"
valid_log_dir = "/home/kaihang/Projects/affnist_classify/logs/valid_logs"+("_dw" if use_depthwise else "")+"/"

model_dir = "/home/kaihang/Projects/affnist_classify/models/models"+("_dw" if use_depthwise else "")+"/"

restore_pretrained_model = False
reset_global_step = False

input_img_size = 32

# Currently the two size must be the same, cause the implementation of loss builder
batch_size = 32

learning_rate = 0.05
learning_decay_rate = 0.333
learning_decay_step = 136106
train_iterations = 600000
valid_iterations = 5
