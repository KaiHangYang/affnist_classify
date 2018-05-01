import tensorflow as tf
import numpy as np

import sys
sys.path.append("../")

from utils import path_definition as m_pd

class M_Model():
    def __init__(self, input_img_size, batch_size):
        self.input_size = input_img_size
        self.batch_size = batch_size

        self.loss = 0
        self.input_images = None
        self.gt_labels = None

        self.learning_rate = 0
        self.merged_summary = None

    def build_model(self, input_image):


