import numpy as np
import cv2

def preprocess(img):
    img = img / 255.0 - 0.5
    return img
