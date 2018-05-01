import numpy as np
import cv2

def preprocess(img, centered_img):
    # img[img <= 100] = 128
    # centered_img[centered_img <= 100] = 128

    img = img / 255.0 - 0.5
    centered_img = centered_img / 255.0 - 0.5

    return img, centered_img
