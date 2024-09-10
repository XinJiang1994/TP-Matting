import cv2
import random
import numpy as np


def erode(x, k_size=5, it_e=1):
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    x = cv2.erode(x, kernel_erode, iterations=it_e)
    return x


def dilate(x, k_size=5, it_d=2):
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    x = cv2.dilate(x, kernel_dilate, iterations=it_d)
    return x


def random_erase(img, min_k_size=5, max_k_size=11, fill_with=-1):
    '''
      img: a tensor with shape [C,H,W]
    '''
    k_num = random.randint(3, 10)
    k_size = random.randint(min_k_size, max_k_size)
    for k in range(k_num):
        x = random.randint(0, img.shape[1]-k_size)
        y = random.randint(0, img.shape[2]-k_size)
        img[:, x:x+k_size, y:y +
            k_size] = np.random.rand(k_size, k_size) if fill_with == -1 else fill_with
    return img
