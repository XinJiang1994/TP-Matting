# Copyright (C) 2024 Jiang Xin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

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
