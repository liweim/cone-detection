import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import pylab as pl
import os
from os.path import join
from Utils import read_txt, write_txt
import cv2
from rename_img import rename_img
import argparse

patch_size = 25

def annotation(img_path, data):
    dirname = os.path.split(img_path)[0]
    basename = os.path.split(img_path)[1]
    num = os.path.split(dirname)[1]

    radius = int((patch_size-1)/2)

    img = cv2.imread(img_path)

    if not os.path.exists(data):
        os.mkdir(data)

    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.title('Please click points')
    point = np.round(pl.ginput(500, timeout = 10^10))

    rename_img(data)
    n = len(os.listdir(data))

    for i in range(point.shape[0]):
        x = int(point[i, 0])
        y = int(point[i, 1])
        if x-radius>=0 and x+radius<img.shape[1] and y-radius>=0 and y+radius<img.shape[0]:
            n = n + 1
            temp = img[y-radius:y+radius+1, x-radius:x+radius+1, :]
            temp = cv2.resize(temp, (patch_size, patch_size))
            cv2.imwrite(join(data, str(n)+'.png'), temp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type = str)
    parser.add_argument("--data", type = str)
    args = parser.parse_args()

    annotation(img_path = args.img_path, data = args.data)
