import cv2
import numpy as np
import os
from os.path import join
import argparse
import pylab as pl
import matplotlib.pyplot as plt
from Utils import write_txt

def callback(x):
    pass

def annotate_region(model_id, img_path, mode):
    img = cv2.imread(img_path)
    row, col = img.shape[:2]

    plt.imshow(img[:,:,::-1])
    points = np.round(pl.ginput(1000, timeout = 10^10))
    xy = []
    xy.append([1, mode])
    for point in points:
        x = int(point[0])
        y = int(point[1])
        xy.append([x, y])
    xy = np.array(xy)

    n = len(os.listdir(join('annotations', model_id)))
    cv2.imwrite(join('annotations', model_id, str(n+1)+'.png'), img)
    write_txt(join('annotations', model_id, str(n+1)+'.txt'), xy, way='w')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--mode", type=int)
    args = parser.parse_args()

    annotate_region(model_id = args.model_id, img_path = args.img_path, mode = args.mode)
