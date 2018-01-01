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

    radius = 30
    for point in points:
        xy = []
        xy.append([1, mode]) #good:0, bad:1, good+bad:-1
        xy.append([30, 30])
        x = int(point[0])
        y = int(point[1])
        cl = max(int(y-radius), 0)
        cr = min(int(y+radius), row)
        rl = max(int(x-radius), 0)
        rr = min(int(x+radius), col)
        img_roi = img[cl:cr,rl:rr]
        n = len(os.listdir(join('images', model_id)))
        cv2.imwrite(join('images', model_id, str(n+1)+'.png'), img_roi)
        write_txt(join('annotations', model_id, str(n+1)+'.txt'), np.array(xy), way='w')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--mode", type=int)
    args = parser.parse_args()

    annotate_region(model_id = args.model_id, img_path = args.img_path, mode = args.mode)
