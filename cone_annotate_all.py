import cv2
import numpy as np
import os
from os.path import join, split
import argparse
import pylab as pl
import matplotlib.pyplot as plt
from Utils import write_csv
import glob

patch_size = 25
radius = int((patch_size-1)/2)

def annotate(img_path):
    img = cv2.imread(img_path)
    row, col = img.shape[:2]
    cv2.line(img, (0,210), (col,210), (0,0,255), 2)
    cv2.line(img, (0,320), (col,320), (0,0,255), 2)

    xy = []
    for label in ['blue', 'yellow', 'orange', 'orange2']:
        plt.imshow(img[:,:,::-1])
        plt.title('annotation '+label+' cones')
        points = np.round(pl.ginput(1000, timeout = 10^10))
        for point in points:
            x = int(point[0])
            y = int(point[1])
            xy.append([x, y, label])
    xy = np.array(xy)

    write_csv(img_path[:-3]+'csv', xy)

def annotate_all():
    for img_path in glob.glob('annotations/circle/results_circle_perfect/*.png'):
        annotate(img_path) 

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--img_path", type=str, help="Image to analyze.")
    # args = parser.parse_args()

    annotate_all()
