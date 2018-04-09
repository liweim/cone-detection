import cv2
import numpy as np
import os
from os.path import join, split
import argparse
import pylab as pl
import matplotlib.pyplot as plt
from Utils import write_txt
import glob

patch_size = 25
radius = int((patch_size-1)/2)

def annotate_region(model_id, img_path, result_path, mode):
    basename = os.path.split(img_path)[1]
    img = cv2.imread(result_path)
    img_source = cv2.imread(img_path)
    row, col = img.shape[:2]
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    r = cv2.selectROI('mask', img)
    cl = max(int(r[0]), 0)
    cr = min(int(r[0]+r[2]), col)
    rl = max(int(r[1]), 0)
    rr = min(int(r[1]+r[3]), row)
    img_roi = img[rl:rr, cl:cr]
    img_source_roi = img_source[rl:rr, cl:cr]

    img_roi[radius, :] = (0,0,255)
    img_roi[img_roi.shape[0]-radius, :] = (0,0,255)
    img_roi[:, radius] = (0,0,255)
    img_roi[:, img_roi.shape[1]-radius] = (0,0,255)

    plt.imshow(img_roi[:,:,::-1])
    points = np.round(pl.ginput(1000, timeout = 10^10))
    xy = []
    xy.append([0, mode]) #good:0, bad:1, good+bad:-1
    for point in points:
        x = int(point[0])
        y = int(point[1])
        #radius = int(np.linalg.norm(point[0] - point[1]))
        xy.append([x, y])
    xy = np.array(xy)

    path = join('annotations', model_id)
    if not os.path.exists(path):
        os.mkdir(path)
    n = 1
    save_path = join('annotations', model_id, str(n))
    while  os.path.exists(save_path+'.png'):
        n = n+1
        save_path = join('annotations', model_id, str(n))

    cv2.imwrite(save_path+'.png', img_source_roi)
    write_txt(save_path+'.txt', xy, way='w')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Folder id to saving anntations and images.")
    parser.add_argument("--img_path", type=str, help="Image to analyze.")
    parser.add_argument("--result_path", type=str, help="Image to analyze.")
    parser.add_argument("--mode", type=int, help="Annotation mode.")
    args = parser.parse_args()

    annotate_region(model_id = args.model_id, img_path = args.img_path, result_path = args.result_path, mode = args.mode)
