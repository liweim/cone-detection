import cv2
import numpy as np
import os
from os.path import join, split
import argparse
import pylab as pl
import matplotlib.pyplot as plt
from Utils import write_txt
import glob

def callback(x):
    pass

def annotate_region(model_id, img_path,  mode):
    plant_id = split(split(split(img_path)[0])[0])[1]
    img_path = split(img_path)[1]
    mark_path = join('tmp', plant_id, 'result', img_path)
    source_path = join('tmp', plant_id, 'images', img_path)
    if not os.path.exists(mark_path):
        mark_path = source_path
    filename = os.path.splitext(mark_path)[0]
    img = cv2.imread(mark_path)

    img_source = cv2.imread(source_path)

    row, col = img.shape[:2]

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    r = cv2.selectROI('mask', img)
    cl = max(int(r[0]), 0)
    cr = min(int(r[0]+r[2]), col)
    rl = max(int(r[1]), 0)
    rr = min(int(r[1]+r[3]), row)
    img_roi = img[rl:rr, cl:cr]
    img_source_roi = img_source[rl:rr, cl:cr]

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
    basename = plant_id+'_'+str(n)
    save_path = join('annotations', model_id, basename+'.png')
    while  os.path.exists(save_path):
        n = n+1
        basename = plant_id+'_'+str(n)
        save_path = join('annotations', model_id, basename+'.png')

    cv2.imwrite(join('annotations', model_id, basename+'.png'), img_source_roi)
    write_txt(join('annotations', model_id, basename+'.txt'), xy, way='w')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Folder id to saving anntations and images.")
    parser.add_argument("--img_path", type=str, help="Image to analyze.")
    parser.add_argument("--mode", type=int, help="Annotation mode.")
    args = parser.parse_args()

    annotate_region(model_id = args.model_id, img_path = args.img_path, mode = args.mode)
