import cv2
import numpy as np
import os
from os.path import join
import argparse
import pylab as pl

def callback(x):
    pass

def annotate_region(cone_id, img_path):
    img = cv2.imread(img_path)
    row, col = img.shape[:2]
    mask = np.copy(img)
    mask[:] = 0

    r = cv2.selectROI('mask', img)
    cl = max(int(r[0]), 0)
    cr = min(int(r[0]+r[2]), col)
    rl = max(int(r[1]), 0)
    rr = min(int(r[1]+r[3]), row)
    img_roi = img[rl:rr, cl:cr]
    mask_roi = mask[rl:rr, cl:cr]

    result = cv2.hconcat((img_roi, mask_roi))
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    row, col = img_roi.shape[:2]

    while(1):
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break

        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        r = cv2.selectROI('mask', result)
        cl = max(int(r[0]), 0)
        cr = min(int(r[0]+r[2]), col)
        rl = max(int(r[1]), 0)
        rr = min(int(r[1]+r[3]), row)
        mask_roi[rl:rr, cl:cr] = img_roi[rl:rr, cl:cr]
        result = cv2.hconcat((img_roi, mask_roi))
        cv2.imshow('mask', result)

    n = len(os.listdir(join('images', cone_id)))
    basename = str(n+1)+'.png'
    cv2.imwrite(join('images', cone_id, basename), img_roi)
    cv2.imwrite(join('annotations', cone_id, basename), mask_roi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cone_id", type=str, help="Plant id.")
    parser.add_argument("--img_path", type=str, help="Image to analyze.")
    args = parser.parse_args()

    annotate_region(cone_id = args.cone_id, img_path = args.img_path)
