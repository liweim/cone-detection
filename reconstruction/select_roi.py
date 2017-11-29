import cv2
import argparse
import os
from os.path import join
import numpy as np

def select_roi(model_id, plant_id, img_path):
    img_path = join('tmp', plant_id, plant_id, plant_id+'_'+img_path)
    img = cv2.imread(img_path)
    cv2.namedWindow('RoI', cv2.WINDOW_NORMAL)
    r = cv2.selectROI('RoI', img)
    row, col = img.shape[:2]
    cl = max(int(r[0]), 0)
    cr = min(int(r[0]+r[2]), col)
    rl = max(int(r[1]), 0)
    rr = min(int(r[1]+r[3]), row)
    background = img[rl:rr, cl:cr]

    mask = np.zeros(background.shape[:2])
    n = len(os.listdir(join('images', model_id)))
    basename = plant_id+'_'+str(n+1)+'.png'
    cv2.imwrite(join('images', model_id, basename), background)
    cv2.imwrite(join('annotations', model_id, basename), mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Which plant to detect.")
    parser.add_argument("--plant_id", type=str, help="Plant id.")
    parser.add_argument("--img_path", type=str, help="Image to analyze.")
    args = parser.parse_args()

    select_roi(model_id = args.model_id, plant_id = args.plant_id, img_path = args.img_path)
