import cv2
import numpy as np
import os
from os.path import join
import pandas as pd
import random

def get_images(bbox_folder_path, zoom_rate = 1, bias_rate = 0):
    dirname = os.path.split(bbox_folder_path)[0]
    bbox_paths = os.listdir(bbox_folder_path)

    for bbox_path in bbox_paths:
        filename = os.path.splitext(bbox_path)[0]
        img_path = join(dirname, 'right', filename+'.png')
        img = cv2.imread(img_path)
        row, col, n = img.shape

        bbox_df = pd.read_csv(join(bbox_folder_path, bbox_path))
        boxes = np.array(bbox_df[:])
        labels = boxes[:,0]
        ymins = boxes[:,1]
        xmins = boxes[:,2]
        ymaxs = boxes[:,3]
        xmaxs = boxes[:,4]
        cxs = (xmins+xmaxs)/2
        cys = (ymins+ymaxs)/2

        for label, xmin, ymin, xmax, ymax, cx, cy in zip(labels, xmins, ymins, xmaxs, ymaxs, cxs, cys):
            length = int(max(xmax - xmin, ymax - ymin)/2)
            random_max = range(int(length/2), length)
            random_min = range(-length, int(-length/2))
            random_range = list(random_min) + list(random_max)
            cx = int(cx + random.choice(random_range) * bias_rate)
            cy = int(cy + random.choice(random_range) * bias_rate)
            length = int(max(xmax - xmin, ymax - ymin)/2 * zoom_rate)
            xl = max(cx-length, 0)
            xr = min(cx+length, row)
            yl = max(cy-length, 0)
            yr = min(cy+length, col)

            cone = img[xl:xr, yl:yr, :]
            cone = cv2.resize(cone, (100, 100))
            if bias_rate > 0:
                label = 'background'
            save_path = join('data', label)
            num = len(os.listdir(save_path))
            save_path = join(save_path, str(num+1)+'.png')
            cv2.imwrite(save_path, cone)

if __name__=='__main__':
	get_images(bbox_folder_path = 'video2/bbox', bias_rate = 1)
