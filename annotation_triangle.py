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
import pandas as pd

for num in range(10):
    img_path = 'annotations/snow/'+str(num)+'.png'
    print(img_path)
    basename = os.path.splitext(img_path)[0]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (320,180))
    cones = []
    for label_id, label_name in enumerate(['blue-top', 'blue-left', 'blue-right', 'yellow-top', 'yellow-left', 'yellow-right', 'orange-top', 'orange-left', 'orange-right']):
        plt.imshow(img[:,:,::-1])
        plt.title('Please click {} cones'.format(label_name))
        point = np.round(pl.ginput(500, timeout = 10^10))

        for i in range(point.shape[0]):
            x = int(point[i, 0])
            y = int(point[i, 1])
            cones.append([x, y, label_id, 0])

    txt_path = basename+'_triangle.csv'
    column_name = ['x', 'y', 'label', 'ratio']
    cone_df = pd.DataFrame(cones, columns=column_name)
    cone_df.to_csv(txt_path, index=None, header=False)