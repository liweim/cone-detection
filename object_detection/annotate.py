import numpy as np
import pandas as pd
from os.path import join
import os
from skimage.io import imread
import matplotlib.pyplot as plt
import pylab as pl
import random

def annotation(img_path, patch_size):
    basenames = os.listdir(img_path)
    dirname = os.path.split(img_path)[0]
    num = os.path.split(dirname)[1]

    radius = int((patch_size-1)/2)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    save_path = join('tmp', num, 'annotations')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(100):
        basename = random.choice(basenames)
        filename = os.path.splitext(basename)[0]
        save_path = join('tmp', num, 'annotations', filename+'.csv')
        if os.path.exists(save_path):
            continue
        img = imread(join(img_path, basename))
        r, c, n = img.shape
        plants = []

        plt.imshow(img)
        plt.title('Please click points')
        points = np.round(pl.ginput(500,timeout=10^10))

        for point in points:
            x = int(point[0])
            y = int(point[1])
            xmin = max(x-radius, 0)
            xmax = min(x+radius, c)
            ymin = max(y-radius, 0)
            ymax = min(y+radius, r)
            plants.append([basename, r, c, num, xmin, ymin, xmax, ymax])

        plant_df = pd.DataFrame(plants, columns=column_name)
        plant_df.to_csv(save_path, index=None)

    return

if __name__ == '__main__':
    annotation('tmp/995/images', 40)
    print('done!')
