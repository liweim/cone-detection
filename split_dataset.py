import numpy as np
import os
from os.path import join
import cv2
import argparse
from random import random, choice
from Utils import read_txt, write_txt, read_csv
from scipy.misc import imresize
from shutil import rmtree, copyfile
import glob
import xml.etree.ElementTree as ET
import pandas as pd

patch_size = 64
radius = 32

def augmentation(img):
    random_rate = random()
    if random_rate > 0.7:
        R = int(patch_size*random_rate/2)
        img = img[radius-R:radius+R, radius-R:radius+R]
        img = cv2.resize(img, (patch_size, patch_size))
    return img

def split_dataset(data_paths):
    path = 'tmp/data'
    if os.path.exists(path):
        rmtree(path)
    for i in range(5):
        os.makedirs(join(path, 'train', str(i)))
        os.makedirs(join(path, 'test', str(i)))
    for data_path in data_paths:
        for id in ['0','1','2','3','4']:
            for img_path in glob.glob(join(data_path, id, '*.png')):
                img = cv2.imread(img_path)
                if random() < 0.7:
                    save_path = join('tmp/data/train', id)
                    # img = augmentation(img)
                else:
                    save_path = join('tmp/data/test', id)
                num = len(os.listdir(save_path))
                cv2.imwrite(join(save_path, str(num)+'.png'),img)

split_dataset(['data/circle/annotations','data/hairpin/annotations'])
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--annotation_paths', nargs='+')
#     parser.add_argument('--data_path', type=str)
#     args = parser.parse_args()

#     generate_data_xml(annotation_paths = args.annotation_paths, data_path = args.data_path)
