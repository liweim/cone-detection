import numpy as np
import os
from os.path import join
import cv2
import argparse
from random import random, choice
from Utils import read_txt, write_txt
from scipy.misc import imresize
from shutil import rmtree, copyfile
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import random

PATCH_SIZE = 32
radius = int((PATCH_SIZE-1)/2)

def generate_data(path):
    data_path = join('tmp','data')
    if os.path.exists(data_path):
        rmtree(data_path)
    os.mkdir(data_path)
    background_path = join(data_path, '0')
    yellow_path = join(data_path, '1')
    blue_path = join(data_path, '2')
    orange_path = join(data_path, '3')
    os.mkdir(background_path)
    os.mkdir(yellow_path)
    os.mkdir(blue_path)
    os.mkdir(orange_path)

    count0 = 0
    column_name = ['x', 'y', 'ratio', 'label']
    xml_path = join('annotations', path, '*.xml')
    for xml_file in glob.glob(xml_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        basename = os.path.splitext(xml_file)[0]
        img_path = basename+'.png'

        img = cv2.imread(img_path)
        row, col = img.shape[:2]
        mask = np.zeros(img.shape[:2]).astype(np.uint8)
        cones = []
        for member in root.findall('object'):
            label = member[0].text
            x1 = int(member[4][0].text)
            y1 = int(member[4][1].text)
            x2 = int(member[4][2].text)
            y2 = int(member[4][3].text)
            x = int((x1+x2)/2)
            y = int((y1+y2)/2)
            max_length = max(abs(x2-x1), abs(y2-y1))
            ratio = max_length/PATCH_SIZE
            cones.append([x, y, ratio, label])
            cv2.circle(mask, (x, y), int(16*ratio), 100, -1)
        txt_path = basename+'.csv'
        df = pd.DataFrame(cones, columns=column_name)
        df.to_csv(txt_path, index=None)

        for x, y, ratio, label in cones:
            cv2.circle(mask, (x, y), int(8*ratio), 0, -1)
        for x, y, ratio, label in cones:
            cv2.circle(mask, (x, y), int(1*ratio), 255, -1)
        # mask[:radius, :] = 0
        # mask[img.shape[0]-radius:, :] = 0
        # mask[:, :radius] = 0
        # mask[:, img.shape[1]-radius:] = 0

        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        pickup_rate = np.sum(mask==255)/np.sum(mask==100)/3
        for x, y, ratio, label in cones:
            if label == 'yellow':
                save_folder_path = yellow_path
            if label == 'blue':
                save_folder_path = blue_path
            if label == 'orange':
                save_folder_path = orange_path
            patch_radius = int(radius * ratio)
            roi_radius = int(16*ratio)
            for c in range(max(x-roi_radius,patch_radius), min(x+roi_radius,col-patch_radius)):
                for r in range(max(y-roi_radius,patch_radius), min(y+roi_radius,row-patch_radius)):
                    if mask[r,c] == 100:
                        if random.random() < pickup_rate:
                            image = img[r-patch_radius:r+patch_radius+1, c-patch_radius:c+patch_radius+1]
                            image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
                            num = len(os.listdir(background_path))
                            cv2.imwrite(join(background_path, str(num)+'.png'), image)
                    if mask[r,c] == 255:
                        image = img[r-patch_radius:r+patch_radius+1, c-patch_radius:c+patch_radius+1]
                        image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
                        num = len(os.listdir(save_folder_path))
                        cv2.imwrite(join(save_folder_path, str(num)+'.png'), image)

    print('Background: {}'.format(len(os.listdir(background_path))))
    print('Yellow: {}'.format(len(os.listdir(yellow_path))))
    print('Blue: {}'.format(len(os.listdir(blue_path))))
    print('Orange: {}'.format(len(os.listdir(orange_path))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    generate_data(path = args.path)
