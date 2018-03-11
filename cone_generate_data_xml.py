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

def random_shift(x, y, max_pixel):
    x_shift = x + random.choice(range(-max_pixel, max_pixel))
    y_shift = y + random.choice(range(-max_pixel, max_pixel))
    return x_shift, y_shift

def generate_data_xml(annotation_path, data_path, no_resize):
    if no_resize:
        PATCH_SIZE = 25
        resize_rate = 0.5
        factor0 = 8
        factor100 = 12
    else:
        PATCH_SIZE = 32
        resize_rate = 0.999
        factor0 = 20
        factor100 = 24
    radius = int((PATCH_SIZE-1)/2)

    data_folder_path = join('tmp',data_path)
    if os.path.exists(data_folder_path):
        rmtree(data_folder_path)
    os.mkdir(data_folder_path)
    background_path = join(data_folder_path, '0')
    yellow_path = join(data_folder_path, '1')
    blue_path = join(data_folder_path, '2')
    orange_path = join(data_folder_path, '3')
    os.mkdir(background_path)
    os.mkdir(yellow_path)
    os.mkdir(blue_path)
    os.mkdir(orange_path)

    count0 = 0
    column_name = ['x', 'y', 'ratio', 'label']
    xml_path = join('annotations', annotation_path, '*.xml')
    for xml_file in glob.glob(xml_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        basename = os.path.splitext(xml_file)[0]
        img_path = basename+'.png'

        img = cv2.imread(img_path)
        img = imresize(img, resize_rate)
        row, col = img.shape[:2]
        mask = np.zeros(img.shape[:2]).astype(np.uint8)
        cones = []
        for member in root.findall('object'):
            label = member[0].text
            x1 = int(member[4][0].text)
            y1 = int(member[4][1].text)
            x2 = int(member[4][2].text)
            y2 = int(member[4][3].text)
            x = int((x1+x2)/2 * resize_rate)
            y = int((y1+y2)/2 * resize_rate)
            # x, y = random_shift(x, y, 3)
            max_length = max(abs(x2-x1), abs(y2-y1)) * resize_rate * 1.5
            ratio = max_length/PATCH_SIZE
            cones.append([x, y, ratio, label])
            cv2.circle(mask, (x, y), int(factor100*ratio), 100, -1)

        txt_path = basename+'.csv'
        df = pd.DataFrame(cones, columns=column_name)
        df.to_csv(txt_path, index=None)

        for x, y, ratio, label in cones:
            cv2.circle(mask, (x, y), int(factor0*ratio), 0, -1)
        for x, y, ratio, label in cones:
            if label == 'yellow':
                cv2.circle(mask, (x, y), 1, 253, -1)
            if label == 'blue':
                cv2.circle(mask, (x, y), 1, 254, -1)
            if label == 'orange':
                cv2.circle(mask, (x, y), 1, 255, -1)
        # mask[:radius, :] = 0
        # mask[img.shape[0]-radius:, :] = 0
        # mask[:, :radius] = 0
        # mask[:, img.shape[1]-radius:] = 0

        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        pickup_rate = np.sum(mask==255)/np.sum(mask==100)
        for x, y, ratio, label in cones:
            patch_radius = int(radius * ratio)
            roi_radius = int(factor100 * ratio)
            if no_resize:
                patch_radius = radius
            for c in range(max(x-roi_radius,patch_radius), min(x+roi_radius,col-patch_radius)):
                for r in range(max(y-roi_radius,patch_radius), min(y+roi_radius,row-patch_radius)):
                    if mask[r,c] == 100:
                        if random.random() < pickup_rate:
                            image = img[r-patch_radius:r+patch_radius+1, c-patch_radius:c+patch_radius+1]
                            image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
                            num = len(os.listdir(background_path))
                            cv2.imwrite(join(background_path, data_path+'_'+str(num)+'.png'), image)

                    if mask[r,c] > 100:
                        if mask[r,c] == 253:
                            save_folder_path = yellow_path
                        if mask[r,c] == 254:
                            save_folder_path = blue_path
                        if mask[r,c] == 255:
                            save_folder_path = orange_path
                        image = img[r-patch_radius:r+patch_radius+1, c-patch_radius:c+patch_radius+1]
                        image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
                        num = len(os.listdir(save_folder_path))
                        cv2.imwrite(join(save_folder_path, data_path+'_'+str(num)+'.png'), image)

    background_folder = 'annotations/background'
    path = join(background_folder, '*.txt')
    for txt_path in glob.glob(path):
        img_path = txt_path[:-3]+'png'
        img = cv2.imread(img_path)
        points = read_txt(txt_path)
        mode = int(points[0][1])
        row, col = img.shape[:2]
        for point in points:
            x = int(point[0])
            y = int(point[1])
            if mode == 1:
                image = img[max(y-radius,0):min(y+radius+1,row), max(x-radius,0):min(x+radius+1,col), :]
                image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
                num = len(os.listdir(background_path))
                cv2.imwrite(join(background_path, data_path+'_'+str(num)+'.png'), image)

    print('Background: {}'.format(len(os.listdir(background_path))))
    print('Yellow: {}'.format(len(os.listdir(yellow_path))))
    print('Blue: {}'.format(len(os.listdir(blue_path))))
    print('Orange: {}'.format(len(os.listdir(orange_path))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--no_resize', type=int)
    args = parser.parse_args()

    generate_data_xml(annotation_path = args.annotation_path, data_path = args.data_path, no_resize = args.no_resize)
