import numpy as np
import os
from os.path import join
import cv2
import argparse
from random import random, choice
from Utils import read_txt, write_txt, isPointsInPolygons
from scipy.misc import imresize
from shutil import rmtree, copyfile
import glob
import xml.etree.ElementTree as ET
import pandas as pd

patch_size = 45
resize_rate = 0.5
factor0 = 8
factor100 = 12
radius = int((patch_size-1)/2)

# patch_size = 45
# resize_rate = 1
# factor0 = 20
# factor100 = 24
# radius = int((patch_size-1)/2)

def random_shift(x, y, max_pixel):
    x_shift = x + choice(range(-max_pixel, max_pixel))
    y_shift = y + choice(range(-max_pixel, max_pixel))
    return x_shift, y_shift

def augmentation(img):
    #zoom
    random_rate = random()
    if random_rate > 0.7:
        R = int(patch_size*random_rate/2)
        img = img[radius-R:radius+R, radius-R:radius+R]
        img = cv2.resize(img, (patch_size, patch_size))
    return img

def generate_data_xml(annotation_paths, data_path):
    if os.path.exists(data_path):
        rmtree(data_path)
    # os.mkdir(data_path)
    for i in range(5):
        os.makedirs(join(data_path, 'train', str(i)))
        os.makedirs(join(data_path, 'test', str(i)))
    classes = ['background', 'blue', 'yellow', 'orange', 'orange2']

    count0 = 0
    column_name = ['x', 'y', 'ratio', 'label']
    train_imgs = []
    for annotation_path in annotation_paths:
        xml_path = join(annotation_path, '*.xml')
        for xml_file in glob.glob(xml_path):
            print(xml_file)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            basename = os.path.splitext(xml_file)[0]
            img_path = basename+'.png'

            img = cv2.imread(img_path)
            img = imresize(img, 1.0*resize_rate)

            row, col = img.shape[:2]
            mask = np.zeros(img.shape[:2]).astype(np.uint8)
            cones = []
            
            for member in root.findall('object'):
                label = member[0].text
                x1 = int(int(member[4][0].text) * resize_rate)
                y1 = int(int(member[4][1].text) * resize_rate)
                x2 = int(int(member[4][2].text) * resize_rate)
                y2 = int(int(member[4][3].text) * resize_rate)
                x = int((x1+x2)/2)
                y = int((y1+y2)/2)

                # mask[y1:y2,x1:x2] = 0
                mask[max(0,y1-10):min(row,y2+10),max(0,x1-10):min(col,x2+10)] = 100
                mask[max(0,y1-3):min(row,y2+3),max(0,x1-3):min(col,x2+3)] = 0
                
                max_length = max(abs(x2-x1), abs(y2-y1)) * 1.5
                ratio = max_length/patch_size
                if ratio > 0.4:
                    cones.append([x1, y1, x2, y2, label, ratio])

            # txt_path = basename+'.csv'
            # column_name = ['x', 'y', 'label', 'ratio']
            # cone_df = pd.DataFrame(cones, columns=column_name)
            # cone_df.to_csv(txt_path, index=None, header=False)

            for x1, y1, x2, y2, label, ratio in cones:
                triangle = np.array([[(x1+x2)/2,y1],[x1,y2],[x2,y2]], np.int32)
                triangle = triangle.reshape((-1,1,2))
                if label == 'blue':
                    cv2.circle(mask, (x, y), 1, 252, -1)
                    cv2.polylines(mask,[triangle],True,252)
                if label == 'yellow':
                    cv2.circle(mask, (x, y), 2, 253, -1)
                    cv2.polylines(mask,[triangle],True,253)
                if label == 'orange':
                    cv2.circle(mask, (x, y), 2, 254, -1)
                    cv2.polylines(mask,[triangle],True,254)
                if label == 'orange2':
                    cv2.circle(mask, (x, y), 2, 255, -1)
                    cv2.polylines(mask,[triangle],True,255)
            mask[:radius, :] = 0
            mask[img.shape[0]-radius:, :] = 0
            mask[:, :radius] = 0
            mask[:, img.shape[1]-radius:] = 0

            # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
            # cv2.imshow('mask', mask)
            # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

            extra_back = 0
            while extra_back < 10:
                r = choice(range(row))
                c = choice(range(col))
                if mask[r,c] == 100:
                    image = img[r-radius:r+radius+1, c-radius:c+radius+1]
                    image = cv2.resize(image, (patch_size, patch_size))
                    if random() < 0.7:
                        path = join(data_path, 'train/0')
                        # train_imgs.append(image)
                    else:
                        path = join(data_path, 'test/0')
                    num = len(os.listdir(path))
                    cv2.imwrite(join(path, str(num)+'.png'), image)
                    extra_back += 1

            for r in range(row):
                for c in range(col):               
                    if mask[r,c] > 100 and random() < 0.2:
                        image = img[r-radius:r+radius+1, c-radius:c+radius+1]
                        image = cv2.resize(image, (patch_size, patch_size))
                        if random() < 0.7:
                            image = augmentation(image)
                            path = join(data_path, 'train')
                        else:
                            path = join(data_path, 'test')
                        if mask[r,c] == 252:
                            path = join(path, '1')
                        if mask[r,c] == 253:
                            path = join(path, '2')
                        if mask[r,c] == 254:
                            path = join(path, '3')
                        if mask[r,c] == 255:
                            path = join(path, '4')
                        num = len(os.listdir(path))
                        cv2.imwrite(join(path, str(num)+'.png'), image)

    background_folder = 'annotations/background'
    path = join(background_folder, '*.txt')
    for txt_path in glob.glob(path):
        img_path = txt_path[:-3]+'png'
        img = cv2.imread(img_path)
        points = read_txt(txt_path)
        row, col = img.shape[:2]
        for point in points[1:]:
            x = int(point[0])
            y = int(point[1])
            image = img[max(y-radius,0):min(y+radius+1,row), max(x-radius,0):min(x+radius+1,col), :]
            image = cv2.resize(image, (patch_size, patch_size))
            if random() < 0.7:
                image = augmentation(image)
                path = join(data_path, 'train', '0')
                train_imgs.append(image)
            else:
                path = join(data_path, 'test', '0')
            num = len(os.listdir(path))
            if np.max(image)>0:
                cv2.imwrite(join(path, str(num)+'.png'), image)
                # print(img_path)
                # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
                # cv2.imshow('mask', image)
                # cv2.waitKey(0)

    for i in range(5):
        path = join(data_path, 'train', str(i))
        print('{}: {}'.format(classes[i], len(os.listdir(path))))

    # train_imgs = np.array(train_imgs)
    # print(np.mean(train_imgs[:,:,:,0]))
    # print(np.mean(train_imgs[:,:,:,1]))
    # print(np.mean(train_imgs[:,:,:,2]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_paths', nargs='+')
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    generate_data_xml(annotation_paths = args.annotation_paths, data_path = args.data_path)
