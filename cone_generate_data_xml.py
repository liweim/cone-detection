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

patch_size = 25
resize_rate = 0.5
factor0 = 8
factor100 = 12
radius = int((patch_size-1)/2)

# patch_size = 32
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

def generate_data_xml(annotation_path, data_path, no_resize):
    data_folder_path = join('tmp',data_path)
    if os.path.exists(data_folder_path):
        rmtree(data_folder_path)
    # os.mkdir(data_folder_path)
    for i in range(4):
        os.makedirs(join(data_folder_path, 'train', str(i)))
        os.makedirs(join(data_folder_path, 'test', str(i)))
    classes = ['background', 'yellow', 'blue', 'orange']

    count0 = 0
    column_name = ['x', 'y', 'ratio', 'label']
    xml_path = join('annotations', annotation_path, '*.xml')
    for xml_file in glob.glob(xml_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        basename = os.path.splitext(xml_file)[0]
        img_path = basename+'.png'

        img = cv2.imread(img_path)
        img = imresize(img, 1.0*resize_rate)
        # mask_img = np.copy(img)
        mask_img = np.random.random_integers(0, 255, img.shape).astype(np.uint8)
        row, col = img.shape[:2]
        mask = np.zeros(img.shape[:2]).astype(np.uint8)
        cones = []
        polygonset = []
        for member in root.findall('object'):
            label = member[0].text
            x1 = int(int(member[4][0].text) * resize_rate)
            y1 = int(int(member[4][1].text) * resize_rate)
            x2 = int(int(member[4][2].text) * resize_rate)
            y2 = int(int(member[4][3].text) * resize_rate)
            x = int((x1+x2)/2)
            y = int((y1+y2)/2)
            # polygonset.append([[y1,(x1+x2)/2],[y2,x1],[y2,x2]])
            triangle = np.array([[(x1+x2)/2,y1],[x1,y2],[x2,y2]], np.int32)
            triangle = triangle.reshape((-1,1,2))
            cv2.polylines(mask,[triangle],True,100)
            # cv2.polylines(mask_img,[triangle],True,(0,0,255))
                        
            # x, y = random_shift(x, y, 3)
            max_length = max(abs(x2-x1), abs(y2-y1)) * 1.5
            ratio = max_length/patch_size
            mask_img[y1:y2+1,x1:x2+1] = img[y1:y2+1,x1:x2+1]
            tmp = 1.5*ratio
            for r in range(max(0,int(y-3*ratio)),min(row,int(y+1*ratio)+1)):
                tmp += 1/4
                for c in range(max(0,int(x-tmp)+1),min(col,int(x+tmp)+1)):
                    mask_img[r, c] = [choice(range(256)), choice(range(256)), choice(range(256))]
            # print(ratio)
            if ratio > 0.5:
                cones.append([x, y, ratio, label])
                # cv2.circle(mask, (x, y), int(factor100*ratio), 100, -1)

        txt_path = basename+'.csv'
        df = pd.DataFrame(cones, columns=column_name)
        df.to_csv(txt_path, index=None)

        # pts = []
        # for r in range(row):
        #     for c in range(col):
        #         pts.append([c,r])
        # inliers, outliers = isPointsInPolygons(pts, polygonset)
        # for r, c in outliers:
        #     mask_img[r, c] = [choice(range(256)), choice(range(256)), choice(range(256))]


        # for x, y, ratio, label in cones:
        #     cv2.circle(mask, (x, y), int(factor0*ratio), 0, -1)
        for x, y, ratio, label in cones:
            if label == 'yellow':
                cv2.circle(mask, (x, y), 2, 253, -1)
            if label == 'blue':
                cv2.circle(mask, (x, y), 2, 254, -1)
            if label == 'orange':
                cv2.circle(mask, (x, y), 2, 255, -1)
        mask[:radius, :] = 0
        mask[img.shape[0]-radius:, :] = 0
        mask[:, :radius] = 0
        mask[:, img.shape[1]-radius:] = 0

        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        # cv2.imshow('mask', mask_img)
        # cv2.waitKey(0)

        pickup_rate = np.sum(mask>100)/np.sum(mask==100)/6
        for x, y, ratio, label in cones:
            patch_radius = int(radius * ratio)
            roi_radius = int(factor100 * ratio)
            if no_resize:
                patch_radius = radius
            for c in range(max(x-roi_radius,patch_radius), min(x+roi_radius,col-patch_radius)):
                for r in range(max(y-roi_radius,patch_radius), min(y+roi_radius,row-patch_radius)):
                    flag = 0
                    if random() < 0.7:
                        image = mask_img[r-patch_radius:r+patch_radius+1, c-patch_radius:c+patch_radius+1]
                        image = cv2.resize(image, (patch_size, patch_size))
                        image = augmentation(image)
                        path = join(data_folder_path, 'train')
                    else:
                        image = mask_img[r-patch_radius:r+patch_radius+1, c-patch_radius:c+patch_radius+1]
                        image = cv2.resize(image, (patch_size, patch_size))
                        path = join(data_folder_path, 'test')
                    if mask[r,c] == 100:
                        if random() < pickup_rate: 
                            path = join(path, '0')   
                            flag = 1                      
                    if mask[r,c] > 100:
                        flag = 1
                        if mask[r,c] == 253:
                            path = join(path, '1')
                        if mask[r,c] == 254:
                            path = join(path, '2')
                        if mask[r,c] == 255:
                            path = join(path, '3')
                    if flag:
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
                path = join(data_folder_path, 'train', '0')
            else:
                path = join(data_folder_path, 'test', '0')
            num = len(os.listdir(path))
            cv2.imwrite(join(path, str(num)+'.png'), image)

    for i in range(4):
        path = join(data_folder_path, 'train', str(i))
        print('{}: {}'.format(classes[i], len(os.listdir(path))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--no_resize', type=int)
    args = parser.parse_args()

    generate_data_xml(annotation_path = args.annotation_path, data_path = args.data_path, no_resize = args.no_resize)
