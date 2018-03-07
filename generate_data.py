from sklearn.model_selection import train_test_split
import numpy as np
import os
from os.path import join
import cv2
import argparse
from random import random, choice
from Utils import read_txt, std_color
from scipy.misc import imresize
from shutil import rmtree
import pandas as pd
from random import random, choice
import glob

PATCH_SIZE = 45
radius = int((PATCH_SIZE-1)/2)

def augmentation(img):
    zoom_rate = random()

    if random() > 0.7:
        M = cv2.getRotationMatrix2D((radius, radius), 90*choice(range(4)), 1)
        img = cv2.warpAffine(img, M, (PATCH_SIZE,PATCH_SIZE))

    if zoom_rate > 0.7:
        R = int(PATCH_SIZE*zoom_rate/2)
        img = img[radius-R:radius+R, radius-R:radius+R]
        img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE))

    return img

def generate_data(model_ids):
    data_path = join('tmp','data')
    if os.path.exists(data_path):
        rmtree(data_path)
    os.mkdir(data_path)
    os.mkdir(join(data_path, 'train'))
    os.mkdir(join(data_path, 'test'))
    os.mkdir(join(data_path, 'train', '0'))
    os.mkdir(join(data_path, 'test', '0'))

    df = pd.read_csv('plant_info.csv')

    global_back_area = 0
    label = 0
    for model_id in model_ids:
        label += 1
        os.mkdir(join(data_path, 'train', str(label)))
        os.mkdir(join(data_path, 'test', str(label)))
        annotation_folder_path = join('annotations', model_id)
        img_paths = glob.glob(annotation_folder_path + '/*.png')

        front_area = 0
        back_area = 0
        for img_path in img_paths:
            print(img_path)
            basename = os.path.split(img_path)[1]
            plant_id = basename.split('_')[0]
            df_plant = df[df['plant_id'] == int(plant_id)]
            resize_rate = float(df_plant['resize_rate'])
            resize_rate = min(resize_rate, 1.5)
            plant_color = int(df_plant['plant_color'])

            filename = os.path.splitext(basename)[0]
            img = cv2.imread(img_path)
            img = std_color(img, plant_color)
            points = read_txt(join(annotation_folder_path, filename+'.txt'))
            mode = int(points[0][1])
            if not resize_rate == 1:
                img = imresize(img, resize_rate)
            mask = np.zeros(img.shape[:2]).astype(np.uint8)
            xs = []
            ys = []
            for point in points[1:]:
                x = int(point[0] * resize_rate)
                y = int(point[1] * resize_rate)
                xs.append(x)
                ys.append(y)
                if mode == -1:
                    cv2.circle(mask, (x, y), 12, 100, -1)
            if mode == -1:
                for x, y in zip(xs, ys):
                    cv2.circle(mask, (x, y), 6, 0, -1)
            for x, y in zip(xs, ys):
                if mode == 1:
                    cv2.circle(mask, (x, y), 2, 200, -1)
                else:
                    cv2.circle(mask, (x, y), 2, 255, -1)
            mask[:radius, :] = 0
            mask[img.shape[0]-radius:, :] = 0
            mask[:, :radius] = 0
            mask[:, img.shape[1]-radius:] = 0

            if mode == -1:
                pickup_rate = np.sum(mask==255)/np.sum(mask==100)
            # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
            # cv2.imshow('mask', mask)
            # cv2.waitKey(0)

            for r in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    if mask[r, c] > 0:
                        flag = 1
                        image = img[r-radius:r+radius+1, c-radius:c+radius+1, :]
                        if random() < 0.7:
                            image = augmentation(image)
                            path = join(data_path, 'train')
                        else:
                            path = join(data_path, 'test')
                        if mask[r, c] == 100:
                            if random() < pickup_rate:
                                back_area += 1
                                global_back_area += 1
                                save_path = join(path, '0')
                            else:
                                flag = 0
                        if mask[r, c] == 200:
                            back_area += 1
                            global_back_area += 1
                            save_path = join(path, '0')
                        if mask[r, c] == 255:
                            front_area += 1
                            save_path = join(path, str(label))
                        if flag:
                            num = str(len(os.listdir(save_path)))
                            cv2.imwrite(join(save_path, num+'.png'), image)

        print('Frontground area: {}'.format(front_area))
        print('Background area: {}'.format(back_area))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ids', nargs='+', help="Which direction to train.")
    args = parser.parse_args()

    generate_data(model_ids = args.model_ids)
