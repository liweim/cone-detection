from sklearn.model_selection import train_test_split
import numpy as np
import os
from os.path import join
import cv2
import argparse
from random import random, choice
from Utils import read_txt
from scipy.misc import imresize
from shutil import rmtree

PATCH_SIZE = 32
radius = int((PATCH_SIZE-1)/2)

def generate_data(model_ids, train_num, annotate_radius):
    data_path = join('tmp','data')
    if os.path.exists(data_path):
        rmtree(data_path)
    os.mkdir(data_path)
    os.mkdir(join(data_path, '0'))
    print('Train num: {}'.format(train_num))

    count0 = 0
    for model_id in model_ids:
        os.mkdir(join(data_path, model_id))
        imgs = []
        masks = []
        flags = []
        back_areas = [0]
        front_areas = [0]
        image_folder_path = join('images', model_id)
        annotation_folder_path = join('annotations', model_id)
        img_paths = os.listdir(image_folder_path)
        for img_path in img_paths:
            filename = os.path.splitext(img_path)[0]
            img_path = join(image_folder_path, img_path)
            img = cv2.imread(img_path)
            points = read_txt(join(annotation_folder_path, filename+'.txt'))
            resize_rate = float(points[0][0])
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
                    cv2.circle(mask, (x, y), annotate_radius*3, 100, -1)
            if mode == -1:
                for x, y in zip(xs, ys):
                    cv2.circle(mask, (x, y), annotate_radius*2, 0, -1)
            for x, y in zip(xs, ys):
                if mode == 1:
                    cv2.circle(mask, (x, y), annotate_radius, 100, -1)
                else:
                    cv2.circle(mask, (x, y), annotate_radius, 255, -1)
            mask[:radius, :] = 0
            mask[img.shape[0]-radius:, :] = 0
            mask[:, :radius] = 0
            mask[:, img.shape[1]-radius:] = 0

            cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
            cv2.imshow('mask', mask)
            cv2.waitKey(0)

            imgs.append(img)
            masks.append(mask)
            flags.append(np.zeros(mask.shape))
            front_area = np.sum(mask==255)
            back_area = np.sum(mask==100)
            front_areas.append(front_area)
            back_areas.append(back_area)

        sum_front_area = np.sum(front_areas)
        sum_back_area = np.sum(back_areas)
        print('Frontground area: {}'.format(sum_front_area))
        print('Background area: {}'.format(sum_back_area))
        if train_num > sum_front_area:
            print('Train number should be less than {}'. format(sum_front_area))
            return
        if train_num > sum_back_area:
            print('Train number should be less than {}'. format(sum_back_area))
            return

        front_areas /= sum_front_area
        back_areas /= sum_back_area
        for i in range(1, len(back_areas)):
            mask = masks[i-1]
            front_areas[i] = front_areas[i] + front_areas[i-1]
            back_areas[i] = back_areas[i] + back_areas[i-1]
            # if mask.max() == 100:
            #     back_areas[i] *= 2
        back_areas /= back_areas[-1]

        num_back = 0
        num_front = 0
        for count in range(train_num):
            rd = random()
            for i in range(len(imgs)):
                img_path = img_paths[i]
                img = imgs[i]
                mask = masks[i]
                rs, cs = mask.shape
                if rs-radius <= radius or cs-radius <= radius:
                    print(img_path + ' size should be at least 45x45')
                    return
                r = choice(range(radius, rs-radius))
                c = choice(range(radius, cs-radius))
                if rd < back_areas[i+1] and rd > back_areas[i]:
                    while not (mask[r, c] == 100 and flags[i][r, c] < 1):
                        r = choice(range(radius, rs-radius))
                        c = choice(range(radius, cs-radius))
                    image = img[r-radius:r+radius+1, c-radius:c+radius+1, :]
                    image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
                    cv2.imwrite(join(data_path, '0', str(count0)+'.png'), image)
                    count0 += 1
                    num_back += 1
                    flags[i][r, c] = 1

        for count in range(train_num):
            rd = random()
            for i in range(len(imgs)):
                img_path = img_paths[i]
                img = imgs[i]
                mask = masks[i]
                rs, cs = mask.shape
                if rs-radius < radius or cs-radius < radius:
                    print(img_path + ' size should be at least 45x45')
                    return
                r = choice(range(radius, rs-radius))
                c = choice(range(radius, cs-radius))
                if rd < front_areas[i+1] and rd > front_areas[i]:
                    while not (mask[r, c] == 255 and flags[i][r, c] < 1):
                    #while not mask[r, c] == 255:
                        r = choice(range(radius, rs-radius))
                        c = choice(range(radius, cs-radius))
                    image = img[r-radius:r+radius+1, c-radius:c+radius+1, :]
                    image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
                    cv2.imwrite(join(data_path, model_id, str(count)+'.png'), image)
                    num_front += 1
                    flags[i][r, c] = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nargs', nargs='+')
    parser.add_argument("--train_num", type=int)
    parser.add_argument("--annotate_radius", type=int)
    args = parser.parse_args()

    generate_data(model_ids = args.nargs, train_num = args.train_num, annotate_radius = args.annotate_radius)
