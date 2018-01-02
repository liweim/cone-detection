from sklearn.model_selection import train_test_split
import numpy as np
import os
from os.path import join
import cv2
import argparse
from random import random, choice
from Utils import read_txt
from scipy.misc import imresize
from shutil import rmtree, copyfile
import glob
import xml.etree.ElementTree as ET

PATCH_SIZE = 32
radius = int((PATCH_SIZE-1)/2)

def generate_data(path, annotate_radius):
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
    xml_path = join('annotations', path, '*.xml')
    img_folder_path = join('images', path)
    for xml_file in glob.glob(xml_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        img_path = join(img_folder_path, root.find('filename').text)
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape[:2]).astype(np.uint8)
        labels = []
        xs = []
        ys = []
        for member in root.findall('object'):
            labels.append(member[0].text)
            x1 = int(member[4][0].text)
            y1 = int(member[4][1].text)
            x2 = int(member[4][2].text)
            y2 = int(member[4][3].text)
            x = int((x1+x2)/2)
            y = int((y1+y2)/2)
            xs.append(x)
            ys.append(y)
            cv2.circle(mask, (x, y), annotate_radius*5, 100, -1)
        for x, y in zip(xs, ys):
            cv2.circle(mask, (x, y), annotate_radius*2, 0, -1)
        for x, y in zip(xs, ys):
            cv2.circle(mask, (x, y), annotate_radius, 255, -1)
        mask[:radius, :] = 0
        mask[img.shape[0]-radius:, :] = 0
        mask[:, :radius] = 0
        mask[:, img.shape[1]-radius:] = 0

        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        count = 0
        for label, x, y in zip(labels, xs, ys):
            if label == 'yellow':
                save_folder_path = yellow_path
            if label == 'blue':
                save_folder_path = blue_path
            if label == 'orange':
                save_folder_path = orange_path
            for c in range(x-annotate_radius, x+annotate_radius):
                for r in range(y-annotate_radius, y+annotate_radius):
                    if mask[r,c] == 255:
                        count += 1
                        image = img[r-radius:r+radius+1, c-radius:c+radius+1]
                        image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
                        num = len(os.listdir(save_folder_path))
                        cv2.imwrite(join(save_folder_path, str(num)+'.png'), image)

        flag = np.zeros(mask.shape)
        for c in range(count):
            rs, cs = mask.shape
            r = choice(range(radius, rs-radius))
            c = choice(range(radius, cs-radius))
            while not (mask[r, c] == 100 and flag[r, c] < 1):
                r = choice(range(radius, rs-radius))
                c = choice(range(radius, cs-radius))
            image = img[r-radius:r+radius+1, c-radius:c+radius+1, :]
            image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
            cv2.imwrite(join(data_path, '0', str(count0)+'.png'), image)
            count0 += 1
            flag[r, c] = 1

    print('Background: {}'.format(len(os.listdir(background_path))))
    print('Yellow: {}'.format(len(os.listdir(yellow_path))))
    print('Blue: {}'.format(len(os.listdir(blue_path))))
    print('Orange: {}'.format(len(os.listdir(orange_path))))

    # count0 = 0
    # for model_id in model_ids:
    #     os.mkdir(join(data_path, model_id))
    #     imgs = []
    #     masks = []
    #     flags = []
    #     back_areas = [0]
    #     front_areas = [0]
    #     image_folder_path = join('images', model_id)
    #     annotation_folder_path = join('annotations', model_id)
    #     annotation_paths = os.listdir(annotation_folder_path)
    #     for annotation_path in annotation_paths:
    #         filename = os.path.splitext(annotation_path)[0]
    #         img_path = join(image_folder_path, img_path)
    #         img = cv2.imread(img_path)
    #         points = read_txt(join(annotation_folder_path, filename+'.txt'))
    #         resize_rate = float(points[0][0])
    #         mode = int(points[0][1])
    #         if not resize_rate == 1:
    #             img = imresize(img, resize_rate)
    #         mask = np.zeros(img.shape[:2]).astype(np.uint8)
    #         xs = []
    #         ys = []
    #         for point in points[1:]:
    #             x = int(point[0] * resize_rate)
    #             y = int(point[1] * resize_rate)
    #             xs.append(x)
    #             ys.append(y)
    #             if mode == -1:
    #                 cv2.circle(mask, (x, y), radius*3, 100, -1)
    #         if mode == -1:
    #             for x, y in zip(xs, ys):
    #                 cv2.circle(mask, (x, y), radius*2, 0, -1)
    #         for x, y in zip(xs, ys):
    #             if mode == 1:
    #                 cv2.circle(mask, (x, y), radius, 100, -1)
    #             else:
    #                 cv2.circle(mask, (x, y), radius, 255, -1)
    #         mask[:radius, :] = 0
    #         mask[img.shape[0]-radius:, :] = 0
    #         mask[:, :radius] = 0
    #         mask[:, img.shape[1]-radius:] = 0
    #
    #         # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    #         # cv2.imshow('mask', mask)
    #         # cv2.waitKey(0)
    #
    #         imgs.append(img)
    #         masks.append(mask)
    #         flags.append(np.zeros(mask.shape))
    #         front_area = np.sum(mask==255)
    #         back_area = np.sum(mask==100)
    #         front_areas.append(front_area)
    #         back_areas.append(back_area)
    #
    #     sum_front_area = np.sum(front_areas)
    #     sum_back_area = np.sum(back_areas)
    #     print('Frontground area: {}'.format(sum_front_area))
    #     print('Background area: {}'.format(sum_back_area))
    #     if train_num > sum_front_area:
    #         print('Train number should be less than {}'. format(sum_front_area))
    #         return
    #     if train_num > sum_back_area:
    #         print('Train number should be less than {}'. format(sum_back_area))
    #         return
    #
    #     front_areas /= sum_front_area
    #     back_areas /= sum_back_area
    #     for i in range(1, len(back_areas)):
    #         mask = masks[i-1]
    #         front_areas[i] = front_areas[i] + front_areas[i-1]
    #         back_areas[i] = back_areas[i] + back_areas[i-1]
    #         # if mask.max() == 100:
    #         #     back_areas[i] *= 2
    #     back_areas /= back_areas[-1]
    #
    #     num_back = 0
    #     num_front = 0
    #     for count in range(train_num):
    #         rd = random()
    #         for i in range(len(imgs)):
    #             img_path = img_paths[i]
    #             img = imgs[i]
    #             mask = masks[i]
    #             rs, cs = mask.shape
    #             if rs-radius <= radius or cs-radius <= radius:
    #                 print(img_path + ' size should be at least 45x45')
    #                 return
    #             r = choice(range(radius, rs-radius))
    #             c = choice(range(radius, cs-radius))
    #             if rd < back_areas[i+1] and rd > back_areas[i]:
    #                 while not (mask[r, c] == 100 and flags[i][r, c] < 1):
    #                     r = choice(range(radius, rs-radius))
    #                     c = choice(range(radius, cs-radius))
    #                 image = img[r-radius:r+radius+1, c-radius:c+radius+1, :]
    #                 image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
    #                 cv2.imwrite(join(data_path, '0', str(count0)+'.png'), image)
    #                 count0 += 1
    #                 num_back += 1
    #                 flags[i][r, c] = 1
    #
    #     for count in range(train_num):
    #         rd = random()
    #         for i in range(len(imgs)):
    #             img_path = img_paths[i]
    #             img = imgs[i]
    #             mask = masks[i]
    #             rs, cs = mask.shape
    #             if rs-radius < radius or cs-radius < radius:
    #                 print(img_path + ' size should be at least 45x45')
    #                 return
    #             r = choice(range(radius, rs-radius))
    #             c = choice(range(radius, cs-radius))
    #             if rd < front_areas[i+1] and rd > front_areas[i]:
    #                 while not (mask[r, c] == 255 and flags[i][r, c] < 1):
    #                 #while not mask[r, c] == 255:
    #                     r = choice(range(radius, rs-radius))
    #                     c = choice(range(radius, cs-radius))
    #                 image = img[r-radius:r+radius+1, c-radius:c+radius+1, :]
    #                 image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
    #                 cv2.imwrite(join(data_path, model_id, str(count)+'.png'), image)
    #                 num_front += 1
    #                 flags[i][r, c] = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument("--annotate_radius", type=int)
    args = parser.parse_args()

    generate_data(path = args.path, annotate_radius = args.annotate_radius)
