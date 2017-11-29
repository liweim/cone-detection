import numpy as np
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping
import scipy.ndimage as sn
from scipy.misc import imresize
import time
from Utils import read_img, write_txt
import argparse
from csv2shp import csv2shp
import cv2

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

def callback(x):
    pass

def trackbar(img, prob_map, threshold):
    title = 'Drag the slider to get the best result'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('plant_distance', title, 2, 20, callback)
    rows, cols = prob_map.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    last = 0
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        plant_distance = cv2.getTrackbarPos('plant_distance', title)
        plant_distance *= 5

        if plant_distance == last:
            continue

        if plant_distance > 0:
            mask = cv2.inRange(prob_map, threshold, 1)
            mask_gray = np.copy(img)
            mask_color = np.copy(img)
            for r in range(rows):
                for c in range(cols):
                    if mask[r, c] == 0:
                        mask_gray[r, c, :] = gray[r, c]

            idx = strict_local_maximum(prob_map, plant_distance, threshold)

            for i in range(len(idx[0])):
                x = int(idx[0][i])
                y = int(idx[1][i])
                plant_size = 2
                mask_gray[x-plant_size:x+plant_size, y-plant_size:y+plant_size] = [0, 0, 255]
                mask_color[x-plant_size:x+plant_size, y-plant_size:y+plant_size] = [0, 0, 255]
            result = cv2.hconcat((mask_gray, mask_color))
            cv2.imshow(title, result)
            last = plant_distance

    print('plant_distance: {}'.format(plant_distance))
    return plant_distance

def load_image(img_path, patch_radius, resize_rate = 1):
    img = cv2.imread(img_path)
    if not resize_rate == 1:
        img = imresize(img, resize_rate)
    if len(img.shape) == 3:
        pad_width = ((patch_radius, patch_radius), (patch_radius, patch_radius), (0, 0))
    else:
        pad_width = ((patch_radius, patch_radius), (patch_radius, patch_radius))
    img_pad = np.lib.pad(img/255, pad_width, 'symmetric')

    return img, img_pad

def strict_local_maximum(prob_map, plant_distance, threshold):
    prob_gau = np.zeros(prob_map.shape)
    sigma = plant_distance/10 + 1
    sn.gaussian_filter(prob_map, sigma, output=prob_gau, mode='mirror')

    prob_fil = np.zeros(prob_map.shape)
    sn.rank_filter(prob_gau, -2, output=prob_fil, footprint=np.ones([plant_distance, plant_distance]))

    temp = np.logical_and(prob_gau > prob_fil, prob_map > threshold) * 1.
    idx = np.where(temp > 0)
    return idx

def plant_detect(img_path, model_id, plant_distance, show_img, threshold, resize_rate, test_mode):
    start = time.clock()

    dirname = os.path.split(img_path)[0]
    basename = os.path.split(img_path)[1]
    filename = os.path.splitext(basename)[0]
    ext = os.path.splitext(basename)[1]
    num = os.path.split(dirname)[1]
    model_path = join('models', 'plant_' + model_id)

    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path +'.h5')

    patch_size = 25
    patch_radius = int((patch_size-1)/2)
    channel = 3
    detect_size = 500
    input_detect_size = detect_size + 2 * patch_radius

    img, img_pad = load_image(img_path, patch_radius, resize_rate)
    if test_mode == 0:
        print('Loaded image: {}'.format(img_path))
        print('Image shape: {}'.format(img.shape))
    rows, cols = img.shape[:2]
    rows_pad, cols_pad = img_pad.shape[:2]
    prob_map = np.zeros([rows, cols])
    if np.min(img_pad)<1:
        for r in range(0, rows_pad, detect_size):
            for c in range(0, cols_pad, detect_size):
                input_image = img_pad[r:r+input_detect_size, c:c+input_detect_size, :]
                if np.min(input_image)<1:
                    input_image = np.expand_dims(input_image, axis = 0)
                    prob = model.predict(input_image)
                    prob = np.squeeze(prob)
                    prob_map[r:r+detect_size, c:c+detect_size] = prob

    if test_mode == 1:
        plant_distance = trackbar(img, prob_map, threshold)
    idx = strict_local_maximum(prob_map, plant_distance, threshold)
    num_plant = len(idx[0])
    if test_mode == 0:
        print("{} plant_{} detected.".format(num_plant, model_id))

    if show_img == 1:
        plant_size = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(prob_map, threshold, 1)
        mask_gray = np.copy(img)
        mask_color = np.copy(img)
        for r in range(rows):
            for c in range(cols):
                if mask[r, c] == 0:
                    mask_gray[r, c, :] = gray[r, c]

    idx_t = []
    for i in range(len(idx[0])):
        x = idx[0][i]
        y = idx[1][i]
        if show_img == 1:
            mask_gray[x-plant_size:x+plant_size, y-plant_size:y+plant_size] = [0, 0, 255]
            mask_color[x-plant_size:x+plant_size, y-plant_size:y+plant_size] = [0, 0, 255]
        x = int(x/resize_rate)
        y = int(y/resize_rate)
        idx_t.append([x,y])

    idx_t = np.array(idx_t)

    save_path = os.path.join('tmp', num, filename) + '.txt'
    write_txt(save_path, idx_t, way='w')

    if (ext == '.tif' or ext == '.tiff') and show_img == 0:
        geo_path = os.path.join('tmp', num, filename) + '.geojson'
        csv2shp(save_path, geo_path, img_path)

    if test_mode == 0:
        m, s = divmod(time.clock() - start, 60)
        h, m = divmod(m, 60)
        print("Run time: {}:{}:{}".format(int(h), int(m), int(s)))

    if show_img and num_plant > 0:
        save_path = os.path.join('tmp', num, filename) + '.png'
        result = cv2.hconcat((mask_gray, mask_color))
        cv2.imwrite(save_path, result)
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', result)
        cv2.waitKey(0)

    return num_plant

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="Image to analyze.")
    parser.add_argument("--model_id", type=str, help="Which plant to detect.")
    parser.add_argument("--plant_distance", type=int, help="Distance between plants.")
    parser.add_argument("--show_img", type=int, help="Show image or not.")
    parser.add_argument("--threshold", type=float, help="Threshold to classify.")
    parser.add_argument("--resize_rate", type=float, help="Resize an image.")
    parser.add_argument("--test_mode", type=int, help="Test mode.")
    args = parser.parse_args()

    plant_detect(img_path = args.img_path, model_id = args.model_id,
        plant_distance = args.plant_distance, show_img = args.show_img,
        threshold = args.threshold, resize_rate = args.resize_rate, test_mode = args.test_mode)
