import sklearn.model_selection
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import model_from_json
from Utils import load_std_image
import scipy.ndimage as sn
import skimage.io
from scipy.misc import imresize
import time
from Utils import read_img, write_txt
import argparse
from data_process2 import slice_img
from csv2shp import csv2shp
import cv2

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

def callback(x):
    pass

def trackbar(img, gray):
    #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('threshold','mask',0,255,callback)
    cv2.createTrackbar('plant_distance','mask',3,100,callback)

    while(1):
        img_temp = img[:,:,::-1]
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        threshold = cv2.getTrackbarPos('threshold','mask')
        plant_distance = cv2.getTrackbarPos('plant_distance','mask')
        mask = cv2.inRange(gray, threshold, 255)
        idx = strict_local_maximum(mask/255, plant_distance, threshold)
        for i in range(len(idx[0])):
            x = int(idx[0][i])
            y = int(idx[1][i])
            plant_size = 3
            img_temp[x-plant_size:x+plant_size, y-plant_size:y+plant_size] = [255, 0, 0]
        #cv2.imshow('img',img_temp)
        cv2.imshow('mask',img_temp)

    cv2.destroyAllWindows()

    threshold /= 255
    print(threshold, plant_distance)
    return threshold, plant_distance

def load_image(img_path, patch_radius, resize_rate = 1):
    img_ori = read_img(img_path)
    if not resize_rate == 1:
        img_ori = imresize(img_ori, resize_rate)
    print("Loaded image: {}".format(img_path))
    print(img_ori.shape)
    img = img_ori / 255
    if len(img.shape) == 3:
        pad_width = ((patch_radius, patch_radius), (patch_radius, patch_radius), (0, 0))
    else:
        pad_width = ((patch_radius, patch_radius), (patch_radius, patch_radius))
    img_pad = np.lib.pad(img, pad_width, 'symmetric')

    return img_ori, img_pad

def strict_local_maximum(prob_map, plant_distance, threshold):
    prob_gau = np.zeros(prob_map.shape)
    sn.gaussian_filter(prob_map, 2, output=prob_gau, mode='mirror')

    prob_fil = np.zeros(prob_map.shape)
    sn.rank_filter(prob_gau, -2, output=prob_fil, footprint=np.ones([plant_distance, plant_distance]))

    temp = np.logical_and(prob_gau > prob_fil, prob_map > threshold) * 1.
    idx = np.where(temp > 0)
    return idx

def plant_detect(img_path, model_path, plant_distance=25, show_img=1, threshold=0.9, resize_rate = 1):
    start = time.clock()

    dirname = os.path.split(img_path)[0]
    basename = os.path.split(img_path)[1]
    filename = os.path.splitext(basename)[0]
    ext = os.path.splitext(basename)[1]
    num = os.path.split(dirname)[1]

    json_file = open(model_path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path+'.h5')
    print("Loaded model from disk")

    patch_size = 25
    patch_radius = int((patch_size-1)/2)
    channel = 3
    detect_size = 512
    input_detect_size = detect_size + 2 * patch_radius

    img, img_pad = load_image(img_path, patch_radius, resize_rate)
    rows, cols, d = img.shape
    rows_pad, cols_pad, d = img_pad.shape
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

    #threshold, plant_distance = trackbar(img, prob_map * 255)
    idx = strict_local_maximum(prob_map, plant_distance, threshold)

    num_plant = len(idx[0])
    idx_t = []
    if show_img == 1:
        img_temp = np.copy(img)
        img_temp.dtype = np.uint8
        plt.imshow(img_temp)
    for i in range(len(idx[0])):
        x = int(idx[0][i]/resize_rate)
        y = int(idx[1][i]/resize_rate)
        if show_img == 1:
            plt.scatter(y, x, marker='o', color='r', s=30)
            plant_size = 2
            img_temp[x-plant_size:x+plant_size, y-plant_size:y+plant_size] = [255, 0, 0]
        idx_t.append([x,y])
    idx_t = np.array(idx_t)
    print("{} plants detected.".format(num_plant))
    save_path = os.path.join('tmp', num, filename) + '.txt'
    write_txt(save_path, idx_t, way='w')
    if (ext == '.tif' or ext == '.tiff') and show_img == 0:
        geo_path = os.path.join('tmp', num, filename) + '.geojson'
        csv2shp(save_path, geo_path, img_path)
    print('saved txt')

    m, s = divmod(time.clock() - start, 60)
    h, m = divmod(m, 60)
    print("Run time: {}:{}:{}".format(int(h), int(m), int(s)))

    if show_img and num_plant > 0:
        save_path = os.path.join('tmp', num, filename) + '.png'
        if not resize_rate == 1:
            img_temp = imresize(img_temp, resize_rate)
        skimage.io.imsave(save_path, img_temp)
        plt.show()
    return num_plant

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default='tmp/1054/1054.tiff', type=str, help="Image to analyze.")
    parser.add_argument("--model_path", default='models/plant_23', type=str, help="Applied network.")
    parser.add_argument("--plant_distance", default=10, type=int, help="Distance between plants.")
    parser.add_argument("--show_img", default=0, type=int, help="Show image or not.")
    parser.add_argument("--threshold", default=0.7, type=float, help="Threshold to classify.")
    parser.add_argument("--resize_rate", default=0.4, type=float, help="Resize an image.")
    args = parser.parse_args()

    plant_detect(img_path=args.img_path, model_path=args.model_path, plant_distance=args.plant_distance, show_img=args.show_img, threshold=args.threshold, resize_rate=args.resize_rate)
