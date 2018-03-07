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
from Utils import read_img, write_txt, std_color
import argparse
from csv2shp import csv2shp
import cv2
from random import choice
import operator
import pandas as pd

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))

def load_image(img_path, patch_radius, resize_rate, plant_color):
    img_source = read_img(img_path)
    img = np.copy(img_source)

    if not resize_rate == 1:
        height, width = img_source.shape[:2]
        img = cv2.resize(img_source, (int(width*resize_rate), int(height*resize_rate)))
    if len(img.shape) == 3:
        pad_width = ((patch_radius, patch_radius), (patch_radius, patch_radius), (0, 0))
    else:
        pad_width = ((patch_radius, patch_radius), (patch_radius, patch_radius))
    img_pad = np.lib.pad(img, pad_width, 'symmetric')
    img_pad = std_color(img_pad, plant_color)
    return img, img_pad/255, img_source

def plant_detect(img_path, model_name, plant_distance, row_distance, threshold, save_img_path = 'default', info = 1, display = 1):
    start = time.clock()

    dirname = os.path.split(img_path)[0]
    basename = os.path.split(img_path)[1]
    filename = os.path.splitext(basename)[0]
    ext = os.path.splitext(basename)[1]
    if ext == '.tif' or ext == '.tiff':
        plant_id = os.path.split(dirname)[1]
        show_img = 0
    else:
        plant_id = os.path.split(os.path.split(dirname)[0])[1]
        show_img = 1
    if display == 0:
        show_img = 0
    model_path = join('models', model_name)

    path = join('tmp', plant_id, 'result')
    if not os.path.exists(path):
        os.mkdir(path)

    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path +'.h5')

    df = pd.read_csv('plant_info.csv')
    df_plant = df[df['plant_id'] == int(plant_id)]
    resize_rate = float(df_plant['resize_rate'])
    resize_rate = min(resize_rate, 1.5)
    plant_color = int(df_plant['plant_color'])

    patch_size = 45
    patch_radius = int((patch_size-1)/2)
    channel = 3
    output_shape = int(model.output.shape[3])
    labels = [1]
    num_label = len(labels)
    detect_size = 1000
    input_detect_size = detect_size + 2 * patch_radius

    img, img_pad, img_source = load_image(img_path, patch_radius, resize_rate, plant_color)
    if info:
        print(img_path, img_pad.shape)

    rows, cols = img.shape[:2]
    rows_pad, cols_pad = img_pad.shape[:2]
    prob_map = np.zeros([rows, cols, num_label])
    max_map = np.zeros([rows, cols])
    max_index = np.zeros([rows, cols]).astype(np.uint8)
    if max(img.shape) < detect_size:
        input_image = np.expand_dims(img_pad, axis = 0)
        prob = model.predict(input_image)
        prob = np.squeeze(prob)
        prob_map = prob[:, :, labels]
    else:
        if np.min(img_pad)<1:
            for r in range(0, rows, detect_size):
                for c in range(0, cols, detect_size):
                    rr = min(r+input_detect_size, rows_pad)
                    cr = min(c+input_detect_size, cols_pad)
                    input_image = np.zeros((input_detect_size, input_detect_size, 3))
                    input_image[:rr-r, :cr-c] = img_pad[r:rr, c:cr]
                    if np.min(input_image)<1:
                        input_image = np.expand_dims(input_image, axis = 0)
                        prob = model.predict(input_image)
                        prob = np.squeeze(prob)
                        rr = min(r+detect_size, rows)
                        cr = min(c+detect_size, cols)
                        prob_map[r:rr, c:cr, :] = prob[:rr-r, :cr-c, labels]

    max_index_temp = prob_map>0.5
    max_map_temp = prob_map * max_index_temp
    max_map = np.sum(max_map_temp, axis = 2)
    max_index = np.dot(max_index_temp, range(1, len(labels)+1))

    sigma = plant_distance/5
    prob_gau = sn.gaussian_filter(max_map, sigma, mode='mirror')
    prob_fil = sn.rank_filter(prob_gau, -2, footprint=np.ones([plant_distance, plant_distance]))
    mask = np.logical_and(prob_gau > prob_fil, max_map >= threshold)
    max_index *= mask

    good_hits = []
    bad_hits = []
    good_hits_temp = []
    row_distance /= 2

    for label_id in range(1, num_label+1):
        good_hit = []
        bad_hit = []
        good_hit_temp = []
        same_index = (max_index == label_id)

        idx = np.where(same_index > 0)
        if row_distance > 0:
            for i in range(len(idx[0])):
                x = idx[0][i]
                y = idx[1][i]
                rl = max(int(x-row_distance), 0)
                rr = min(int(x+row_distance), rows)
                cl = max(int(y-row_distance), 0)
                cr = min(int(y+row_distance), cols)
                x_resized = int(x/resize_rate)
                y_resized = int(y/resize_rate)
                if np.sum(same_index[rl:rr,cl:cr]) > 1:
                    good_hit_temp.append([x,y])
                    good_hit.append([x_resized,y_resized])
                else:
                    bad_hit.append([x_resized,y_resized])
        else:
            for i in range(len(idx[0])):
                x = idx[0][i]
                y = idx[1][i]
                x_resized = int(x/resize_rate)
                y_resized = int(y/resize_rate)
                good_hit_temp.append([x,y])
                good_hit.append([x_resized,y_resized])
        good_hits += good_hit
        good_hits_temp += good_hit_temp
        bad_hits += bad_hit
        num_good_hit = len(good_hit)
        num_bad_hit = len(bad_hit)
        if info:
            print('Plant {} has {}, {} of them are bad hits'.format(label_id, np.sum(same_index), np.sum(num_bad_hit)))

    good_hits = np.array(good_hits)
    bad_hits = np.array(bad_hits)

    if len(good_hits) > 0:
        save_path = os.path.join('tmp', plant_id, 'result', filename + '.txt')
        write_txt(save_path, good_hits, way='w')
    if len(bad_hits) > 0:
        bad_save_path = os.path.join('tmp', plant_id, 'result', filename + '_bad.txt')
        write_txt(bad_save_path, bad_hits, way='w')

    if (ext == '.tif' or ext == '.tiff') and show_img == 0:
        geo_path = os.path.join('tmp', plant_id, filename) + '.geojson'
        csv2shp(save_path, geo_path, img_path)
        print('Saved geojson file')

    num_plant = len(good_hits)
    print("{} detected {} plants.".format(img_path, num_plant))
    if info:
        m, s = divmod(time.clock() - start, 60)
        h, m = divmod(m, 60)
        print("Run time: {}:{}:{}".format(int(h), int(m), int(s)))

    if show_img:
        mark_size = 2
        plant_region = cv2.inRange(max_map, threshold, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_gray = np.copy(img)
        mask_color = np.copy(img_source)
        for r in range(rows):
            for c in range(cols):
                if plant_region[r, c] == 0:
                    mask_gray[r, c, :] = gray[r, c]
        if not resize_rate == 1:
            height, width = img_source.shape[:2]
            mask_gray = cv2.resize(mask_gray, (width, height))
        color_range = range(255)
        color = np.zeros([num_label, 3])
        for i in range(num_label):
            color[i] = [choice(color_range), choice(color_range), choice(color_range)]

        for x, y in good_hits_temp:
            index = max_index[int(x),int(y)] - 1
            x = int(x/resize_rate)
            y = int(y/resize_rate)
            mask_gray[x-mark_size:x+mark_size, y-mark_size:y+mark_size] = [0,0,255] #color[index]
            mask_color[x-mark_size:x+mark_size, y-mark_size:y+mark_size] = [0,0,255] #color[index]
        for x, y in bad_hits:
            mask_gray[x-mark_size:x+mark_size, y-mark_size:y+mark_size] = [0,0,0]
            mask_color[x-mark_size:x+mark_size, y-mark_size:y+mark_size] = [0,0,0]

        if num_plant > 0:
            gap = np.zeros((mask_color.shape[0], 10, 3)).astype(np.uint8)
            result = cv2.hconcat((mask_color, gap, mask_gray))

            #plant_size, result = near_neighbor(mask_color, plant_region, local_maximas)
            save_path = os.path.join('tmp', plant_id, 'result', filename + '.png')
            cv2.imwrite(save_path, result)
            if not save_img_path == 'default':
                cv2.imwrite(save_img_path, result)
            if info:
                cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('result', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return num_plant

def near_neighbor(img, plant_region, local_maximas):
    row, col = plant_region.shape
    checked_map = np.zeros(plant_region.shape)
    region_set = []
    finish_flags = np.zeros(len(local_maximas))

    iterate = 0
    while min(finish_flags) == 0:
        for index, (local_maxima, finish_flag) in enumerate(zip(local_maximas, finish_flags)):
            if finish_flag == 0:
                flag = 0
                if iterate == 0:
                    cx = local_maxima[0]
                    cy = local_maxima[1]
                    region_set.append([])
                    region_set[index].append([cx,cy])
                    checked_map[cx,cy] = 1
                region_set_tmp = list(region_set[index])
                for xy in region_set_tmp:
                    x = xy[0]
                    y = xy[1]
                    if x-1>=0 and plant_region[x-1,y]>0 and checked_map[x-1,y] == 0:
                        region_set[index].append([x-1,y])
                        checked_map[x-1,y] = 1
                        flag = 1
                    if x+1<row and plant_region[x+1,y]>0 and checked_map[x+1,y] == 0:
                        region_set[index].append([x+1,y])
                        checked_map[x+1,y] = 1
                        flag = 1
                    if y-1>=0 and plant_region[x,y-1]>0 and checked_map[x,y-1] == 0:
                        region_set[index].append([x,y-1])
                        checked_map[x,y-1] = 1
                        flag = 1
                    if y+1<col and plant_region[x,y+1]>0 and checked_map[x,y+1] == 0:
                        region_set[index].append([x,y+1])
                        checked_map[x,y+1] = 1
                        flag = 1
                if flag == 0:
                    finish_flags[index] = 1
        iterate += 1

    region_mark = np.zeros(img.shape).astype(np.uint8)
    plant_size = np.zeros(len(local_maximas))
    color_range = range(255)
    mark_size = 1
    for index, (local_maxima, region_set) in enumerate(zip(local_maximas, region_set)):
        color = [choice(color_range), choice(color_range), choice(color_range)]
        plant_size[index] = len(region_set)
        if plant_size[index] > 0:
            cx = local_maxima[0]
            cy = local_maxima[1]
            for xy in region_set:
                x = xy[0]
                y = xy[1]
                region_mark[x, y] = color
            region_mark[cx-mark_size:cx+mark_size, cy-mark_size:cy+mark_size] = [0, 255, 255]
    gap = np.zeros((row, 10, 3)).astype(np.uint8)
    result = cv2.hconcat((img, gap, region_mark))
    #print(plant_size)
    return plant_size, result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="Image to analyze.")
    parser.add_argument("--model_name", type=str, help="Which model to use.")
    parser.add_argument("--plant_distance", type=int, help="Distance between plants.")
    parser.add_argument("--row_distance", type=int, help="Distance between rows.")
    parser.add_argument("--threshold", type=float, help="Threshold to be classified as plants.")
    args = parser.parse_args()

    plant_detect(img_path = args.img_path, model_name = args.model_name,
        plant_distance = args.plant_distance, row_distance = args.row_distance,
        threshold = args.threshold)
