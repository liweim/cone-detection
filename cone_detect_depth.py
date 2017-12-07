import sklearn.model_selection
import numpy as np
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import model_from_json
import scipy.ndimage as sn
import time
import argparse
import cv2
import pandas as pd
import random
from reconstruction import reconstruction

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

color = [(0, 255, 255), (255, 0, 0), (0, 165, 255)]
class_label = ['yellow', 'blue', 'orange']

def callback(x):
    pass
'''
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
'''
def trackbar(img, prob_map, num_class, threshold):
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('cone_distance','img',2,20,callback)

    last = 0
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        cone_distance = cv2.getTrackbarPos('cone_distance','img')
        cone_distance *= 5

        if cone_distance == last:
            continue

        if cone_distance > 0:
            temp_img = np.copy(img)
            masks = np.zeros(img.shape[0:2])
            for i_class in range(num_class):
                mask = prob_map[:, :, i_class]
                index_map = mask > threshold
                mask = mask * index_map
                masks = np.logical_or(masks, index_map)
                idxes = strict_local_maximum(mask, cone_distance, threshold)

                for idx in range(len(idxes[0])):
                    x = int(idxes[0][idx])
                    y = int(idxes[1][idx])
                    temp_img[x-2:x+2, y-2:y+2] = color[i_class]
            for i in range(3):
                temp_img[:, :, i] = temp_img[:, :, i] * masks
            cv2.imshow('img',temp_img)
    cv2.destroyAllWindows()

    print(cone_distance)
    return cone_distance

def img_padding(img_ori, patch_radius):
    img = img_ori / 255
    pad_width = ((patch_radius, patch_radius), (patch_radius, patch_radius), (0, 0))
    img_pad = np.lib.pad(img, pad_width, 'symmetric')
    return img_pad

def strict_local_maximum(prob_map, plant_distance, threshold):
    prob_gau = np.zeros(prob_map.shape)
    sigma = plant_distance/10 + 1
    sn.gaussian_filter(prob_map, 3, output=prob_gau, mode='mirror')

    prob_fil = np.zeros(prob_map.shape)
    sn.rank_filter(prob_gau, -2, output=prob_fil, footprint=np.ones([plant_distance, plant_distance]))

    temp = np.logical_and(prob_gau > prob_fil, prob_map > threshold) * 1.
    idx = np.where(temp > 0)
    return idx

def cone_detect_roi(csv_folder_path, model_path, bias_rate, threshold):
    dirname = os.path.split(csv_folder_path)[0]

    json_file = open(model_path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path+'.h5')
    print("Loaded model from disk")

    patch_size = 25
    patch_radius = int((patch_size-1)/2)
    channel = 3
    num_class = 2
    classes = range(1, num_class+1)
    detect_size = 51
    detect_radius = int((detect_size-1)/2)
    input_detect_size = detect_size + 2 * patch_radius

    csv_paths = os.listdir(csv_folder_path)
    start = time.clock()
    for csv_path in csv_paths:
        filename = os.path.splitext(csv_path)[0]
        img_path = join(dirname, 'right', filename+'.png')
        img = cv2.imread(img_path)
        row, col, n = img.shape

        csv_path = join(csv_folder_path, csv_path)
        bbox_df = pd.read_csv(csv_path)
        boxes = np.array(bbox_df[:])
        labels = boxes[:,0]
        ymins = boxes[:,1]
        xmins = boxes[:,2]
        ymaxs = boxes[:,3]
        xmaxs = boxes[:,4]
        cxs = (xmins+xmaxs)/2
        cys = (ymins+ymaxs)/2

        cones = []
        cone_id = 0
        temp_img = np.copy(img)
        for label, xmin, ymin, xmax, ymax, cx, cy in zip(labels, xmins, ymins, xmaxs, ymaxs, cxs, cys):
            cx = int(cx + random.choice(range(-10, 10)) * bias_rate)
            cy = int(cy + random.choice(range(-10, 10)) * bias_rate)
            xl = max(cx-detect_radius, 0)
            xr = min(cx+detect_radius, row)
            yl = max(cy-detect_radius, 0)
            yr = min(cy+detect_radius, col)
            cv2.rectangle(temp_img,(yl,xl),(yr,xr),(0,255,0),1)
            roi = img[xl:xr, yl:yr, :]
            rows, cols, n = roi.shape

            img_pad = img_padding(roi, patch_radius)
            rows_pad, cols_pad, d = img_pad.shape
            prob_map = np.zeros([rows, cols, num_class])

            input_image = np.expand_dims(img_pad, axis = 0)
            prob = model.predict(input_image)
            prob = np.squeeze(prob)
            prob_map = prob[:, :, classes]

            '''
            for i_class in range(num_class):
                index_map = prob_map[:, :, i_class] > threshold
                for r in range(rows):
                    for c in range(cols):
                        if index_map[r, c]:
                            x = r + cx - detect_radius
                            y = c + cy - detect_radius
                            temp_img[x, y, :] = [0, 0, 255]
            '''
            '''
            for i_class in range(num_class):
                mask = prob_map[:, :, i_class]
                index_map = mask > threshold
                mask = mask * index_map
                idxes = strict_local_maximum(mask, detect_size, threshold)

                for idx in range(len(idxes[0])):
                    x = int(idxes[0][idx]) + cx - detect_radius
                    y = int(idxes[1][idx]) + cy - detect_radius
                    cones.append([x, y, i_class])

            for i_cone in range(len(cones)):
                x = cones[i_cone][0]
                y = cones[i_cone][1]
                predict_class = cones[i_cone][2]
                color_show = color[predict_class]
                cv2.circle(img, (y, x), 3, color_show, -1)
                cv2.putText(img, str(i_cone), (y, x), cv2.FONT_HERSHEY_SIMPLEX,0.5, color_show, 2)
            '''

            prob_gau = np.zeros(prob_map.shape)
            for i in range(num_class):
                #sn.gaussian_filter(prob_map[:, :, i], 2, output=prob_gau[:, :, i], mode='mirror')
                prob_gau[:, :, i] = cv2.GaussianBlur(prob_map[:, :, i], (5, 5), 0)
            #print(prob_map)
            max_prob = prob_gau.max()
            if max_prob > threshold:
                cone_id += 1
                x, y, predict_class = np.where(prob_gau == max_prob)
                if len(x) > 1:
                    x = int(np.median(x))
                    y = int(np.median(y))
                #print(cone_id, x, y, predict_class, max_prob)
                x = x + cx - detect_radius
                y = y + cy - detect_radius

                color_show = color[predict_class[0]]
                cv2.circle(temp_img, (y, x), 3, color_show, -1)
                cv2.putText(temp_img, str(cone_id), (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_show, 2)

        save_path = join('result/2_5_right', filename+'.png')
        cv2.imwrite(save_path, temp_img)
        '''
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', temp_img)
        cv2.waitKey(0)
        '''
    whole_time = time.clock() - start
    average_time = whole_time/len(csv_paths)
    print(average_time)

def detection(img, model, cone_distance, threshold, patch_radius, num_class, classes, detect_size, input_detect_size):
    img_pad = img_padding(img, patch_radius)
    cv2.imshow('img', img)
    rows, cols, d = img.shape
    rows_pad, cols_pad, d = img_pad.shape
    prob_map = np.zeros([rows, cols, num_class])

    if np.min(img_pad)<1:
        for r in range(0, rows_pad, detect_size):
            for c in range(0, cols_pad, detect_size):
                input_image = img_pad[r:r+input_detect_size, c:c+input_detect_size, :]
                if np.min(input_image)<1:
                    input_image = np.expand_dims(input_image, axis = 0)
                    prob = model.predict(input_image)
                    prob = np.squeeze(prob)
                    prob_map[r:r+detect_size, c:c+detect_size, :] = prob[:, :, classes]

    #cone_distance = trackbar(img, prob_map, num_class, threshold)

    temp_img = np.zeros(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(3):
        temp_img[:, :, i] = gray

    cones = []
    for i_class in range(num_class):
        mask = prob_map[:, :, i_class]
        index_map = mask > threshold
        mask = mask * index_map
        idxes = strict_local_maximum(mask, cone_distance, threshold)

        for r in range(rows):
            for c in range(cols):
                if index_map[r, c]:
                    temp_img[r, c, :] = img[r, c, :]

        for idx in range(len(idxes[0])):
            x = int(idxes[0][idx])
            y = int(idxes[1][idx])
            #depth = factor / disp[x, y]
            #cones.append([x, y, class_label[i_class], depth])
            #print(x, y, class_label[i_class], depth)
            cones.append([x, y, class_label[i_class]])
            print(x, y, class_label[i_class])
            temp_img[x-2:x+2, y-2:y+2] = color[i_class]
    return temp_img, cones

def cone_detect_depth(img_path, model_path, cone_distance, threshold):
    imgL, imgR, disp, factor = reconstruction(img_path)
    basename = os.path.split(img_path)[1]

    json_file = open(model_path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path+'.h5')
    print("Loaded model from disk")

    patch_size = 25
    patch_radius = int((patch_size-1)/2)
    channel = 3
    num_class = 1
    classes = range(1, num_class+1)
    detect_size = 512
    input_detect_size = detect_size + 2 * patch_radius

    left_result, left_cones = detection(imgL, model, cone_distance, threshold,
        patch_radius, num_class, classes, detect_size, input_detect_size)
    right_result, right_cones = detection(imgR, model, cone_distance, threshold,
        patch_radius, num_class, classes, detect_size, input_detect_size)

    disparity = []
    if len(left_cones) == len(right_cones):
        for left_cone, right_cone in zip(left_cones, right_cones):
            pt1 = np.array(left_cone[:2])
            pt2 = np.array(right_cone[:2])
            disparity.append(np.linalg.norm(pt1 - pt2))
    depth = factor/np.array(disparity)
    print(depth)
    '''
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', temp_img)
    cv2.waitKey(0)
    save_path = join('result', basename)
    cv2.imwrite(save_path, temp_img)
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type = str)
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--cone_distance", type = int)
    parser.add_argument("--threshold", type = float)
    args = parser.parse_args()

    cone_detect_depth(img_path = args.img_path, model_path = args.model_path, cone_distance = args.cone_distance, threshold = args.threshold)
