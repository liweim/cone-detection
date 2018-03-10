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
from scipy.misc import imresize

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# set_session(tf.Session(config=config))

patch_size = 45
patch_radius = int((patch_size-1)/2)
resize_rate = 1
color = [(0, 255, 255), (255, 0, 0), (0, 165, 255)]
class_label = ['yellow', 'blue', 'orange']

def callback(x):
    pass

def trackbar(img, prob_map, num_class):
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('threshold','img',90,100,callback)
    cv2.createTrackbar('cone_distance','img',20,100,callback)

    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        threshold = cv2.getTrackbarPos('threshold','img')
        cone_distance = cv2.getTrackbarPos('cone_distance','img')

        threshold /= 100
        temp_img = np.copy(img)
        masks = np.zeros(img.shape[0:2])
        for i_class in range(num_class):
            mask = prob_map[:, :, i_class]
            index_map = mask > threshold
            mask = mask * index_map
            masks = np.logical_or(masks, index_map)
            idxes = strict_local_maximum(mask, cone_distance)

            for idx in range(len(idxes[0])):
                x = int(idxes[0][idx])
                y = int(idxes[1][idx])
                temp_img[x-2:x+2, y-2:y+2] = color[i_class]
        for i in range(3):
            temp_img[:, :, i] = temp_img[:, :, i] * masks
        cv2.imshow('img',temp_img)
    cv2.destroyAllWindows()

    print(threshold, cone_distance)
    return threshold, cone_distance

def load_image(img_ori, patch_radius):
    img = img_ori / 255
    pad_width = ((patch_radius, patch_radius), (patch_radius, patch_radius), (0, 0))
    img_pad = np.lib.pad(img, pad_width, 'symmetric')
    return img_pad

def strict_local_maximum(prob_map, cone_distance):
    prob_gau = np.zeros(prob_map.shape)
    sn.gaussian_filter(prob_map, 2, output=prob_gau, mode='mirror')

    prob_fil = np.zeros(prob_map.shape)
    sn.rank_filter(prob_gau, -2, output=prob_fil, footprint=np.ones([cone_distance, cone_distance]))

    idx = np.where(prob_gau > prob_fil)
    return idx

def load_network(model_path):
    json_file = open(model_path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path+'.h5')
    return model

def cone_detect(img_path, model, cone_distance, threshold, display_result = 1):
    basename = os.path.split(img_path)[1]

    channel = 3
    num_class = 3
    classes = range(1, num_class+1)
    up_limit = 170
    down_limit = 290 * resize_rate

    img_source = cv2.imread(img_path)
    if resize_rate == 1:
        img = np.copy(img_source)
    else:
        img = imresize(img_source, resize_rate)

    img = img[int(up_limit*resize_rate):int(down_limit*resize_rate),:,:]
    img_pad = load_image(img, patch_radius)
    rows, cols = img.shape[:2]
    rows_pad, cols_pad = img_pad.shape[:2]
    prob_map = np.zeros([rows, cols, num_class])

    input_image = np.expand_dims(img_pad, axis = 0)
    prob = model.predict(input_image)
    prob = np.squeeze(prob)
    prob_map = prob[:, :, classes]

    #threshold, cone_distance = trackbar(img, prob_map, num_class)

    # temp_img = np.zeros(img.shape)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # for i in range(3):
    #     temp_img[:, :, i] = gray

    # cones = []

    max_index_map = prob_map > threshold
    max_map_tmp = prob_map * max_index_map
    max_map = np.sum(max_map_tmp, 2)
    idxes = strict_local_maximum(max_map, cone_distance)
    for idx in range(len(idxes[0])):
        x = int(idxes[0][idx])
        y = int(idxes[1][idx])
        if max_map[x, y] > 0:
            i_class = int(np.where(max_index_map[x, y] > 0)[0])
            x = int(x/resize_rate+up_limit)
            y = int(y/resize_rate)
            # cones.append([x, y, i_class])
            # print(x, y, i_class)
            cv2.circle(img_source, (y, x), 1, color[i_class], -1)

    # for i_class in range(num_class):
    #     mask = prob_map[:, :, i_class]
    #     index_map = mask > threshold
    #     mask = mask * index_map
    #     idxes = strict_local_maximum(mask, cone_distance)
    #
    #     # for r in range(rows):
    #     #     for c in range(cols):
    #     #         if index_map[r, c]:
    #     #             rt = int(r*2)
    #     #             ct = int(c*2)
    #     #             temp_img[rt, ct, :] = img[rt, ct, :]
    #
    #     for idx in range(len(idxes[0])):
    #         x = int(idxes[0][idx]/resize_rate+up_limit)
    #         y = int(idxes[1][idx]/resize_rate)
    #         # cones.append([x, y, i_class])
    #         # print(x, y, i_class)
    #         cv2.circle(img_source, (y, x), 1, color[i_class], -1)

    if display_result:
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('img', img_source)
        cv2.waitKey(0)
    save_path = join('tmp', 'result', basename)
    cv2.imwrite(save_path, img_source)

def cone_detect_roi(csv_folder_path, model, bias_rate, threshold):
    dirname = os.path.split(csv_folder_path)[0]

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

            img_pad = load_image(roi, patch_radius)
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
                idxes = strict_local_maximum(mask, detect_size)

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

def efficient_sliding_window(img_path, model_path, cone_distance, threshold, display_result = 1):
    model = load_network(model_path)
    cone_detect(img_path, model, cone_distance, threshold, display_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type = str)
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--cone_distance", type = int)
    parser.add_argument("--threshold", type = float)
    args = parser.parse_args()

    efficient_sliding_window(img_path = args.img_path, model_path = args.model_path, cone_distance = args.cone_distance, threshold = args.threshold)
