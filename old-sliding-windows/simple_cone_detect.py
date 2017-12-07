import numpy as np
import cv2
import os
from os.path import join
from keras.models import model_from_json
import pandas as pd
import time
import random
from scipy.ndimage.filters import gaussian_filter

np.set_printoptions(threshold=np.nan)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

def cone_detection(csv_folder_path, model_path, zoom_rate, bias_rate):
    dirname = os.path.split(csv_folder_path)[0]

    json_file = open(model_path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path+'.h5')
    print("Loaded model from disk")

    input_shape = model.input.shape
    patch_size = int(input_shape[1])
    channel = int(input_shape[3])
    class_num = int(model.output.shape[1])-1
    classes = range(1, class_num+1)

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

        cone_id = 0
        for label, xmin, ymin, xmax, ymax, cx, cy in zip(labels, xmins, ymins, xmaxs, ymaxs, cxs, cys):
            cx = int(cx + random.choice(range(-10, 10)) * bias_rate)
            cy = int(cy + random.choice(range(-10, 10)) * bias_rate)
            length = int(max(xmax - xmin, ymax - ymin)/2 * zoom_rate)
            xl = max(cx-length, 0)
            xr = min(cx+length, row)
            yl = max(cy-length, 0)
            yr = min(cy+length, col)
            cv2.rectangle(img,(yl,xl),(yr,xr),(0,255,0),1)
            roi = img[xl:xr, yl:yr, :]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi = cv2.resize(roi, (patch_size, patch_size))
            input_image = np.zeros([1, patch_size, patch_size, channel])
            input_image[0] = roi
            prob = model.predict(input_image)
            prob = prob[:,classes]
            prob = np.squeeze(prob)
            max_prob = prob.max()
            if max_prob > 0.1:
                cone_id += 1
                predict_label = np.argmax(prob)

                if predict_label == 0:
                    predict_class = 'yellow'
                    color = (0, 255, 255)
                elif predict_label == 1:
                    predict_class = 'blue'
                    color =(255, 0, 0)
                else:
                    predict_class = 'orange'
                    color = (0, 165, 255)
                cv2.circle(img, (cy, cx), 3, color, -1)
                cv2.putText(img, str(cone_id), (cy, cx), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

        save_path = join('result/2_5_right', filename+'.png')
        cv2.imwrite(save_path, img)
        '''
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        '''
    whole_time = time.clock() - start
    average_time = whole_time/len(csv_paths)
    print(average_time)

if __name__ == '__main__':
    cone_detection(csv_folder_path = 'video2/bbox', model_path = 'models/model', zoom_rate = 1.2, bias_rate = 0)
