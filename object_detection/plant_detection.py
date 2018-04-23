import numpy as np
import os
from os.path import join
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import pandas as pd
import time
from skimage.io import imread, imsave
import scipy.ndimage as sn

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def read_img(img_path,type=np.uint8):
    img = imread(img_path).astype(type)
    if  len(img.shape)==3 and img.shape[2] == 4:
        img = np.ascontiguousarray(img[:,:,0:3])
    return img

def strict_local_maximum(prob_map, plant_distance, threshold):
    prob_map_tmp = prob_map[:,:,0]
    prob_map_tmp = np.squeeze(prob_map_tmp)
    #prob_gau = np.zeros(prob_map_tmp.shape)
    #sn.gaussian_filter(prob_map_tmp, 2, output=prob_gau, mode='mirror')

    prob_fil = np.zeros(prob_map_tmp.shape)
    sn.rank_filter(prob_map_tmp, -2, output=prob_fil, footprint=np.ones([plant_distance, plant_distance]))

    temp = np.logical_and(prob_map_tmp > prob_fil, prob_map_tmp > threshold) * 1.
    index = temp>0
    prob_map_max = prob_map[index]
    return prob_map_max

def write_txt(txt_path,point,way='w'):
    with open(txt_path,way) as f:
        for i in range(len(point)):
            f.write(str(point[i,0])+' '+str(point[i,1])+' '+str(point[i,2])+' '+str(point[i,3]))
            f.write('\n')
    return

def plant_detect(img_path, overlay=30, input_size=300, show_img = 1):
    dirname = os.path.split(img_path)[0]
    basename = os.path.split(img_path)[1]
    filename = os.path.splitext(basename)[0]
    ext = os.path.splitext(basename)[1]
    num = os.path.split(dirname)[1]
    PATH_TO_CKPT = join('tmp', 'graph', 'frozen_inference_graph.pb')
    PATH_TO_LABELS = 'object-detection.pbtxt'
    save_txt_path = join('tmp', num, filename+'.txt')
    save_img_path = join('tmp', num, filename+'.png')
    NUM_CLASSES = 1

    img = read_img(img_path)
    rows, cols, n =img.shape
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    prob_map = np.zeros([rows, cols, 7])

    if show_img:
        plt.imshow(img)
    start = time.clock()
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for row in range(0,rows,input_size-overlay):
            for col in range(0,cols,input_size-overlay):
                image = np.zeros((input_size,input_size,3))
                image_tmp = img[row:row+input_size, col:col+input_size, :]
                image[0:image_tmp.shape[0],0:image_tmp.shape[1],:]=image_tmp
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_expanded = np.expand_dims(image, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
                # Visualization of the results of a detection.

                boxes = np.squeeze(boxes)*input_size
                scores = np.squeeze(scores)

                xmins = boxes[:,0]+row
                ymins = boxes[:,1]+col
                xmaxs = boxes[:,2]+row
                ymaxs = boxes[:,3]+col
                ys = (ymins+ymaxs)/2
                xs = (xmins+xmaxs)/2
                for score, x, y, xmin, ymin, xmax, ymax in zip(scores, xs, ys, xmins, ymins, xmaxs, ymaxs):
                    x = int(x)
                    y = int(y)
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)
                    if x<rows and y<cols:
                        prob_map[x, y, :] = [score, x, y, xmin, ymin, xmax, ymax]
                print(len(scores))

    prob_map_max = strict_local_maximum(prob_map, plant_distance=overlay, threshold=0.3)#0.3
    write_txt(save_txt_path, prob_map_max[:,3:7])

    m, s = divmod(time.clock() - start, 60)
    h, m = divmod(m, 60)
    print("Run time: {}:{}:{}".format(int(h), int(m), int(s)))

    if show_img:
        xs = prob_map_max[:,1]
        ys = prob_map_max[:,2]
        xmins = prob_map_max[:,3]
        ymins = prob_map_max[:,4]
        xmaxs = prob_map_max[:,5]
        ymaxs = prob_map_max[:,6]

        current_axis = plt.gca()
        for x, y, xmin, ymin, xmax, ymax in zip(xs, ys, xmins, ymins, xmaxs, ymaxs):
            current_axis.add_patch(plt.Rectangle((ymin, xmin), ymax-ymin, xmax-xmin, color='red', fill=False, linewidth = 2))
            area = (xmax-xmin)*(ymax-ymin)
            plt.plot(y,x,'rx',markersize=12)
            #current_axis.text(ymin, xmin, 'area:'+str(area), size='x-large', color='white', bbox={'facecolor':'red', 'alpha':1.0})
        plt.show()


if __name__ == '__main__':
    plant_detect(img_path = 'tmp/997/997.png', overlay = 30, input_size = 300, show_img = 1)
    print('done!')
