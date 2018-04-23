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

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def test_image(csv_path):
    PATH_TO_CKPT = 'graph/frozen_inference_graph.pb'
    PATH_TO_LABELS = 'object-detection.pbtxt'
    NUM_CLASSES = 1

    csv=pd.read_csv(csv_path)

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

    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    test_path = 'test'
    save_path = 'result'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    test_image_path=os.listdir(test_path)
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(1,2) ]

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
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
        for image_path in test_image_path:
            img_path=join(test_path,image_path)
            save_image_path = join(save_path, image_path)
            image = Image.open(img_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            start = time.clock()
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
            m, s = divmod(time.clock() - start, 60)
            h, m = divmod(m, 60)
            print("Run time: {}:{}:{}".format(int(h), int(m), int(s)))
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=4)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)

            line = csv[csv.filename==image_path]
            labels = line['class'].tolist()
            xmins = line['xmin'].tolist()
            ymins = line['ymin'].tolist()
            xmaxs = line['xmax'].tolist()
            ymaxs = line['ymax'].tolist()

            current_axis = plt.gca()
            for label, xmin, ymin, xmax, ymax in zip(labels, xmins, ymins, xmaxs, ymaxs):
                current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='red', fill=False, linewidth = 2))
                #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'red', 'alpha':1.0})

            plt.savefig(save_image_path)
            plt.show()

if __name__=='__main__':
    test_image('tmp/912/test.csv')
    print('done!')
