from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

from os.path import join
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from shutil import copyfile
from random import random

def xml_to_csv(xml_path, split_rate=0.2):
    data_path = os.path.split(xml_path)[0]
    img_path = join(data_path,'right')
    test_path = 'test'
    train_list = []
    val_list = []
    test_list = []
    for xml_file in glob.glob(xml_path + '/*.xml'):
        tmp_list=[]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            try:
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                tmp_list.append(value)
            except ValueError:
                pass
        if random()>split_rate:
            train_list+=tmp_list
        elif random()<0.1:
            filename = os.path.splitext(xml_file)[0]
            filename = os.path.split(filename)[1]+'.png'
            image_from_path = join(img_path, filename)
            image_to_path = join(test_path, filename)
            copyfile(image_from_path, image_to_path)
            test_list+=tmp_list
        else:
            val_list+=tmp_list
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    train_df = pd.DataFrame(train_list, columns=column_name)
    train_df.to_csv(join(data_path,'train.csv'), index=None)
    val_df = pd.DataFrame(val_list, columns=column_name)
    val_df.to_csv(join(data_path,'val.csv'), index=None)
    test_df = pd.DataFrame(test_list, columns=column_name)
    test_df.to_csv(join(data_path,'test.csv'), index=None)
    print('Successfully converted xml to csv.')
    return

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'yellow':
        label =  1
    elif row_label == 'blue':
        label =  2
    elif row_label == 'pedestrian':
        label =  3
    elif row_label == 'Weiming':
        label =  4
    elif row_label == 'Dan':
        label =  5
    elif row_label == 'Meet':
        label =  6
    elif row_label == 'Zijian':
        label =  7
    else:
        label = None
    return label

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def xml2tfRecord(xml_path, split_rate):
    data_path = os.path.split(xml_path)[0]
    xml_to_csv(xml_path, split_rate)
    img_path = join(data_path, 'right')
    stuffs=['train', 'val', 'test']

    for stuff in stuffs:
        csv_path = join(data_path, stuff+'.csv')
        record_path = join(data_path, stuff+'.record')

        writer = tf.python_io.TFRecordWriter(record_path)
        examples = pd.read_csv(csv_path)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, img_path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        output_path = join(os.getcwd(), record_path)
        print('Successfully created the TFRecords: {}'.format(output_path))
if __name__ == '__main__':
    xml2tfRecord(xml_path='video2/annotations_right', split_rate=0.3)
