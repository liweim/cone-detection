import numpy as np
import pandas as pd
from os.path import join
import os
import io
from PIL import Image
from random import random
import tensorflow as tf
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from shutil import copyfile, rmtree

def merge_csv(csv_path, split_rate):
    dirname = os.path.split(csv_path)[0]
    basenames = os.listdir(csv_path)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    train_df = pd.DataFrame(columns=column_name)
    val_df = pd.DataFrame(columns=column_name)
    for basename in basenames:
        plant_df = pd.read_csv(join(csv_path, basename))
        if not plant_df.empty:
            if random()>split_rate:
                train_df = pd.concat([train_df, plant_df])
            else:
                val_df = pd.concat([val_df, plant_df])
    train_df.to_csv(join(dirname, 'train.csv'), index=None)
    val_df.to_csv(join(dirname, 'val.csv'), index=None)

def class_text_to_int(row_label, plants):
    n = 0
    for plant in plants:
        n += 1
        if row_label == plant:
            return  n

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, plants):
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
        row['class'] = str(row['class'])
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], plants))

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

def csv2tfrecord(plants, split_rate):
    img_to_path = 'tmp/images'
    csv_to_path = 'tmp/annotations'

    if os.path.exists(img_to_path):
        rmtree(img_to_path)
    os.mkdir(img_to_path)
    if os.path.exists(csv_to_path):
        rmtree(csv_to_path)
    os.mkdir(csv_to_path)
    for plant in plants:
        img_from_path = join('tmp', plant, 'images')
        csv_from_path = join('tmp', plant, 'annotations')
        csv_paths = os.listdir(csv_from_path)
        for csv_path in csv_paths:
            filename = os.path.splitext(csv_path)[0]
            img_path = filename+'.png'
            copyfile(join(img_from_path, img_path), join(img_to_path, img_path))
            copyfile(join(csv_from_path, csv_path), join(csv_to_path, csv_path))

    merge_csv(csv_to_path, split_rate)
    dirname = os.path.split(csv_to_path)[0]
    img_path = join(dirname, 'images')
    stuffs=['tmp/train', 'tmp/val']

    for stuff in stuffs:
        csv_path = stuff+'.csv'
        record_path = stuff+'.record'

        writer = tf.python_io.TFRecordWriter(record_path)
        examples = pd.read_csv(csv_path)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, img_path, plants)
            writer.write(tf_example.SerializeToString())
        writer.close()
        output_path = join(os.getcwd(), record_path)
        print('Successfully created the TFRecords: {}'.format(output_path))

if __name__ == '__main__':
    csv2tfrecord(plants = ['996', '997', '998'], split_rate=0.3)
