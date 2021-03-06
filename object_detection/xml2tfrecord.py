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
import xml.etree.ElementTree as ET
from shutil import copyfile, rmtree
from random import random

def xml_to_csv(xml_path, split_rate=0.2):
    data_path = os.path.split(xml_path)[0]
    img_path = join(data_path,'images')

    train_list = []
    val_list = []
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
        else:
            val_list+=tmp_list
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    train_df = pd.DataFrame(train_list, columns=column_name)
    train_df.to_csv(join('tmp/train.csv'), index=None)
    val_df = pd.DataFrame(val_list, columns=column_name)
    val_df.to_csv(join('tmp/val.csv'), index=None)
    print('Successfully converted xml to csv.')
    return

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'blue':
        return  1
    if row_label == 'yellow':
        return  2
    if row_label == 'orange':
        return  3
    if row_label == 'orange2':
        return  4
    else:
        print(row_label)
        return 0

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
		length = np.linalg.norm((row['xmax'],row['ymax'])-(row['xmin'],row['ymin']))
		print(length)
		row['class'] = str(row['class'])
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
	xml_to_csv(xml_path, split_rate)
	dirname = os.path.split(xml_path)[0]
	img_path = join(dirname, 'images')
	stuffs=['tmp/train', 'tmp/val']

	for stuff in stuffs:
	    csv_path = stuff+'.csv'
	    record_path = stuff+'.record'
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
	xml2tfRecord(xml_path='tmp/annotations', split_rate=0.3)
