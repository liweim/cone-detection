from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import cv2

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

from os.path import join
import glob
import xml.etree.ElementTree as ET
from shutil import copyfile, rmtree
from random import random

# resize_rate = 1
# heightUp = 0
# height = 360
# width = 640

def xml_to_csv(path, split_rate=0.3):
	xml_path = '../annotations/'+path+'/rectified'

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
						# int(int(member[4][0].text)*resize_rate),
						# min(int(int(member[4][1].text)*resize_rate-heightUp),height),
						# int(int(member[4][2].text)*resize_rate),
						# min(int(int(member[4][3].text)*resize_rate-heightUp),height)
						int(member[4][0].text),
						int(member[4][1].text),
						int(member[4][2].text),
						int(member[4][3].text)
						)
				tmp_list.append(value)

				img_path = '../annotations/'+path+'/rectified/'+value[0]
				print(img_path)
				img = cv2.imread(img_path)
				cv2.rectangle(img, (value[4],value[5]), (value[6],value[7]), (0,0,255), 1)
				cv2.namedWindow('img', cv2.WINDOW_NORMAL)
				cv2.imshow('img',img)
				cv2.waitKey(0)


			except ValueError:
			    pass
		if random()>split_rate:
		    train_list+=tmp_list
		else:
		    val_list+=tmp_list
	column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

	train_df = pd.DataFrame(train_list, columns=column_name)
	train_df.to_csv(join('tmp/train_'+path+'.csv'), index=None)
	val_df = pd.DataFrame(val_list, columns=column_name)
	val_df.to_csv(join('tmp/val_'+path+'.csv'), index=None)

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
	# print(join(path, '{}'.format(group.filename)))
	with tf.gfile.GFile(join(path, '{}'.format(group.filename)), 'rb') as fid:
		encoded_img = fid.read()
	encoded_img_io = io.BytesIO(encoded_img)
	image = Image.open(encoded_img_io)
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
	    classes.append(class_text_to_int(row['class']))

	tf_example = tf.train.Example(features=tf.train.Features(feature={
	    'image/height': dataset_util.int64_feature(height),
	    'image/width': dataset_util.int64_feature(width),
	    'image/filename': dataset_util.bytes_feature(filename),
	    'image/source_id': dataset_util.bytes_feature(filename),
	    'image/encoded': dataset_util.bytes_feature(encoded_img),
	    'image/format': dataset_util.bytes_feature(image_format),
	    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
	    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
	    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
	    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
	    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
	    'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))
	return tf_example

def xml2tfRecord(paths, split_rate):
	for path in paths:
		xml_to_csv(path, split_rate)
	stuffs=['tmp/train', 'tmp/val']
	for stuff in stuffs:
		record_path = stuff+'.record'
		writer = tf.python_io.TFRecordWriter(record_path)
		for path in paths:
			img_path = '../annotations/'+path+'/rectified'
			csv_path = stuff+'_'+path+'.csv'

			examples = pd.read_csv(csv_path)
			grouped = split(examples, 'filename')
			for group in grouped:
			    tf_example = create_tf_example(group, img_path)
			    writer.write(tf_example.SerializeToString())
		writer.close()
		output_path = join(os.getcwd(), record_path)
		print('Successfully created the TFRecords: {}'.format(output_path))

if __name__ == '__main__':
    # xml2tfRecord(paths = ['skidpad1', 'sunnny', 'rainy', 'rainy2'], split_rate = 0.3)
    xml2tfRecord(paths = ['skidpad1'], split_rate = 0.3)
