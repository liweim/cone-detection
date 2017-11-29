from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import h5py
import time
from os.path import join
import cv2
import argparse
import random

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, BatchNormalization,Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

PATCH_SIZE = 25
CHANNEL = 3
num_class = len(os.listdir('images')) + 1

def load_data(train_num):
    images = []
    classes = []
    labels = []
    imgs = []
    masks = []
    radius = 12

    annotation_folder_path = join('annotations')
    image_folder_path = join('images')
    cone_ids = os.listdir(annotation_folder_path)
    for cone_id in cone_ids:
        annotation_paths = os.listdir(join(annotation_folder_path, cone_id))
        for img_path in annotation_paths:
            img = cv2.imread(join(image_folder_path, cone_id, img_path))
            mask = cv2.imread(join(annotation_folder_path, cone_id, img_path), 0)
            imgs.append(img)
            masks.append(mask)
            labels.append(int(cone_id))

    num = np.zeros(num_class)
    random_range = range(len(imgs))
    while max(num < train_num):
        random_id = random.choice(random_range)
        img = imgs[random_id]
        mask = masks[random_id]
        label = labels[random_id]
        rs, cs = mask.shape
        r = random.choice(range(radius, rs-radius))
        c = random.choice(range(radius, cs-radius))
        image = img[r-radius:r+radius+1, c-radius:c+radius+1, :]
        if np.max(image) > 0:
            if mask[r, c] > 0 and num[label] < train_num:
                images.append(image)
                classes.append(label)
                num[label] += 1
            if mask[r, c] == 0 and num[0] < train_num:
                images.append(image)
                classes.append(0)
                num[0] += 1

    print(np.array(images).shape)
    x = np.array(images, dtype = np.float32)
    x /= 255
    y = np.array(classes, dtype = np.uint8)
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 1, 1, y.shape[1])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test

def network():
    model = Sequential()
    model.add(Conv2D(32, (7, 7), input_shape = (None, None, CHANNEL), activation='relu'))
    model.add(Conv2D(32, (7, 7),  activation = 'relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (5, 5), activation = 'relu'))
    model.add(Conv2D(64, (5, 5), activation = 'relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(Conv2D(num_class, (3, 3), activation = 'softmax'))

    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
    return model

def train_model(train_num):
    save_path = join('models', 'model_cone')
    batch_size = 256
    x_train, x_test, y_train, y_test = load_data(train_num)

    model = network()
    model_json = model.to_json()
    with open(save_path + '.json', "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk")

    tb = TensorBoard(log_dir='tmp/logs/model')
    estop = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = ModelCheckpoint(save_path + '.h5', monitor="val_loss",save_best_only=True, verbose=1)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=256, callbacks=[tb,estop,checkpoint])
    model.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_num", type=int, help="Train num")
    args = parser.parse_args()

    train_model(train_num = args.train_num)
