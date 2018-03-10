from sklearn.model_selection import train_test_split
import numpy as np
import os
import h5py
import time
from os.path import join
import cv2
import argparse
from random import random, choice
from Utils import read_txt
from scipy.misc import imresize

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, BatchNormalization,Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# set_session(tf.Session(config=config))

PATCH_SIZE = 25
radius = int((PATCH_SIZE-1)/2)
CHANNEL = 3

def load_data(data_path):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    train_path = join('tmp', data_path, 'train')
    test_path = join('tmp', data_path, 'test')
    labels = os.listdir(train_path)
    num_label = len(labels)
    for label in labels:
        folder_path = join(train_path, label)
        paths = os.listdir(folder_path)
        for path in paths:
            image = cv2.imread(join(folder_path, path))
            image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
            x_train.append(np.array(image, dtype = np.float32)/255)
            y_train.append(label)

        folder_path = join(test_path, label)
        paths = os.listdir(folder_path)
        for path in paths:
            image = cv2.imread(join(folder_path, path))
            image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
            x_test.append(np.array(image, dtype = np.float32)/255)
            y_test.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    num_y_train = len(y_train)
    y_train = np_utils.to_categorical(y_train)
    y_train = y_train.reshape(num_y_train, 1, 1, num_label)
    num_y_test = len(y_test)
    y_test = np_utils.to_categorical(y_test)
    y_test = y_test.reshape(num_y_test, 1, 1, num_label)

    return x_train, x_test, y_train, y_test, num_label, num_y_train

def network(num_label, epoch, lr):
    model = Sequential()
    model.add(Conv2D(64, (7, 7), input_shape = (None, None, CHANNEL), activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (7, 7),  activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), activation = 'relu', kernel_initializer='he_normal'))
    model.add(Conv2D(128, (5, 5), activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), activation = 'relu', kernel_initializer='he_normal'))
    model.add(Conv2D(num_label, (3, 3), activation = 'softmax'))

    adam=Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
    return model

def train_model(model_name, data_path, checkpoint):
    save_path = join('models', model_name)
    x_train, x_test, y_train, y_test, num_label, train_num = load_data(data_path)
    # datagen = ImageDataGenerator(samplewise_center=False,samplewise_std_normalization=False,rotation_range=1,zoom_range=0.3,shear_range=0,vertical_flip=True,horizontal_flip=True,fill_mode="nearest")

    lr = 0.0001
    epoch = 100
    if train_num < 10000:
        batch_size = 64
    elif train_num < 100000:
        batch_size = 256
    else:
        batch_size = 1024
    model = network(num_label, epoch, lr)
    model_json = model.to_json()
    with open(save_path + '.json', "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk")

    if not checkpoint == 'None':
        model.load_weights(checkpoint)
        print('Using transfer learning')

    tb = TensorBoard(log_dir='tmp/logs/model')
    estop = EarlyStopping(monitor='val_loss', patience=10)
    save_checkpoint = ModelCheckpoint(save_path + '.h5', monitor="val_loss",save_best_only=True, verbose=1)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size=batch_size, callbacks=[tb, estop, save_checkpoint])
    # model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), validation_data = (x_test, y_test), steps_per_epoch = len(x_train) / batch_size, epochs = epoch, callbacks = [tb, estop, save_checkpoint])
    model.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name.")
    parser.add_argument("--data_path", type=str, help="Data path.")
    parser.add_argument("--checkpoint", type=str, default='None', help="Using checkpoint.")
    args = parser.parse_args()

    train_model(model_name = args.model_name, data_path = args.data_path, checkpoint = args.checkpoint)
