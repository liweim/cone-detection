import sklearn.model_selection
import numpy as np
import os
from os.path import join
import h5py
import time
import cv2

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
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
CHANNEL=3

def load_data(labels):
    images = []
    classes = []

    num_class = 0
    back_folder_path = join('data', str(labels[0]))
    back_paths = os.listdir(back_folder_path)
    num_back = len(back_paths)
    for back_path in back_paths:
        back_image = cv2.imread(join(back_folder_path, back_path))
        back_image = cv2.resize(back_image, (PATCH_SIZE, PATCH_SIZE))
        images.append(back_image)
        classes.append(num_class)

    for label in labels[1:]:
        num_class += 1
        front_folder_path = join('data', str(label))
        front_paths = os.listdir(front_folder_path)
        num_front = len(front_paths)
        num_copy = int(num_back/num_front)
        for front_path in front_paths:
            front_image = cv2.imread(join(front_folder_path, front_path))
            front_image = cv2.resize(front_image, (PATCH_SIZE, PATCH_SIZE))
            for i in range(num_copy):
                images.append(front_image)
                classes.append(num_class)

    x = np.array(images, dtype = np.float32)
    x /= 255
    y = np.array(classes)
    num_y = len(y)
    y = np_utils.to_categorical(y)
    y = y.reshape(num_y, 1, 1, num_class+1)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test

def network(num_class):
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
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def train_model(save_path, labels, transfer_learning = 1):
    batch_size = 256
    x_train, x_test, y_train, y_test=load_data(labels)
    datagen = ImageDataGenerator(samplewise_center=False,samplewise_std_normalization=False,rotation_range=0.3,zoom_range=0.2,shear_range=0,vertical_flip=True,horizontal_flip=True,fill_mode="nearest")
    #datagen.fit(x_train)
    #datagen = ImageDataGenerator(zoom_range=0.2,fill_mode="nearest")

    num_class=len(labels)
    model = network(num_class)
    if transfer_learning == 1:
        #model = model_from_json(save_path+'.json')
        model.load_weights(save_path+'.h5')
    tb = TensorBoard(log_dir='tmp/logs/final_model')
    estop = EarlyStopping(monitor='val_loss', patience=20)
    checkpoint = ModelCheckpoint(save_path+'.h5', monitor="val_loss",save_best_only=True, verbose=1)
    #model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=256, callbacks=[tb,estop,checkpoint])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), validation_data = (x_test, y_test), steps_per_epoch = len(x_train) / batch_size, epochs = 100, callbacks = [tb, estop, checkpoint])
    model.summary()

    model_json = model.to_json()
    with open(save_path+'.json', "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk")

    model_eval = model
    model_eval.load_weights(save_path+'.h5')
    #model.save_weights(save_path+'.h5')
    start=time.clock()
    scores = model_eval.evaluate(x_test, y_test, verbose=0)
    print('Evaluate time: {}'.format(time.clock()-start))
    print("Test loss: %.4f, Test accuracy: %.2f%%" % (scores[0],scores[1]*100))

if __name__ == '__main__':
    train_model(save_path="models/model_cone",labels=['background', 'yellow', 'blue'], transfer_learning = 0)
