'''This module describes the models for training'''

import os
import sys
import time
sys.path.append('..')

import keras
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import flip_axis, random_shift
from keras.utils import to_categorical

from robocar42 import config

def model(load, shape, classes_num, tr_model=None):
    '''
    Returns a convolutional model from file or to train on.
    '''
    if load and tr_model: return load_model(tr_model)

    conv3x3_l, dense_layers = [24, 32, 40, 48], [512, 64, 16]

    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='elu', input_shape=shape))
    model.add(MaxPooling2D())
    for i in range(len(conv3x3_l)):
        model.add(Conv2D(conv3x3_l[i], (3, 3), activation='elu'))
        if i < len(conv3x3_l) - 1:
            model.add(MaxPooling2D())
    model.add(Flatten())
    for dl in dense_layers:
        model.add(Dense(dl, activation='elu'))
        model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )
    return model

def get_X_y(data_files):
    '''
    Read the csv files and generate X/y pairs.
    '''
    pass

def _generator(batch_size, classes, X, y):
    '''
    Generate batches for training
    '''
    pass

def train(conf, model, train_name=None):
    '''
    Load the network and data, fit the model, save it
    '''
    if model:
        net = model(load=True, shape=conf['shape'], tr_model=model)
    else:
        net = model(load=False, shape=conf['shape'])
    net.summary()
    X, y, = get_X_y()
    Xtr, Xval, ytr, yval = train_test_split(
                                X, y,
                                test_size=conf['val_split'],
                                random_state=random.randint(0, 100)
                           )
    tr_classes = [[] for _ in range(conf[''])]
    for i in range(len(ytr)):
        for j in range(NUM_CLASSES):
            if ytr[i][j]:
                tr_classes[j].append(i)
    val_classes = [[] for _ in range(NUM_CLASSES)]
    for i in range(len(yval)):
        for j in range(NUM_CLASSES):
            if yval[i][j]:
                val_classes[j].append(i)

    net.fit_generator(
        _generator(conf['batch'], tr_classes, Xtr, ytr),
        validation_data=_generator(conf['batch'], val_classes, Xval, yval),
        validation_steps=max(len(Xval) // conf['batch'], 1),
        steps_per_epoch=1,
        epochs=1
    )
    net.fit_generator(
        _generator(conf['batch'], tr_classes, Xtr, ytr),
        validation_data=_generator(conf['batch'], val_classes, Xval, yval),
        validation_steps=max(len(Xval) // conf['batch'], 1),
        steps_per_epoch=conf['steps'],
        epochs=conf['epochs']
    )
    net.save()
    