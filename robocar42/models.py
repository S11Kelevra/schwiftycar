'''This module describes the models for training'''

import os
import sys
import time
sys.path.append('..')

import keras
import csv, random
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import flip_axis, random_shift
from keras.utils import to_categorical
from robocar42 import config
import numpy as np

model_conf = config.model_parser_config('model_1.ini')

NUM_CLASSES = 4

def model(load, shape, classes_num = NUM_CLASSES, tr_model=None):             # called in drive.py (inputs: load=True, shape from model_1.ini, 4, None(unless specified))
    '''
    Returns a convolutional model from file or to train on.
    '''

    if load and tr_model:                                       # if specific model is called to train, load_model
        return load_model(tr_model)

    conv3x3_l, dense_layers = [24, 32, 40, 48], [512, 64, 16]   # creates tuples for conv3x3_l and desne_layers

    model = Sequential()                                        # creates an empty linear stack of layers
    model.add(Conv2D(16, (5, 5), activation='elu', input_shape=shape))  # adds a layer with these perameters: (16) convolutional filters, (5) rows by (5) columns in each convolutional kernal, activation = elu (Exponential Linear Unit function), shape = 120, 320, 3(depth width height of each digit image)
    model.add(MaxPooling2D())                           # layer reduces the number of parameters in the model by taking the max of the values from the previous filter
    for i in range(len(conv3x3_l)):                     # adds additional layers for [24, 32, 40, 48]
        model.add(Conv2D(conv3x3_l[i], (3, 3), activation='elu'))
        if i < len(conv3x3_l) - 1:                      # adds another maxpooling2d layer between layers until last layer
            model.add(MaxPooling2D())
    model.add(Flatten())                                # adds a flattening layer (makes the weights from the convolutional layers 1 dimensional)
    for dl in dense_layers:                             # adds a Dense Layer as well as a Dropout Layer for all 3 dense_layers
        model.add(Dense(dl, activation='elu'))          # first parameter is equal to output size
        model.add(Dropout(0.5))                         # applies a 50% dropout to its inputs (previous layers outputs in this case)
    model.add(Dense(classes_num, activation='softmax')) # makes a final layer with 4 possible outputs (from classes_num)
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",                               # default settings for "Adam - A Method for Stochastic Optimization"
        metrics=['accuracy']                            # metrics used for classification
    )
    return model

def get_X_y(csv_file, cam_num):
    '''
    Read the csv files and generate X/y pairs.
    '''
    # Added from older code base
    """Read the log file and turn it into X/y pairs. Add an offset to left images, remove from right images."""
    X, y = [], []
    #ata_file = os.path.join(data_set, os.path.basename(data_set)) + ".csv"

    with open(csv_file) as fin:
        reader = csv.reader(fin)
        next(reader, None)
        for img1, img2, command in reader:
            if os.path.basename(cam_num) == '1':
                _to_be_added = cam_num
                _to_be_added = os.path.join(_to_be_added, img1)
            else:
                _to_be_added = cam_num
                _to_be_added = os.path.join(_to_be_added, img2)
            if not os.path.exists(_to_be_added):
                print("Image %s does not exist", _to_be_added)
                sys.exit(1)
            #if not os.path.exists(_to_be_added2):
                #print("Image %s does not exist", _to_be_added2)
                #sys.exit(1)
            X.append(_to_be_added)
            y.append(int(command))
    return X, to_categorical(y, num_classes=NUM_CLASSES)
    # ____________________________________________________

def load_image(path):
    """Process and augment an image."""
    print("Loading : " + path)
    shape=[model_conf['shape'][0],model_conf['shape'][1]]
    #print(shape)
    image = load_img(path)

    return aimage

def process_image(image):
    """Process and augment an image."""
    #tmp = image.size
    #print(image.size)
    #print(0, tmp[1] // 3, tmp[0], tmp[1])
    #image = image.crop((0, tmp[1] // 3, tmp[0], tmp[1]))
    #print("Cropped = %s", image.size)
    aimage = img_to_array(image)
    aimage = aimage.astype(np.float32) / 255.
    aimage = aimage - 0.5
    #print(aimage.shape)
    return aimage

def _generator(batch_size, classes, X, y):
    '''
    Generate batches for training
    '''
    # added from older code base
    """Generate batches of training data forever."""
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            # random.seed(random.randint(0, 9001))
            class_i = random.randint(0, NUM_CLASSES - 1)
            # sample_index = random.randint(0, len(classes[class_i]) - 1)
            sample_index = random.choice(classes[class_i])
            command = y[sample_index]
            image = load_image(X[sample_index])
            batch_X.append(image)
            batch_y.append(command)
        yield np.array(batch_X), np.array(batch_y)
 # ____________________________________________________

def train(csv_file, model_name=None, cam_num=None, is_new_model=False):
    '''
    Load the network and data, fit the model, save it
    '''
    print("Starting train!")

    if not is_new_model:                                   # if a model was entered, load it
        print("Using model!")
        net = model(load=True, shape=model_conf['shape'], tr_model=model_name)
    else:                                       # otherwise create a new model
        print("Generating model!")
        net = model(load=False, shape=model_conf['shape'], tr_model=model_name)
    net.summary()                               # prints a summary representation of the model
    X, y, = get_X_y(csv_file, cam_num)                   # give list of files
    Xtr, Xval, ytr, yval = train_test_split(    # test_train_split: returns list containing train-test split of inputs
                                X, y,
                                test_size=model_conf['val_split'],            # val_split = 0.15 from model_1.ini
                                random_state=random.randint(0, 100)     # the seed used by the RNG
                           )
    tr_classes = [[] for _ in range(NUM_CLASSES)]
    for i in range(len(ytr)):
        for j in range(NUM_CLASSES):
            if ytr[i][j]:
                tr_classes[j].append(i)
    val_classes = [[] for _ in range(NUM_CLASSES)]
    for i in range(len(yval)):
        for j in range(NUM_CLASSES):
            if yval[i][j]:
                val_classes[j].append(i)
    tmp1= _generator(model_conf['batch'], tr_classes, Xtr, ytr)
    tmp4 = tmp1.next()
    #print("tmp1 : len %i", len(tmp4))
    #print(tmp4[0].shape)
    #print(tmp4[1].shape)
    tmp2= _generator(model_conf['batch'], val_classes, Xval, yval)
    #print("tmp2 : %s", tmp2.shape)
    #print(tmp2.next().shape)
    tmp3 = max(len(Xval) // model_conf['batch'], 1)
    #print(tmp3.shape)
    net.fit_generator(                  # returns a History object
        tmp1,
        validation_data=tmp2,
        validation_steps=tmp3,    # total number of steps to yield from generator before stopping
        steps_per_epoch=1,              # steps to yield from generator before declaring one epoch finish and starting the next
        epochs=1                        # total number of iterations on the data
    )
    net.fit_generator(
        _generator(model_conf['batch'], tr_classes, Xtr, ytr),
        validation_data=_generator(model_conf['batch'], val_classes, Xval, yval),
        validation_steps=max(len(Xval) // model_conf['batch'], 1),
        steps_per_epoch=model_conf['steps'],  # (200), from model_1.ini
        epochs=model_conf['epochs']           # (10), from model_1.ini
    )
    net.save(model_name)          # saves the model as a .h5 file.
