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

def model(load, shape, classes_num, tr_model=None):             # called in drive.py (inputs: load=True, shape from model_1.ini, 4, None(unless specified))
    '''
    Returns a convolutional model from file or to train on.
    '''
    if load and tr_model: return load_model(tr_model)           # if specific model is called to train, load_model

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

def get_X_y(data_files):
    '''
    Read the csv files and generate X/y pairs.
    '''
    # Added from older code base
    """Read the log file and turn it into X/y pairs. Add an offset to left images, remove from right images."""
    X, y = [], []
    with open(data_file) as fin:
        reader = csv.reader(fin)
        next(reader, None)
        for img, command in reader:
            X.append(img.strip())
            y.append(int(command))
    return X, to_categorical(y, num_classes=NUM_CLASSES)
    # ____________________________________________________

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
            image, command = process_image(img_dir + X[sample_index], command, augment=augment)
            batch_X.append(image)
            batch_y.append(command)
        yield np.array(batch_X), np.array(batch_y)
 # ____________________________________________________

def train(conf, model, train_name=None):
    '''
    Load the network and data, fit the model, save it
    '''
    print("Starting train!")
    if model:                       # if a model was entered, load it
        print("Model entered!")
        net = model(load=True, shape=conf['shape'], tr_model=model)
    else:                           # otherwise create a new model
        print("No model entered")
        net = model(load=False, shape=conf['shape'])
    net.summary()                   # prints a summary representation of the model
    X, y, = get_X_y(train_name)     # give list of files
    Xtr, Xval, ytr, yval = train_test_split(    # test_train_split: returns list containing train-test split of inputs
                                X, y,
                                test_size=conf['val_split'],    # val_split = 0.15 from model_1.ini
                                random_state=random.randint(0, 100) # the seed used by the RNG
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

    net.fit_generator(                  # returns a History object
        _generator(conf['batch'], tr_classes, Xtr, ytr),
        validation_data=_generator(conf['batch'], val_classes, Xval, yval),
        validation_steps=max(len(Xval) // conf['batch'], 1),    # total number of steps to yield from generator before stopping
        steps_per_epoch=1,              # steps to yield from generator before declaring one epoch finish and starting the next
        epochs=1                        # total number of iterations on the data
    )
    net.fit_generator(
        _generator(conf['batch'], tr_classes, Xtr, ytr),
        validation_data=_generator(conf['batch'], val_classes, Xval, yval),
        validation_steps=max(len(Xval) // conf['batch'], 1),
        steps_per_epoch=conf['steps'],  # (200), from model_1.ini
        epochs=conf['epochs']           # (10), from model_1.ini
    )
    net.save()          # saves the model as a .h5 file.
