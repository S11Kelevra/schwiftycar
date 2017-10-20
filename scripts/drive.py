'''
Drive entry point
'''

import sys
import os
from os.path import dirname
from os.path import basename
import time
import argparse
import atexit
import csv
import signal
#import logging
sys.path.append(dirname(dirname(os.path.realpath(__file__))))

import cv2
import numpy as np
import pygame
import pyping

from robocar42 import config, util, models
from robocar42.car import Car
from robocar42.display import Display

logger = util.configure_log('drive')

NUM_CLASSES = 4
delta_time = 1000
conf_level = 0.3

rc_car = Car(config.vehicle_parser_config('vehicle.ini'))       # parses values from vehicle.ini into an object Car
disp_conf = config.display_parser_config('display.ini')         # parses values from display.ini
cam_1 = config.camera_parser_config('camera_1.ini')             # parses values from camera_1.ini
cam_2 = config.camera_parser_config('camera_2.ini')             # parses values from camera_2.ini
model_conf = config.model_parser_config('model_1.ini')          # sets values from model_1.ini

links = ['/fwd', '/fwd/lf', '/fwd/rt', '/rev', '/rev/lf', '/rev/rt', '']    # list of commands for a controller
actions = [pygame.K_UP,pygame.K_LEFT,pygame.K_RIGHT,pygame.K_DOWN]
attributes=['cam_1', 'cam_2', 'action']                         # list of attributes
rev_action = 3                                                  # other drive values are 0 - 2

def check_cameras():
    '''
    checks to see if the cameras are online
    '''
    print("Pinging those hot cams!")
    try:
        cam_1_ret = pyping.ping(cam_1['cam_url'], udp=True).ret_code    # pings camera1
        cam_2_ret = pyping.ping(cam_2['cam_url'], udp=True).ret_code    # pings camera2
        if not cam_1_ret and not cam_2_ret:
            return True                                                 # returns true if cameras ping
    except:                                                             # if there's an exception, pass
        pass
    return False

def ctrl_c_handler(signum, frame):      # takes the signal number and the current stack frame
    '''
    closes pygame
    '''
    print("Quitting!")
    pygame.quit()
    exit(0)

def cleanup(display):       # takes the object Display from display.py
    '''
    closes display
    '''
    print("stopping display")
    display.stop()
    exit(0)

def conv_to_vec(action):    # takes a string containted within links ex:('/fwd/lf')
    '''
    Convert old car actions into vect for car module
    '''
    print("Converting actions to vectors!")
    logger.debug("conv_to_vec=%s", action)
    req = [0] * 4
    if '/fwd' in action: req[0] = 1     # forward sets req[0] to 1
    elif '/rev' in action: req[0] = -1  # reverse sets req[0] to -1
    else: req[0] = 0                    # sitting still sets req[0] to 0
    req[1] = rc_car.max_speed           # sets req[1] to the max_speed
    if '/lf' in action: req[2] = -1     # left sets req[2] to -1
    elif '/rt' in action: req[2] = 1    # right sets req[2] to 1
    else: req[2] = 0                    # straight sets req[2] to 0
    req[3] = -1                         # req[3] = -1
    logger.debug("return of conv_to_vec: req=%s", req)
    return req

def concantenate_image(images):                     # takes a set of images
    '''
    Processes images and return normalized/combined single image
    '''
    print("Concatinating camera images!")
    images[0] = np.swapaxes(images[0], 1, 0)        # swaps the axes in the images (mirror)
    images[1] = np.swapaxes(images[1], 1, 0)
    aimage = np.concatenate(tuple(images), axis=1)  # concatenates the images along axis 1
    aimage = cv2.resize(aimage,
                disp_conf['sdshape'],
                interpolation=cv2.INTER_AREA)       #resizes the image
    aimage = aimage / 255.
    aimage = aimage - 0.5
    return aimage

def auto_drive(images):                                         # takes a set of images
    print ("Autopilot engaged!")
    if images:                                                  # if there are images...
        prec_image = concantenate_image(images)                 # returns a concatonated and resized image
        pred_act = model.predict(np.array([prec_image]))[0]     # creates an array based on prec_image
        logger.info("Lft: %.2f | Fwd: %.2f | Rht: %.2f | Rev: %.2f" %
            (pred_act[1], pred_act[0], pred_act[2], pred_act[3]))   # logs the percentage of each action
        act_i = np.argmax(pred_act)                             # returns the index of the maximum values in pred_act
        action = act_i if (pred_act[act_i] >= conf_level) else rev_action   # sets the action if it is 30+% confident
        if act_i < len(links):
            rc_car.drive(conv_to_vec(links[action]))            # converts the action to a vector, then executes the action
        return action, True
    else:                                                       # if there are no images, log and return None/False
        logger.error("Error: no images for prediction")
        return None, False

def manual_drive(intent, teach=False):          # takes intent (key up/left,down/right) and the input teach(str)
    print("Moving to manual controls!")
    for act_i in range(len(actions)):
        tmp = actions[act_i]
        if tmp==intent:                         # matches the intended action with the list of actions
            logger.debug("acting out %d" % tmp)
            if not teach:                       # if teach if false...
                rc_car.drive(conv_to_vec(links[act_i]))
            return act_i, True
    return None, False

def drive(auto, set_name, rec_dirs=None, teach=False):  # takes args.auto (str), args.train (str), rec_dirs(path/str), and args.teach(str)
    ot = 0                                              # original time
    img_ot = 0                                          # image original time
    running = True
    logger.debug("Starting drive: auto=%s, set_name=%s, rec_dirs=%s, teach=%s", auto, set_name, rec_dirs, teach)
    intent = 0
    label_path = os.path.join(set_name, set_name+'.csv') # makes a .csv(comma separated values) file of the set name
    label_path = os.path.join(config.pre_path, label_path)

    with open(label_path, 'w') as csv_file:             # opens setname.cvs in write
        entry = None
        csv_writer = csv.writer(csv_file, delimiter=',')    # returns an object converting the data into delimited strings
        csv_writer.writerow(attributes)                 # writes attributes to the file object (cam_1, can_2, action)
        while running:                                  # set true in variable declarations
            for event in pygame.event.get():            # removes events in the queue and returns them in a list, then reads the events
                if event.type == pygame.QUIT:           # if pygame quits, running becomes false
                    running = False
            ct = time.time()                            # returns the time in seconds
            drive = True if (ct - ot) * 1000 > rc_car.exp + delta_time else drive
            surface, images, filenames = disp.show((rec_dirs), rec_dirs)    # ?
            logger.debug("screen.blit1")
            screen.blit(surface[0], (0,0))              # renders objects onto the screen
            logger.debug("screen.blit2")
            screen.blit(surface[1], (disp_conf['oshape'][0],0))
            logger.debug("display.flip")
            pygame.display.flip()
            logger.debug("key.get_pressed")
            keys = pygame.key.get_pressed()             # returns the state of all keyboard buttons
            for act_i in range(len(actions)):           # sets the intent to the action input
                tmp = actions[act_i]                    # sets tmp = to one of the actions (up, left, right, down)
                if keys[tmp]:
                    logger.debug("Key pressed %d" % tmp)
                    intent = tmp                        # sets intent equal to tmp (one of the keys up/left/right/down)
            if keys[pygame.K_ESCAPE] or keys[pygame.K_q] or \
                pygame.event.peek(pygame.QUIT):         # If exited with esc, q, or the quit event
                logger.debug("Exit pressed")           # then notify the debug log, and return to main
                return
            if drive and not auto:                      # if in manual...
                logger.debug("Manual Drive. drive=%s", drive)
                drive = False
                car_act, flag = manual_drive(intent, teach)
                if flag:                                # if manual_drive returned the 'True' flag
                    entry = [                           # gathers info for a log entry (found in final if statement)
                        filenames[0],
                        filenames[1],
                        car_act
                    ]
                intent = 0
                ot = ct
            if keys[pygame.K_a]:                        # toggles auto on
                auto = True
                logger.info("Autopilot mode on!")
            if keys[pygame.K_s]:                        # toggles auto off
                auto = False
                logger.info("Autopilot mode off!")
            keys = []
            logger.debug("pre-pygame.event.pump")
            pygame.event.pump()                         # internally process pygame event handlers
            if images and auto and drive:
                logger.debug("Auto drive")
                drive = False
                cat_act, flag = auto_drive(images)      # calls auto_drive
                ot = ct
                if flag:                                # if auto_drive returned the 'True' flag
                    entry = [                           # gathers info for a log entry (found in final if statement)
                        filenames[0],
                        filenames[1],
                        car_act
                    ]
            if entry:                                   # writes entry information to the csv object then clears entry
                logger.debug("csv-writer: %s", entry)
                csv_writer.writerow(entry)
                entry = None

def gen_default_name():
    '''
    generates a name for the training images
    '''
    rec_folder = "rec_%s" % time.strftime("%d_%m_%H_%M")    # sets folder name after day,month,hour,minute
    print("Generating dank meme: " + rec_folder)
    return rec_folder

def build_parser():
    print("Building parser!")
    parser = argparse.ArgumentParser(description='Drive')
    parser.add_argument(
        '-auto',                                        # autonomous or manual
        action='store_true',
        default=False,                                  # default is auto
        help='Auto on/off. Default: off')
    parser.add_argument(
        '-teach',                                       # teach on or off
        action='store_true',
        default=False,                                  # default off
        help='Teach on/off. Default: off')
    parser.add_argument(
        '-model',                                       # specify behavioural model
        type=str,
        help='Specify model to use for auto drive')
    parser.add_argument(
        '-train',                                       # specify the name of the training set
        type=str,
        help='Specify name of training image set',
        default=gen_default_name()                      # generates a default name unless specified
        )
    return parser

def check_arguments(args):              # takes the list of arguments (args)
    '''
    checks that model was input
    '''
    print("Checking arguments!")
    if args.auto and not args.model:    # checks the input arguments for auto and model
        return False                    # if no model is found but auto is, return False
    return True

def check_set_name(rec_folder):     # takes a path name as a string (?)
    '''
    checks the set names folder/0 and folder/1
    to be sure they don't exist. Otherwise it'd
    overwrite.
    '''
    print("Checking set name!")
    rec_dirs = [rec_folder+'/'+str(i) for i in range(2)]
    for directory in rec_dirs:
        if os.path.exists(directory):
            return False
    return True

if __name__ == '__main__':
    print("Connecting to skynet")
    signal.signal(signal.SIGINT, ctrl_c_handler)    # set handler number
    parser = build_parser()                         # builds string from inputs
    args = parser.parse_args()                      # parses string into arguments

    if not check_arguments(args):                   # checks if the arguments are valid
        logger.error("Error: Invalid command line arguments")

    if check_cameras():                             # if cameras are running, pass true to the model.model
        rec_folder = os.path.join(config.pre_path, args.train)      # concatenates path and args.train
        if not check_set_name(rec_folder):                          # checks for directory names
            logger.error("Error: Invalid setname. Name is unavailable.")
        rec_dirs = [rec_folder+'/'+str(i+1) for i in range(2)]              # creates directories name/0 name/1
        for directory in rec_dirs:
            print directory
            os.makedirs(directory)

        model = models.model(True, model_conf['shape'],     # sets convolutional model
                    NUM_CLASSES,                        # NUM_CLASSES = 4 (set at beginning of file)
                    args.model)                         # adds the model input
        rc_car.start()                                  # tells car to start but not move (TODO? if statement error?)
        disp = Display('main', disp_conf, ['camera_1.ini', 'camera_2.ini']) # sets disp to Display(robocar42/display.py)
        atexit.register(cleanup, disp)                          # upon close, runs cleanup w/ the arg (disp), closing the display
        logger.debug("initializing pygame")
        pygame.init()                                           # initializes all imported pygame modules
        screen = pygame.display.set_mode(disp_conf['doshape'])  # set screen to doshapeX and Y from display.ini
        drive(args.auto, args.train, rec_dirs, args.teach)      # starts the drive function
        logger.debug("exiting drive, onto disp.stop")
        disp.stop()             # stops the display
        logger.debug("entering pygame.quit")
        pygame.quit()           # quits pygame
    else:                       # occurs if check_cameras returned false, meaning they could not be pinged.
        logger.error("Error: Unable to reach cameras!"
            "\nPlease make sure to check if the car is"
            " on and batteries are fully charged.\nTry again.")
	exit(0)
