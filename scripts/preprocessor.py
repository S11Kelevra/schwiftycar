'''
This script processes the data: increasing the data set size, similarity
detection, and image augmentation.
'''

import sys
import os
import re
from os.path import dirname
from os.path import basename
import csv
import time
import argparse

import logging
sys.path.append(dirname(dirname(os.path.realpath(__file__))))

import numpy as np      # needed?
import cv2
import pandas as pd     # needed?

from robocar42 import util
from robocar42 import config
from robocar42 import preprocess

logger = util.configure_log('preprocess') # set the logger name for this script

op_list = [
    ('darken', preprocess.darken),
    ('brighten', preprocess.brighten),
    ('flip', preprocess.flip)
]

reverse = [0, 2, 1, 3]

def build_parser():
    '''
    Build Parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sim_cutoff',                                      # argument for similarity threshold, type=float default 0.1
        type=float,
        help="Image similarity threshold. Default: 0.1",
        default=0.1)
    parser.add_argument(
        '-equalize',                                        # choose to apply equalize histogram on images, default = no
        action='store_true',
        default=False,
        help='Apply equalize histogram on images, Default: False'
        )
    parser.add_argument(                                    # name of data set to preprocess
        'set_name',
        type=str,
        help="Name of data set"
    )
    return parser

def process_image(path, name, command, op_todo):
    '''
    Perform augmentation operations on images
    '''

    image_paths = [os.path.join(path[i], name[i]) for i in range(len(path))]
    aug_images = []

    for ops in op_todo:
        new_command = command
        for ind in range(len(image_paths)):
            img_orig = cv2.imread(image_paths[ind], 1)
            new_image = img_orig
            output_prepend = ""
            for op in ops:
                output_prepend += op[0]+"_"
                new_image = op[1](new_image)
                if op[0] == 'flip':
                    new_command = reverse[command]
            cv2.imwrite(
                filename=os.path.join(path[ind], output_prepend+name[ind]),
                img=new_image)
        aug_images.append([
            output_prepend+name[0],
            output_prepend+name[1],
            new_command])

    return aug_images

def augment(set_name, equalize=False):
    '''

    '''
    data_path = os.path.join(config.data_path, set_name)        # data_path = ROOT_DIR/data_sets/set_name
    label_path = os.path.join(set_name, set_name+'.csv')        # label_path = set_name/set_name.csv
    label_path = os.path.join(config.data_path, label_path)     # label_path = ROOT_DIR/data_sets/set_name/set_name/set_name.csv

    cam_1_path = os.path.join(data_path, '1')                   # cam1or2 = ROOT_DIR/data_sets/set_name/1or2
    cam_2_path = os.path.join(data_path, '2')

    op_todo = [
        ([op_list[0]]),
        ([op_list[1]]),
        ([op_list[2]]),
        ([op_list[0],op_list[2]]),
        ([op_list[1],op_list[2]])
    ]

    with open(label_path, 'r') as in_csv:                                   # open csv file in read
        for line in in_csv:                                                 # Line by line
            if re.search(r"(flip|autocont|equalize|darken|brighten)", line):# ???
                util.progress_bar(1, 1)
                return

    if equalize:                                                            # equalize images (default false)
        with open(label_path, 'r') as in_csv:                               # open csv file in read
            reader = csv.reader(in_csv, delimiter=',')                      # read in data with ',' dln
            attribute = next(reader, None)                                  # ??? needed?
            entries = list(reader)                                          # reader broken into entries list
            for entry in entries:                                           # for each element in entry
                img_1_path = os.path.join(cam_1_path, entry[0])             # img_x_path = ROOT_DIR/data_sets/set_name/x/entry[x]
                img_2_path = os.path.join(cam_2_path, entry[1])
                cam_1_img = preprocess.equalize(cv2.imread(img_1_path))     # cam_x_img equalized (contrast increase)
                cam_2_img = preprocess.equalize(cv2.imread(img_2_path))     # using cv2 functions
                cv2.imwrite(filename=img_1_path,img=cam_1_img)              # cv2.imwrite to save boosted image to
                cv2.imwrite(filename=img_2_path,img=cam_2_img)              # same directory (overwrite?)

    logger.info("Preprocessing images...")                                  # Begin log for actual preprocessing
    with open(label_path, 'a+') as io_csv:                                  # 
        io_csv.seek(0)
        reader = csv.reader(io_csv, delimiter=',')
        attribute = next(reader, None)
        entries = list(reader)
        cnt_total = len(entries)
        cnt_iter = 0
        util.progress_bar(cnt_iter, cnt_total)
        for entry in entries:
            logger.debug("working on %s" % str(entry))
            cnt_iter += 1
            util.progress_bar(cnt_iter, cnt_total)
            new_entries = process_image(
                [cam_1_path, cam_2_path],
                [entry[0],entry[1]],
                int(entry[-1]),
                op_todo)
            writer = csv.writer(io_csv, delimiter=',')
            for new_entry in new_entries:
                writer.writerow(new_entry)
            time.sleep(0.1)

def main():
    parser = build_parser()             # parses input (similarity threshold, equalize, name of data set)
    args = parser.parse_args()          # values taken into args

    logger.critical("Process Start")                # begin the log
    if not args.set_name:
        logger.error("ERROR - Invalid set name")    # logs error if no set by that name
        exit(0)                                     # exits

    pre_path = os.path.join(config.pre_path, args.set_name)             # pre_path = ROOT_DIR/data_sets/set_name
    pre_label_path = os.path.join(args.set_name, args.set_name+'.csv')  # pre_label_path = set_name/set_name.csv
    pre_label_path = os.path.join(config.pre_path, pre_label_path)      # pre_label_path = ROOT_DIR/data_sets/set_name/....?

    if not os.path.exists(pre_path):
        logger.error("ERROR - Unable to find preproc. set: %s" % args.set_name) # if set name doesn't exist log error
        exit(0)                                                                 # and exit
    if not os.path.exists(pre_label_path):                                      # if csv file does not exit log error
        logger.error("ERROR - Unable to find csv file: %s" % args.set_name)     # and exit
        exit(0)

    logger.info("Interpolating Images")                 # interpolations step
    entries = preprocess.interpolate(args.set_name)     # unused?
    logger.info("Image Similarity Detection")           # image similarity detection
    entries = preprocess.similarity_detection(args.set_name, args.sim_cutoff)
    logger.info("Image Augmentation")                   # image augmentation
    augment(args.set_name, args.equalize)
    logger.debug("Process End")

if __name__ == '__main__': # main if
    main()
