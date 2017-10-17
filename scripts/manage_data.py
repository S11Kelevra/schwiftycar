'''
This script manages the merging of data sets
'''

import sys
import os
import re
from os.path import dirname
from os.path import basename
import csv
import argparse
import shutil

import logging
sys.path.append(dirname(dirname(os.path.realpath(__file__))))

import pandas as pd

from robocar42 import util
from robocar42 import config

logger = util.configure_log('data_manage')

def build_parser():
    '''
    returns the parsed data sets and name of output
    '''
    parser = argparse.ArgumentParser()  # set parser
    parser.add_argument(                # parses out set names (set1 + set2 + ...)
        'datasets',
        nargs='+',                      # deliniator '+'
        type=str,
        help='List of datasets to put together')
    parser.add_argument(
        'outset',
        type=str,
        help='Name of output dataset'
        )
    return parser

def merge(argset, outset):
    '''
    '''
    outset_path = os.path.join(config.data_path, outset)        # Outset_path = data_path+outset (data_sets+output name)
    outset_label = os.path.join(outset, outset+'.csv')          # Outset label is joined outset and outset .csv
    outset_label = os.path.join(config.data_path, outset_label) # Outset label is rejoined with data_path and itself
    outset_cam_1 = os.path.join(outset_path, '1')               # paths for cam1 and cam2 are joined with outset_path
    outset_cam_2 = os.path.join(outset_path, '2')
    if not os.path.exists(outset_path):                         # creates the required directories if they don't exist
        os.makedirs(outset_path)                                #  |
    if not os.path.exists(outset_cam_1):                        #  |
        os.makedirs(outset_cam_1)                               #  |
    if not os.path.exists(outset_cam_2):                        #  |
        os.makedirs(outset_cam_2)                               #  \/

    if outset in argset:
        argset.remove(outset)

    with open(outset_label, 'a+') as outset_file:                   # open csv file in append mode as outset file??
        for each_set in argset:                                     # for all data sets in argset
            set_path = os.path.join(config.data_path, each_set)     # set_path = data_sets/set_name
            set_label = os.path.join(each_set, each_set+'.csv')     # set_label = set_name/set_name.csv
            set_label = os.path.join(config.data_path, set_label)   # set_label = data_sets/set_name/set_name.csv
            set_cam_1 = os.path.join(set_path, '1')                 # set_cams = data_sets/set_name/1or2
            set_cam_2 = os.path.join(set_path, '2')

            if not (os.path.exists(set_path) and                    # Error msg if directories don't exist
                os.path.exists(set_cam_1) and
                os.path.exists(set_cam_2) and
                os.path.exists(set_label)):
                logger.error("Dataset: %s is invalid." % each_set)
                continue

            set_cam_1_files = os.listdir(set_cam_1)             # set_cam(1or2) is now a list of the entries in
            set_cam_2_files = os.listdir(set_cam_2)             # data_sets/set_name/1or2
            for file in set_cam_1_files:                        # for all the files in camera 1 folder
                shutil.move(
                    os.path.join(set_cam_1, file),              # move file from data_sets/set_name/1or2
                    os.path.join(outset_cam_1, file))           # to the output folder
            for file in set_cam_2_files:                        # repeat for cam 2
                shutil.move(
                    os.path.join(set_cam_2, file),
                    os.path.join(outset_cam_2, file))
            df = pd.read_csv(set_label,                         # data frame is read into by pandas read
                names=['cam_1', 'cam_2', 'action'], header=0)   # column names  and header
            df.to_csv(outset_file, index=False)                 # data frame then converted to csv file with name ??

def main():
    parser = build_parser()     # Parses arguments
    args = parser.parse_args()  # arguments are set from parser

    if len(args.datasets) >= 2:
        argset = list(set(args.datasets))                   # converts data set from parsed arguments to list
        logger.info("Merging data sets %s." % str(argset))
        merge(argset, args.outset)                          # passes argset(list of data sets) and
    logger.info("Merging complete")                         # outset(name of output) to merge

if __name__ == '__main__':
    main()
