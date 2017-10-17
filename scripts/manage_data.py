'''
This script manages
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'datasets',
        nargs='+',
        type=str,
        help='List of datasets to put together')
    parser.add_argument(
        'outset',
        type=str,
        help='Name of output dataset'
        )
    return parser

def merge(argset, outset):

    outset_path = os.path.join(config.data_path, outset)
    outset_label = os.path.join(outset, outset+'.csv')
    outset_label = os.path.join(config.data_path, outset_label)
    outset_cam_1 = os.path.join(outset_path, '1')
    outset_cam_2 = os.path.join(outset_path, '2')
    if not os.path.exists(outset_path):
        os.makedirs(outset_path)
    if not os.path.exists(outset_cam_1):
        os.makedirs(outset_cam_1)
    if not os.path.exists(outset_cam_2):
        os.makedirs(outset_cam_2)

    if outset in argset:
        argset.remove(outset)

    with open(outset_label, 'a+') as outset_file:
        for each_set in argset:
            set_path = os.path.join(config.data_path, each_set)
            set_label = os.path.join(each_set, each_set+'.csv')
            set_label = os.path.join(config.data_path, set_label)
            set_cam_1 = os.path.join(set_path, '1')
            set_cam_2 = os.path.join(set_path, '2')

            if not (os.path.exists(set_path) and
                os.path.exists(set_cam_1) and
                os.path.exists(set_cam_2) and
                os.path.exists(set_label)):
                logger.error("Dataset: %s is invalid." % each_set)
                continue

            set_cam_1_files = os.listdir(set_cam_1)
            set_cam_2_files = os.listdir(set_cam_2)
            for file in set_cam_1_files:
                shutil.move(
                    os.path.join(set_cam_1, file),
                    os.path.join(outset_cam_1, file))
            for file in set_cam_2_files:
                shutil.move(
                    os.path.join(set_cam_2, file),
                    os.path.join(outset_cam_2, file))
            df = pd.read_csv(set_label,
                names=['cam_1', 'cam_2', 'action'], header=0)
            df.to_csv(outset_file, index=False)

def main():
    parser = build_parser()
    args = parser.parse_args()

    if len(args.datasets) >= 2:
        argset = list(set(args.datasets))
        logger.info("Merging data sets %s." % str(argset))
        merge(argset, args.outset)
    logger.info("Merging complete")

if __name__ == '__main__':
    main()
