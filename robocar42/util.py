'''This modules describes the utility functions for the project'''

from __future__ import print_function
import os
import sys
import time
import logging
sys.path.append('..')

from robocar42 import config

def progress_bar(iteration, total):
    percent = ("{0:.2f}").format(100 * (iteration / float(total)))
    filledLength = int(50 * iteration // total)
    bar = '*' * filledLength + '-' * (50 - filledLength)
    print ('\rProgress |%s| %s%% Complete' % (bar, percent), end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def configure_log(name):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(config.log_path+'/'+name+'.log')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
