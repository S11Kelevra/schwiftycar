'''
Script that handles training of preprocessed data
'''
import sys
import os
from os.path import dirname
import argparse
import datetime
sys.path.append(dirname(dirname(os.path.realpath(__file__))))
from robocar42 import config, util, models

now = datetime.datetime.now()
display_conf = config.display_parser_config('display.ini')

if __name__ == '__main__':
    print("Starting!")
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument(
        '-data_set',
        type=str,
        help='Name of the training set folder. Default: ts_0',
        default="ts_0"
    )
    parser.add_argument(
        '-model_name',
        type=str,
        default=None,
        help='Path to model h5 file. Model should be on the same path.',
    )
    args = parser.parse_args()
    print("Parsing arguments!")
    print("Set name = " + args.data_set)
    #print("Batch size = " + str(model_conf['batch']))
    data_set = os.path.join(config.data_path , args.data_set)
    if not os.path.exists(data_set):
        print("Data set does not exist")
        exit()
        data_file = os.path.join(data_set, os.path.basename(data_set)) + ".csv"
        if not os.path.exists(data_file):
            print("No file at %s", data_file)
            exit()
    if args.model_name:
        model_name = os.path.join(config.model_path, args.model_name)
        if not os.path.exists(model_name):
            print("H5 file does not exist!")
            exit()
        print("Model name = " + model_name)
        print("Data set =" + data_set)
        models.train(model_name, data_set)
    else:
        print("Making model name")
        model_name = os.path.join(config.model_path, now.strftime("%Y-%m-%d %H:%M") + '.h5')
        print("Model name = " + model_name)
        print("Data set =" + data_set)
        models.train(model_name, data_set, is_new_model=True)
    exit()