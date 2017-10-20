'''
Script that handles training of preprocessed data
'''
import sys
import os
from os.path import dirname
import argparse
sys.path.append(dirname(dirname(os.path.realpath(__file__))))
from robocar42 import config, util, models


model_conf = config.model_parser_config('model_1.ini')
display_conf = config.display_parser_config('display.ini')

if __name__ == '__main__':
    print("Starting!")
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument(
        '-set_name',
        type=str,
        help='Name of the training set folder. Default: ts_0',
        default="ts_0"
    )
    parser.add_argument(
        '-model',
        type=str,
        default='',
        help='Path to model h5 file. Model should be on the same path.'
    )

    args = parser.parse_args()
    print("Parsing arguments!")
    print("Set name = " + args.set_name)
    print("Batch size = ", model_conf['batch'])
    data_set = os.path.join(config.data_path , args.set_name)
    if not os.path.exists(data_set):
        print("Data set does not exist")
        exit()
    model_name = os.path.join(config.model_path , args.model)
    print("Model name = " + str(model_name))

    print("Data set =" + data_set)
    models.train(model_conf, model_name, data_set)
    exit()