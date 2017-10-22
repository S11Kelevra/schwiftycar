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
        '--cam_num',
        type = str,
        help = 'Path to camera folder from project directory, must be model_name/1 or model_name/2',
        default=None,
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        default=None,
        help='Path of csv file to use',
    )
    #parser.add_argument(
    #    '-data_set',
    #    type=str,
    #    help='Name of the training set folder. Default: ts_0',
    #    default="ts_0"
    #)
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Model h5 file or name of new output. Model should be on the same path.',
    )
    args = parser.parse_args()
    print("Parsing arguments!")
    print("Image folder = " + args.cam_num)
    cam_num = os.path.join(config.data_path, args.cam_num)
    if not '1' <= os.path.basename(args.cam_num) <= '2':
        print("Entered cam_num: " + os.path.basename(args.cam_num))
        print("camera_number must be named 1 or 2 to read properly from csv")
        exit()
    if not os.path.exists(cam_num):
        print("Image folder does not exist at :" + cam_num)
        exit()
    csv_file = os.path.join(config.data_path, args.csv_file)
    if not os.path.exists(csv_file):
        print("No csv file at %s", csv_file)
        exit()
    model_name = os.path.join(config.model_path, args.model_name) + '_camera_' + os.path.basename(args.cam_num) + '.h5'
    if not os.path.exists(model_name):
        print("Making model name")
        print("Model name = " + model_name)
        print("Image folder =" + cam_num)
        models.train(csv_file, model_name, cam_num, is_new_model=True)
    else:
        model_name = os.path.join(config.model_path, args.model_name)
        #if not os.path.exists(model_name):
            #print("H5 file does not exist!")
            #exit()
        print("Model name = " + model_name)
        print("Data set =" + cam_num)
        models.train(model_name, cam_num)
    exit()