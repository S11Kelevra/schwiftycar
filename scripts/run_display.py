'''
This script runs the display module for capturing/viewing test
'''
import sys
import os
from os.path import dirname
from os.path import basename
import time
import signal
import argparse
import atexit
sys.path.append(dirname(dirname(os.path.realpath(__file__))))

import pygame

from robocar42.display import Display
from robocar42 import config


def ctrl_c_handler(signum, frame):
    pygame.quit()
    exit(0)

def build_parser():
    '''
    Build Parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-save',
        action='store_true',
        default=False,
        help='Enable recording from camera. Default: False'
        )
    return parser

def cleanup(display):
    display.stop()
    exit(0)

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()


    signal.signal(signal.SIGINT, ctrl_c_handler)
    disp_conf = config.display_parser_config('display.ini')
    disp = Display('main', disp_conf, ['camera_1.ini', 'camera_2.ini'])
    atexit.register(cleanup, disp)
    pygame.init()
    screen = pygame.display.set_mode(disp_conf['doshape'])

    if args.save:
        rec_folder = "rec_%s" % time.strftime("%d_%m_%Y_%H_%M_%S")
        rec_folder = os.path.join(config.pre_path, rec_folder)
        rec_dirs = [rec_folder+'/'+str(i) for i in range(2)]
        for directory in rec_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
    else:
        rec_dirs = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        surface, images, filenames = disp.show(args.save, rec_dirs)
        print images[0].shape
        screen.blit(surface[0], (0,0))
        screen.blit(surface[1], (disp_conf['oshape'][0],0))
        pygame.display.flip()
    disp.stop()
    pygame.quit()
