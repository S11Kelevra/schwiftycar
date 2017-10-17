'''
This script runs the camera capturing process for viewing test.
'''

import sys
import os
from os.path import dirname
from os.path import basename
import time
import argparse
import atexit
import signal
import logging
sys.path.append(dirname(dirname(os.path.realpath(__file__))))

import numpy as np
import pygame

from robocar42 import config
from robocar42.camera import Camera

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

def ctrl_c_handler(signum, frame):
    pygame.quit()
    exit(0)

def cleanup(cams):
    for cam in cams:
        cam.stop()
    exit(0)

def main():
    '''
    Main
    '''
    signal.signal(signal.SIGINT, ctrl_c_handler)
    parser = build_parser()
    args = parser.parse_args()

    if args.save:
        rec_folder = "rec_%s" % time.strftime("%d_%m_%Y_%H_%M_")
        rec_folder = os.path.join(config.pre_path, rec_folder)
        rec_dirs = [rec_folder+'/'+str(i) for i in range(2)]
        for directory in rec_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)

    cam_1_conf = config.camera_parser_config('camera_1.ini')
    cam_2_conf = config.camera_parser_config('camera_2.ini')
    disp_conf = config.display_parser_config('display.ini')
    cam_1 = Camera('cam_1', cam_1_conf)
    cam_2 = Camera('cam_2', cam_2_conf)
    atexit.register(cleanup, [cam_1, cam_2])
    pygame.init()
    screen = pygame.display.set_mode(disp_conf['doshape'])
    surface = [pygame.Surface(disp_conf['oshape']) for i in range(2)]

    running = True
    img_ind = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        image_1 = cam_1.get()
        image_2 = cam_2.get()
        pygame.pixelcopy.array_to_surface(surface[0], image_1)
        pygame.pixelcopy.array_to_surface(surface[1], image_2)
        screen.blit(surface[0], (0,0))
        screen.blit(surface[1], (disp_conf['oshape'][0],0))
        pygame.display.flip()
        if args.save:
            file_1_name = 'img_%05d.bmp' % img_ind
            file_2_name = 'img_%05d.bmp' % img_ind
            file_1_name = os.path.join(rec_dirs[0], file_1_name)
            file_2_name = os.path.join(rec_dirs[1], file_2_name)
            pygame.image.save(surface[0], file_1_name)
            pygame.image.save(surface[1], file_2_name)
            img_ind += 1
    cam_1.stop()
    cam_2.stop()
    pygame.quit()

if __name__ == '__main__':
    main()
