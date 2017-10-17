'''This module describes the Display class which camera frames.'''

import os
import sys
import time
from threading import Thread
sys.path.append('..')

import pygame

from robocar42 import config, camera
from robocar42.CustomException import DisplayError

class Display(object):
    '''
    Display object class
    '''
    def __init__(self, name, disp_conf, cam_confs, cam_num=2):
        '''
        Init function for Display
        '''
        if cam_num < 0 or not cam_confs:
            raise DisplayError("Display object has no cameras to instantiate.")
        self.cam_list = []
        for i in range(cam_num):
            cam_conf = config.camera_parser_config(cam_confs[i])
            cam = camera.Camera('cam_'+str(i+1), cam_conf)
            self.cam_list.append(cam)
        self.frame_counter = 0
        self.disp_conf = disp_conf
        self.sv_thread_list = []
        self.capturing = True
        t = Thread(target=self._check_threads)
        t.daemon = True
        t.start()

    def stop(self):
        '''
        Stop function for display. Safely closes all threads and process from
        used by Camera objects and Display
        '''
        stop_threads = []
        for cam in self.cam_list:
            t = Thread(target=cam.stop)
            stop_threads.append(t)
            t.start()
        for t in stop_threads:
            t.join()
        self.capturing = False
        for thread in self.sv_thread_list:
            thread.join()

    def _check_threads(self):
        '''
        Regularly go through thread list and remove completed threads
        '''
        if self.capturing:
            self.sv_thread_list = [t for t in self.sv_thread_list if t.isAlive()]

    def show(self, save=False, record_dirs=None):
        '''
        Display current frames onto display.
        If save is set and record dirs are given, display will save the frames
        of each camera to their corresponding directories
        Otherwise, display will just show the frames only
        '''
        if save and record_dirs:
            return self.record(record_dirs)
        else:
            return self.view_only()

    def view_only(self):
        '''
        Show current frames only:
        returns pygame frames for display, numpy array for inference
        '''
        ret_frames = []
        ret_images = []
        for cam in self.cam_list:
            image = cam.get()
            ret_images.append(image)
            surface = pygame.Surface(self.disp_conf['oshape'])
            pygame.pixelcopy.array_to_surface(surface, image) 
            ret_frames.append(surface)
        return ret_frames, ret_images, None

    def record(self, record_dirs):
        '''
        Record current frames into file
        returns pygame frames for display, numpy array for inference, and
        filenames for csv entry
        '''
        if len(record_dirs) != len(self.cam_list):
            raise DisplayError("Number of directories given does"
                               "not match number of cameras")
        ret_frames = []
        ret_images = []
        ret_names = []
        self.frame_counter += 1
        for ind in range(len(self.cam_list)):
            filename = 'img_%05d.bmp' % self.frame_counter
            ret_names.append(filename)
            filename = os.path.join(record_dirs[ind], filename)
            image = self.cam_list[ind].get()
            ret_images.append(image)
            surface = pygame.Surface(self.disp_conf['oshape'])
            pygame.pixelcopy.array_to_surface(surface, image)
            ret_frames.append(surface)
            save_thread = Thread(target=self.save, args=(filename, surface))
            self.sv_thread_list.append(save_thread)
            save_thread.start()
        return ret_frames, ret_images, ret_names

    def save(self, filename, image):
        '''
        I/O operation of saving images, threaded
        '''
        pygame.image.save(image, filename)
