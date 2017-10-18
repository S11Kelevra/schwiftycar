'''
This module describes the Controller class which handles the driving
functionaliy of the rc car
'''

import os
import sys
import time
import subprocess as sp
from threading import Thread
sys.path.append('..')

import pygame

from robocar42 import config
from robocar42.car import Car

class ControllerCore(object):
    '''
    Core Controller object class
    '''
    def __init__(self, set_name, car, auto=False):
        '''
        :param set_name:
        :param car:
        :param auto:
        '''
        self.set_name = set_name
        self.auto = auto
        self.car = car
        self.pressed = False

    def send_control(self):
        #todo
        '''
        :return:
        '''
        pass

    def drive(self):
        #todo
        '''
        :return:
        '''
        pass

    def is_pressed(self):
        return self.pressed

class RCController(object):     # if this is broken, take out the 't'
    '''
    RC Controller object
    '''
    def __init__(self, set_name, car, auto=False):
        '''
        :param set_name:
        :param car:
        :param auto:
        '''
        super(RCController, self).__init__(set_name, car, auto)

    def send_control(self):
        #todo
        '''
        :return:
        '''
        pass
