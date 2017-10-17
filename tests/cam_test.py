import sys
import os
from os.path import dirname
import unittest
sys.path.append(dirname(dirname(os.path.realpath(__file__))))

from robocar42 import config
from robocar42.camera import Camera

class TestCamera(unittest.TestCase):

    def setUp(self):
        cam_conf = config.camera_parser_config('camera_1.ini')
        disp_conf = config.display_parser_config('display.ini')
        self.cam = Camera('cam', cam_conf, disp_conf['oshape'], True)

    def test_capture(self):
        image = self.cam.get()
        print(image.shape)
        print(self.cam.shape)
        assert image.shape == self.cam.shape

    def test_image(self):
        image = self.cam.get()
        print(type(image))

    def test_stop(self):
        self.cam.stop()
        assert self.cam.stopped == False

if __name__ == '__main__':
    unittest.main()