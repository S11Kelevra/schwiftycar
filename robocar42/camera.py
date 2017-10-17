'''This module describes the Camera class which captures the camera feed.'''

import os
import sys
import time
import subprocess as sp
from threading import Thread
sys.path.append('..')

import numpy as np

from robocar42 import config

class CameraCore(object):
    '''
    Core Camera object class
    '''
    def __init__(self, name, res=(320, 240)):
        self.name = name
        self.image = np.zeros((res[1], res[0], 3))
        self.shape = (res[0], res[1], 3)
        self.cam_thread = None

    def get(self):
        '''
        Gets frame from camera
        '''
        return self.image

    def stop(self):
        '''
        Stops camera
        '''
        self.cam_thread.join()

    def update(self):
        '''
        Updates frame
        '''
        pass

    def start(self):
        '''
        Starts the camera background thread for capturing frames
        '''
        self.cam_thread = Thread(target=self.update)
        # self.cam_thread.daemon = True
        self.cam_thread.start()
        time.sleep(1)
        return self


class Camera(CameraCore):
    '''
    Camera object class.
    '''
    def __init__(self, name, conf, res=(320, 240), log=False):
        super(Camera, self).__init__(name, res)
        log_name = '%s/%s.log' % (config.log_path, self.name)
        if log:
            self._stderr = open(log_name, 'w')
        else:
            self._stderr = open('/dev/null', 'w')
        _r, _w = os.pipe()
        self._r = os.fdopen(_r)
        self._w = os.fdopen(_w, 'w')
        self.image = None
        self.stopped = False

        #setting up camera
        input_url = ("%s://%s:%s@%s/%s" % (
            conf['cam_protocol'],
            conf['cam_user'],
            conf['cam_pass'],
            conf['cam_url'],
            conf['cam_channel']))
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_url,
            # '-fflags', 'nobuffer',
            # '-preset', 'ultrafast',
            '-vf', 'fps=15',
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo',
            'pipe:1'
        ]
        self.ffmpeg = sp.Popen(
            ffmpeg_cmd,
            stdout=self._w,
            stderr=self._stderr,
            bufsize=10**8
        )
        time.sleep(3)
        self.recording = True
        self.start()

    def stop(self):
        '''
        Closes file descriptor and terminates the ffmpeg program
        '''
        self._w.close()
        self._stderr.close()
        if self.ffmpeg:
            self.ffmpeg.kill()
        self.recording = False
        self.cam_thread.join()

    def update(self):
        '''
        Captures the current frame of the camera and save the current frame
        '''
        while True:
            raw_image = self._r.read(
                self.shape[0]
                * self.shape[1]
                * self.shape[2]
            )
            if self.recording:
                new_image = np.fromstring(raw_image, dtype='uint8')
                new_image = new_image.reshape((self.shape[1],
                                               self.shape[0],
                                               self.shape[2]))
                new_image = np.swapaxes(new_image, 0, 1)
                self.image = new_image
            else:
                self._r.close()
                self.stopped = True
                break
