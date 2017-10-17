'''
This module describes the Car class, which contains the information required
to move the car.
'''

import urllib2

class Car(object):
    '''
    RC Car object class. Contains car wifi url, the important steering angles
    (straight, left, right), speed settings and command expiration time
    '''

    #list of allowed drive commands
    ops = {
        'direct': {-1: '/rev', 0: '', 1: '/fwd'},
        'steer': {-1: '/lf', 0: '/st', 1: '/rt'}
    }

    def __init__(self, config):
        self.url = config['car_url']
        self.straight = config['straight']
        self.left = config['left']
        self.right = config['right']
        self.max_speed = config['speed']
        self.exp = config['exp']

    def drive(self, req):
        '''
        Generates and executes a drive command based on the request. Direction
        and steer must be specified as integers between -1 and 1. Refer to
        attribute "ops" for more information.
        '''
        direct = Car.ops['direct'][req[0]]
        spd = '/m'
        if req[1] < 0:
            spd += str(0)
        elif req[1] > self.max_speed:
            spd += str(self.max_speed)
        else:
            spd += str(req[1])
        steer = Car.ops['steer'][req[2]]
        if req[2] == 0:
            angle = str(self.straight)
        elif (req[3] < min(self.left, self.right) or
              req[3] > max(self.left, self.right)):
            if req[2] == -1:
                angle = str(self.left)
            elif req[2] == 1:
                angle = str(self.right)
        else:
            angle = str(req[3])
        cmd = 'http://' + self.url + direct + spd + steer + angle + '/exp' + str(self.exp)
        try:
            urllib2.urlopen(cmd, timeout=2)
            return cmd
        except urllib2.URLError:
            return None

    def start(self):
        '''
        Pings the car to verify that it works
        '''
        if self.drive([0, 0, 0, 0]):
            return True
        return False
