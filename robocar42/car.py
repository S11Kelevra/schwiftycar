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
        if req[1] < 0:                  # spd is determined by req[1].
            spd += str(0)               # if req[1] is less than 0, add 0
        elif req[1] > self.max_speed:   # if req[1] is greater than max speed, add max
            spd += str(self.max_speed)
        else:                           # if it's a valid value, add the value
            spd += str(req[1])
        steer = Car.ops['steer'][req[2]]    # sets the steer value (/lf, /st /rt)
        if req[2] == 0:
            angle = str(self.straight)
        elif (req[3] < min(self.left, self.right) or
              req[3] > max(self.left, self.right)):     # determines the steer value
            if req[2] == -1:                            # angles left
                angle = str(self.left)
            elif req[2] == 1:                           # angles right
                angle = str(self.right)
        else:                                           # if req[2] is not -1, 0, or 1...
            angle = str(req[3])                         # angle is set to req[3]
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
        if self.drive([0, 0, 0, 0]):    # sends 0's in a list to drive.
            return True
        return False
