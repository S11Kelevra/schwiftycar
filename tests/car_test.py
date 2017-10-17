import unittest
import sys
import os
from os.path import dirname
sys.path.append(dirname(dirname(os.path.realpath(__file__))))

from robocar42 import config
from robocar42.car import Car

class TestCar(unittest.TestCase):

    def setUp(self):
        car_conf = config.vehicle_parser_config('vehicle.ini')
        self.car = Car(car_conf)

    def test_drive_1(self):
        car_conf = config.vehicle_parser_config('vehicle.ini')
        cmd = (
            'http://' +
            str(car_conf['car_url']) +
            '/fwd' +
            '/m' + str(car_conf['speed']) +
            '/lf' + str(75)
        )
        print("Car class output: %s" % self.car.drive([1, 250, -1, 75]))
        print("Hardcoded output: %s" % cmd)
        assert self.car.drive([1, 250, -1, 75]) == cmd

    def test_drive_2(self):
        car_conf = config.vehicle_parser_config('vehicle.ini')
        cmd = (
            'http://' +
            str(car_conf['car_url']) +
            '/rev' +
            '/m' + str(100) +
            '/rt' + str(100)
        )
        print("Car class output: %s" % self.car.drive([-1, 100, 1, 100]))
        print("Hardcoded output: %s" % cmd)
        assert self.car.drive([-1, 100, 1, 100]) == cmd

    def test_drive_3(self):
        car_conf = config.vehicle_parser_config('vehicle.ini')
        cmd = (
            'http://' +
            str(car_conf['car_url']) +
            '/fwd' +
            '/m' + str(car_conf['speed']) +
            '/st' + str(car_conf['straight'])
        )
        print("Car class output: %s" % self.car.drive([1, 250, 0, 100]))
        print("Hardcoded output: %s" % cmd)
        assert self.car.drive([1, 250, 0, 100]) == cmd

    def test_drive_4(self):
        car_conf = config.vehicle_parser_config('vehicle.ini')
        cmd = (
            'http://' +
            str(car_conf['car_url']) +
            '/fwd' +
            '/m' + str(car_conf['speed']) +
            '/lf' + str(100)
        )
        print("Car class output: %s" % self.car.drive([1, 250, -1, 100]))
        print("Hardcoded output: %s" % cmd)
        assert self.car.drive([1, 250, -1, 100]) == cmd

if __name__ == '__main__':
    unittest.main()
