import os
from os.path import dirname
import configparser
from CustomException import ValidationError

ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

config = configparser.ConfigParser()

config_path = os.path.join(ROOT_DIR, "config")
data_path = os.path.join(ROOT_DIR, "data_sets")
pre_path = os.path.join(ROOT_DIR, "pre")
model_path = os.path.join(ROOT_DIR, "models")
log_path = os.path.join(ROOT_DIR, "logs")
download_path = os.path.join(ROOT_DIR, "Download") # for staging

# all paths must exist to project to fully work
if (not os.path.exists(config_path) or not os.path.exists(data_path) or
    not os.path.exists(pre_path) or
    not os.path.exists(model_path) or not os.path.exists(log_path)):
    print "Setting up project folders..."
if not os.path.exists(config_path):
    os.makedirs(config_path)
    # raise ValidationError("Director: config path does not exist.")
if not os.path.exists(data_path):
    os.makedirs(data_path)
    # raise ValidationError("Directory: data_sets path does not exist.")
if not os.path.exists(pre_path):
    os.makedirs(pre_path)
    # raise ValidationError("Directory: stream path does not exist.")
if not os.path.exists(model_path):
    os.makedirs(model_path)
    # raise ValidationError("Directory: model path does not exist.")
if not os.path.exists(log_path):
    os.makedirs(log_path)
    # raise ValidationError("Directory: log path does not exist.")

def vehicle_parser_config(config_name):
    '''
    Parser for vehicle configuration file
    '''
    config_file = os.path.join(config_path,config_name)
    config.read(config_file)
    cfg = {}
    vehicle = config['vehicle']
    cfg['car_url'] = vehicle.get('url')
    cfg['straight'] = vehicle.getint('straight')
    cfg['left'] = vehicle.getint('max_left')
    cfg['right'] = vehicle.getint('max_right')
    cfg['exp'] = vehicle.getint('exp')
    cfg['speed'] = vehicle.getint('speed')
    return cfg

def camera_parser_config(config_name):
    '''
    Parser for camera configuration file
    '''
    config_file = os.path.join(config_path, config_name)
    config.read(config_file)
    cfg = {}
    camera = config['camera']
    cfg['cam_url'] = camera.get('url')
    cfg['cam_protocol'] = camera.get('protocol')
    cfg['cam_user'] = camera.get('user')
    cfg['cam_pass'] = camera.get('pass')
    cfg['cam_channel'] = camera.get('channel')
    cfg['cam_fps'] = camera.getint('fps')
    return cfg

def display_parser_config(config_name):
    '''
    Parser for display configuration file
    '''
    config_file = os.path.join(config_path, config_name)
    config.read(config_file)
    cfg = {}
    image = config['image']
    cfg['oshape'] = (image.getint('oshapeX'),
                    image.getint('oshapeY'))
    cfg['sshape'] = (image.getint('sshapeX'),
                    image.getint('sshapeY'))
    cfg['doshape'] = (image.getint('doshapeX'),
                    image.getint('doshapeY'))
    cfg['sdshape'] = (image.getint('sdshapeX'),
                    image.getint('sdshapeY'))
    return cfg

def model_parser_config(config_name):
    '''
    Parser for training model configuration file
    '''
    config_file = os.path.join(config_path, config_name)
    config.read(config_file)
    cfg = {}
    model = config['model']
    shape = map(int, model.get('shape').split(','))
    cfg['shape'] = tuple(shape)
    cfg['epochs'] = model.getint('epochs')
    cfg['steps'] = model.getint('steps')
    cfg['batch'] = model.getint('batch_size')
    cfg['val_split'] = model.getfloat('val_split')
    return cfg

def cloud_parser_config(config_name):
    '''
    Parser for cloud storage configuration file
    '''
    config_file = os.path.join(config_path, config_name)
    config.read(config_file)
    cfg = {}
    cloud = config['cloud']
    cfg['dataset'] = cloud.get('dataset')
    cfg['table'] = cloud.get('table')
    cfg['bucket'] = cloud.get('bucket')
    cfg['folder'] = cloud.get('folder')
    cfg['datastore'] = cloud.get('datastore')
    cfg['datastore_namespace'] = cloud.get('datastore_namespace')
    cfg['datastore_project'] = cloud.get('datastore_project')
    cfg['datastore_entity'] = cloud.get('datastore_entity')
    cfg['tmp'] = cloud.get('tmp')
    cfg['staging'] = cloud.get('staging')
    return cfg

def drive_parser_config(config_name):
    '''
    Parser for drive config
    '''
    config_file = os.path.join(config_path, config_name)
    config.read(config_file)
    cfg = {}
    drive = config['drive']
    cfg['vehicle_conf'] = drive.get('vehicle')
    cfg['display'] = drive.get('display')
    cfg['delay'] = drive.getint('delay')
    cfg['classes'] = drive.getint('classes')
    cfg['level'] = drive.getfloat('level')
    cfg['teach'] = drive.getboolean('teach')
    cameras = drive.get('camera').split(',')
    cfg['camera'] = [cam.strip() for cam in cameras]
    return cfg

