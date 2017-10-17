'''
Checks staging folder for new datasets.
Downloads, unpacks, and adds them to processor queue
'''

import sys
import os
from os.path import dirname
from os.path import basename
import time
import zipfile
import signal
sys.path.append(dirname(dirname(os.path.realpath(__file__))))

from google.cloud import storage
from google.cloud import datastore

from robocar42 import util
from robocar42.util import progress_bar
from robocar42 import config

logger = util.configure_log('staging')

def get_staging():
    '''
    Gets the list of files under the staging folder
    '''
    cloud_conf = config.cloud_parser_config('cloud.ini')
    client = storage.Client()
    bucket = client.get_bucket(cloud_conf['bucket'])
    folder = cloud_conf['folder']+'/'
    blobs = list(bucket.list_blobs(prefix=folder, delimiter='/'))
    return blobs

def parse_meta_file(meta_file):
    ret_dict = {}
    with open(meta_file, 'r') as meta:
        for line in meta:
            key, value = tuple([pair.strip() for pair in line.split('=')])
            ret_dict[key] = value
    return ret_dict

def update_process_queue(meta_file, set_name):
    '''
    Updates dataset processing queue
    '''
    logger.info("Adding %s onto preprocess queue" % set_name)
    cloud_conf = config.cloud_parser_config('cloud.ini')
    key_val_pairs = parse_meta_file(meta_file)
    client = datastore.Client(
                namespace=cloud_conf['datastore_namespace'],
                project=cloud_conf['datastore_project'])
    ds_key = client.key('preprocess')
    ds_entity = datastore.Entity(ds_key)

    ds_query = client.query(kind='preprocess')
    ds_query.add_filter('set_name', '=', set_name)
    if not list(ds_query.fetch()):
        ds_entity['set_name'] = set_name
        for key in key_val_pairs:
            ds_entity[key] = key_val_pairs[key]
        client.put(ds_entity)
    else:
        logger.warning("Warning - %s entry already exists." % set_name)

def download_and_unpack(blob, filename):
    '''
    Downloads files from the staging folder, unpacks them, then removes
    original files from the staging directory. Adds files onto a database
    for preprocessing step to work with
    '''
    logger.info("Downloading %s..." % filename)
    local_filename = os.path.join(config.download_path, filename)
    blob.download_to_filename(local_filename)

    # Completed download
    meta_file = ''
    with zipfile.ZipFile(local_filename) as zp:
        logger.info("-----Unpacking %s-----" % filename)
        total = len(zp.namelist())
        count = 0
        progress_bar(count, total)
        for name in zp.namelist():
            if '.bmp' in name:
                zp.extract(name, config.data_path)
            elif '.meta' in name:
                zp.extract(name, config.data_path)
                meta_file = os.path.join(config.data_path, name)
            elif '.csv' in name:
                zp.extract(name, config.data_path)
            count += 1
            progress_bar(count, total)

    if meta_file:
        # add onto process queue
        no_ext = os.path.splitext(filename)[0]
        update_process_queue(meta_file, no_ext)
    else:
        logger.warning("Warning - %s meta file does not exist." % filename)

    #After extraction, delete from bucket and machine intermediary files
    if blob.exists():
        try:
            blob.delete()
        except google.cloud.exceptions.NotFound as err:
            pass

    if os.path.exists(local_filename):
        try:
            os.remove(local_filename)
        except OSError as err:
            pass

def interrupt_handler(signum, frame):
    global interrupted
    interrupted = True
    logger.info("Exiting program...")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, interrupt_handler)
    if not os.path.exists(config.download_path):
        os.makedirs(config.download_path)
    interrupted = False
    while not interrupted:
        blobs = get_staging()
        for blob in blobs:
            bname = basename(blob.name)
            if bname:
                # check if file already exists
                no_ext = os.path.splitext(bname)[0]
                ext_path = os.path.join(config.data_path, no_ext)
                if not os.path.exists(ext_path):
                    download_and_unpack(blob, bname)
                    logger.info("Queued: %s" % bname)
        #Sleep for 10 seconds after each iteration
        time.sleep(10)
    logger.info("Program Terminated")
