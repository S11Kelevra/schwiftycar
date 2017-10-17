'''This modules describes the preprocess functions'''

import os
import sys
import shutil
import logging
from collections import deque
sys.path.append('..')

import numpy as np
import pandas as pd
import cv2

from robocar42 import config
from robocar42 import util

logger = logging.getLogger('preprocess')

def adjust_gamma(image, gamma=1.0):
    '''
    Adjusts brightness of an image
    '''
    invGamma = 1.0/gamma
    table = np.array([((i/255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def darken(image):
    '''
    Darkens image
    '''
    return adjust_gamma(image, .5)

def brighten(image):
    '''
    Brightens image
    '''
    return adjust_gamma(image, 2.)

def flip(image):
    '''
    Flips image
    '''
    return cv2.flip(image, 1)

def equalize(image):
    '''
    Equalize Image
    '''
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def interpolate(set_name):
    '''
    Increase data set by labeling the in between frames
    '''
    pre_path = os.path.join(config.pre_path, set_name)
    pre_label_path = os.path.join(set_name, set_name+'.csv')
    pre_label_path = os.path.join(config.pre_path, pre_label_path)

    data_path = os.path.join(config.data_path, set_name)
    label_path = os.path.join(set_name, set_name+'.csv')
    label_path = os.path.join(config.data_path, label_path)

    #make new data_path directory if it does not exists
    cam_1_out = os.path.join(data_path, '1')
    cam_2_out = os.path.join(data_path, '2')

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(cam_1_out):
        os.makedirs(cam_1_out)
    if not os.path.exists(cam_2_out):
        os.makedirs(cam_2_out)

    cam_1_path = os.path.join(pre_path, '1')
    cam_1_files = os.listdir(cam_1_path)
    cam_1_files.sort()
    cam_1_files = deque(cam_1_files)
    cam_2_path = os.path.join(pre_path, '2')
    cam_2_files = os.listdir(cam_2_path)
    cam_2_files.sort()
    cam_2_files = deque(cam_2_files)

    df = pd.read_csv(pre_label_path, names=['cam_1', 'cam_2', 'action'])

    entries = []
    ind = 2
    st_ind = 1
    end_ind = -1
    ranges = []
    if ind >= len(df.index):
        ranges.append([st_ind, st_ind])
    else:
        while ind < len(df.index):
            if df['action'].iloc[st_ind] == df['action'].iloc[ind]:
                end_ind = ind
            else:
                if end_ind >= 0:
                    ranges.append([st_ind, end_ind])
                else:
                    ranges.append([st_ind, st_ind])
                st_ind = ind
                end_ind = -1
            if ind == len(df.index) - 1:
                if (end_ind >= 0):
                    ranges.append([st_ind, end_ind])
                else:
                    ranges.append([st_ind, st_ind])
            ind += 1

    for inds in ranges:
        while cam_1_files and df['cam_1'].iloc[inds[0]] != cam_1_files[0]:
            logger.debug("Skipping %s", cam_1_files.popleft())
        while cam_2_files and df['cam_2'].iloc[inds[0]] != cam_2_files[0]:
            logger.debug("Skipping %s", cam_2_files.popleft())
        cam_1_list = deque()
        while cam_1_files:
            if df['cam_1'].iloc[inds[1]] != cam_1_files[0]:
                cam_1_list.append(cam_1_files.popleft())
            if df['cam_1'].iloc[inds[1]] == cam_1_files[0]:
                cam_1_list.append(cam_1_files.popleft())
                break
        cam_2_list = deque()
        while cam_2_files:
            if df['cam_2'].iloc[inds[1]] != cam_2_files[0]:
                cam_2_list.append(cam_2_files.popleft())
            if df['cam_2'].iloc[inds[1]] == cam_2_files[0]:
                cam_2_list.append(cam_2_files.popleft())
                break

        while cam_1_list and cam_2_list:
            cam_1_name = set_name+'_'+cam_1_list[0]
            cam_2_name = set_name+'_'+cam_2_list[0]
            entries.append({
                'cam_1': cam_1_name,
                'cam_2': cam_2_name,
                'action': df['action'].iloc[inds[0]]
            })
            logger.debug("Moving %s -> %s" % (cam_1_list[0], cam_1_name))
            logger.debug("Moving %s -> %s" % (cam_2_list[0], cam_2_name))
            shutil.copy(
                os.path.join(cam_1_path, cam_1_list[0]),
                os.path.join(cam_1_out, cam_1_name))
            shutil.copy(
                os.path.join(cam_2_path, cam_2_list[0]),
                os.path.join(cam_2_out, cam_2_name))
            cam_1_list.popleft()
            cam_2_list.popleft()

    df = pd.DataFrame(entries,
            columns=['cam_1',
                     'cam_2',
                     'action']
         )
    df.to_csv(label_path, index=False)
    return entries

def similarity_detection(set_name, cutoff=0.0):
    data_path = os.path.join(config.data_path, set_name)
    label_path = os.path.join(set_name, set_name+'.csv')
    label_path = os.path.join(config.data_path, label_path)
    #make new data_path directory if it does not exists
    cam_1_path = os.path.join(data_path, '1')
    cam_2_path = os.path.join(data_path, '2')

    prev_img = None
    prev_name = ""
    rat_lst = []
    res_lst = []
    df = pd.read_csv(label_path,
            names=['cam_1', 'cam_2', 'action'], header=0)
    ind = 1
    entries = []
    sample_length = len(df)
    count = 0
    for ind in range(sample_length):
        util.progress_bar(ind, sample_length)
        cam_1 = df['cam_1'].iloc[ind]
        cam_2 = df['cam_2'].iloc[ind]
        img_paths = [
            os.path.join(cam_1_path, cam_1),
            os.path.join(cam_2_path, cam_2)]
        action = df['action'].iloc[ind]
        img_1 = cv2.imread(img_paths[0], 1)
        img_2 = cv2.imread(img_paths[1], 1)
        img = np.concatenate((img_1, img_2), axis=1)
        img = img / 255.
        if type(prev_img) != type(None):
            diff = np.fabs(img - prev_img)
            avg = (img + prev_img) / 2.
            ratio = np.sum(diff) / np.sum(avg)
        else:
            ratio = 1.
        if ratio < cutoff:
            os.remove(img_paths[0])
            os.remove(img_paths[1])
            logger.debug("Image %s is a duplicate of %s" %
                (str([cam_1, cam_2]), str(prev_name)))
            count += 1
            continue
        entries.append([cam_1, cam_2, action])
        prev_img = img
        prev_name = [cam_1, cam_2]
    if cutoff:
         df = pd.DataFrame(entries,
            columns=['cam_1',
                     'cam_2',
                     'action']
         )
         df.to_csv(label_path, index=False)
         logger.info("\nDuplicates detected: %d" % count)
    return entries
