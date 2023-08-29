# -*- coding: utf-8 -*-
from pathlib import Path

import h5py
import joblib
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
seed_value = 1234567890
random.seed(seed_value)

def flip_with_probability(value, probability):
    if random.random() < probability:
        return 1 - value
    else:
        return value


def pack(dir_stft, binary=False, probability_of_flip=0.01):
    feature_ori_list = []
    feature_ck_list = []
    feature_wh_list = []
    feature_res_list = []
    label_list = []
    for file in tqdm(os.listdir(dir_stft)):
        I_stft = Image.open(dir_stft + file).convert('L')
        I_stft = np.array(I_stft)
        feature_ori_list.append(I_stft)
        I_stft = Image.open(dir_stft + '../ck/' + file).convert('L')
        feature_ck_list.append(np.array(I_stft))
        I_stft = Image.open(dir_stft + '../wh/' + file).convert('L')
        feature_wh_list.append(np.array(I_stft))
        I_stft = Image.open(dir_stft + '../res/' + file).convert('L')
        feature_res_list.append(np.array(I_stft))
        txt_dir = '../data/official/train/'

        num = int(file.split('.')[0][22:])
        txt_name = file[:22] + '.txt'
        try:
            array = np.loadtxt(txt_dir + txt_name)
        except:
            # print("Skipped", file)
            continue
        try:
            crackles = array[:, 2:4][int(num), 0]
            wheezes = array[:, 2:4][int(num), 1]
        except:
            # print("Skipped", file)
            continue
        if binary:
            if crackles == 0 and wheezes == 0:
                label = flip_with_probability(0, probability_of_flip)
            else:
                label = flip_with_probability(1, probability_of_flip)
        else:
            if crackles == 0 and wheezes == 0:
                label = 0
            elif crackles == 1 and wheezes == 0:
                label = 1
            elif crackles == 0 and wheezes == 1:
                label = 2
            else:
                label = 3
        label_list.append(label)
    return feature_ori_list, feature_ck_list, feature_wh_list, feature_res_list, label_list


def one_hot(x, K):
    # x is a array from np
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


if __name__ == '__main__':
    probability_of_flip = 0.01
    folder_name = f'binary_{probability_of_flip * 100}_percent_noise'
    Path(f'./pack/{folder_name}').mkdir(parents=True, exist_ok=True)

    com_path = './analysis/tqwt_cycles/train/'
    ori, ck, wh, res, label = pack(com_path + 'ori/', binary=True, probability_of_flip=probability_of_flip)
    joblib.dump((ori, ck, wh, res, label), open(f'./pack/{folder_name}/tqwt1_4_train.p', 'wb'))
    print('Done! train')

    com_path = './analysis/tqwt_cycles/test/'
    ori, ck, wh, res, label = pack(com_path + 'ori/', binary=True, probability_of_flip=probability_of_flip)
    joblib.dump((ori, ck, wh, res, label), open(f'./pack/{folder_name}/tqwt1_4_test.p', 'wb'))

    print('Done! test')
