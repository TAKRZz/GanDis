# -*- coding: utf-8 -*-
# @Author   : TAKR-Zz
# @Time     : 2022/7/11 21:11
# @FileName : randomSampleGenerate
import time

import numpy as np
import joblib

np.random.seed(2022)


def normal_census(datasets):
    '''
        [[1, 9], [0, 7], [0, 39], [0, 15], [0, 6], [0, 13], [0, 5], [0, 4], [0, 1], [0, 99], [0, 39], [0, 99], [0, 39]]
        {9: 'sex', 1: 'age', 8: 'race'}
        ['age', 'workclass', 'fnlwgt', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    '''
    newData = []
    for x in datasets:
        newEle = [int(x[0] * 4 + 5), int(x[1] * 3.5 + 3.5), int(x[2] * 19.5 + 19.5), int(x[3] * 7.5 + 7.5),
                  int(x[4] * 3 + 3), int(x[5] * 6.5 + 6.5), int(x[6] * 2.5 + 2.5), int(x[7] * 2 + 2),
                  int(x[8] * 0.5 + 0.5), int(x[9] * 49.5 + 49.5), int(x[10] * 19.5 + 19.5), int(x[11] * 49.5 + 49.5),
                  int(x[12] * 19.5 + 19.5)]
        newData.append(newEle)
    return np.array(newData)


MLPMODEL = joblib.load("./model/census/MLP_unfair1.pkl")

MAX_SAMPLE = 100000
DISC_SET = set()
DISC_SET_LIST = []


# REAL_SAMPLE = 0


def is_disc(sample):
    int1 = 0
    int0 = 0
    for i in range(0, 2):
        sample[8] = i
        if MLPMODEL.predict([sample])[0] == 1:
            int1 += 1
        else:
            int0 += 1
    # print(int0, int1)
    if int1 * int0 != 0:
        return 1
    return 0


# sample_arr = normal_census(np.random.random((10000, 13)))
# res = MLPMODEL.predict(sample_arr)

# print(np.array(res).shape)

start = time.time()
while len(DISC_SET_LIST) < MAX_SAMPLE:
    sample_arr = normal_census(np.random.random((10000, 13)))
    s0 = []
    s1 = []
    for s in sample_arr:
        s[8] = 0
        s0.append(list(s))
        s[8] = 1
        s1.append(list(s))
    s0 = np.array(s0, dtype='float32')
    s1 = np.array(s1, dtype='float32')
    r0 = MLPMODEL.predict(s0)
    r1 = MLPMODEL.predict(s1)
    r = r0 + r1
    for i in range(r.shape[0]):
        if r[i] == 1:
            # print(i
            DISC_SET_LIST.append(sample_arr[i])

    # np.save()
    # print("--------------------------------")
    # quit()

    # print(s0[0])
    # print(s1[0])
    # quit()
    print("Takes :", time.time() - start, "s  Length ：", len(DISC_SET_LIST))
    # print(sample_arr.shape)
    if len(DISC_SET_LIST) % 5000 <= 1:
        np.save("datasets/census/random_sample{}.npy".format(len(DISC_SET_LIST)),
                np.array(DISC_SET_LIST, dtype='float32'))

# print(len(DISC_SET_LIST))
