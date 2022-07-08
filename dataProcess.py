# -*- coding: utf-8 -*-
# @Author  : TAKR-Zz
# @Time    : 2022/6/28 15:43
# @Function:

import joblib
import numpy
import numpy as np
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
# MLPMODEL = joblib.load("model/bank/MLP_unfair.pkl")
MLPMODEL = joblib.load("model/census/MLP_unfair1.pkl")

data = numpy.load("datasets/census/local_samples_MLP_expga3.npy")

count = 0
da = np.load("datasets/census/new_data1.npy")
print(da.shape)

def is_disc(sample):
    int1 = 0
    int0 = 0
    for i in range(1, 10):
        sample[8] = i
        if MLPMODEL.predict([sample])[0] == 1:
            int1 += 1
        else:
            int0 += 1
    # print(int0, int1)
    if int1 * int0 != 0:
        return 1
    return 0


def disc_Score(input):
    max_pro = 0.0
    min_pro = 1.0
    for ind in range(1, 10):
        input[8] = ind
        # print(input)
        resu = MLPMODEL.predict_proba([input])
        # print(res)
        if max_pro < resu[0][1]:
            max_pro = resu[0][1]
        if min_pro > resu[0][1]:
            min_pro = resu[0][1]
    return max_pro - min_pro


# score = []
# sumScore = 0.0
# print(data.shape)
# res = []
# for d in data:
#     i = is_disc(d)
#     if i == 1:
#         count += 1
#         res.append(d)
#         a = disc_Score(d)
#         score.append(a)
#         print(a)
#         sumScore += a
#         print("SumScore :", sumScore, "Count :", count)
#     else:
#         print("No Dis")
#
# print("Avg Score : ", sumScore / count)
# avgScore = sumScore / count
#
# newData = []
#
# for x in data:
#     if disc_Score(x) >= 0.10:
#         newData.append(x)
# newData = np.array(newData)
#
# np.save("datasets/census/new_data1.npy", newData)


a = tf.constant([1])
print(np.array(a)[0])
