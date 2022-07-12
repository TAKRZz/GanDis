# -*- coding: utf-8 -*-
# @Author   : TAKR-Zz
# @Time     : 2022/7/10 15:22
# @FileName : loss_process


import numpy as np
import matplotlib.pyplot as plt

x_arr = np.arange(1, 20000, 20)
print(x_arr.shape)

# gen_loss = np.load("gen_loss.npy")
# print(gen_loss[0:100])
# new_loss = []
# for i in range(1000):
#     if i % 10 == 0:
#         new_loss.append(gen_loss[i])
# gen_loss = np.array(new_loss)
# print("shape :", gen_loss.shape)
# print("mean  :", gen_loss.mean())
# print("max   :", gen_loss.max())
# print("min   :", gen_loss.min())

# plt.plot(x_arr, gen_loss)
# plt.show()

# dis_loss = np.load("dis_loss.npy")
# print(dis_loss[0:100])
# new_loss = []
# for i in range(1000):
#     if i % 10 == 0:
#         new_loss.append(dis_loss[i])
# dis_loss = np.array(new_loss)
# print("shape :", dis_loss.shape)
# print("mean  :", dis_loss.mean())
# print("max   :", dis_loss.max())
# print("min   :", dis_loss.min())
#
# plt.plot(x_arr, dis_loss)
# plt.show()
#
# noi_loss = np.load("noi_loss.npy")
# new_loss = []
# for i in range(1000):
#     if i % 10 == 0:
#         new_loss.append(noi_loss[i])
# noi_loss = np.array(new_loss)
# print("shape :", noi_loss.shape)
# print("mean  :", noi_loss.mean())
# print("max   :", noi_loss.max())
# print("min   :", noi_loss.min())
# plt.plot(x_arr, noi_loss)
# plt.show()
