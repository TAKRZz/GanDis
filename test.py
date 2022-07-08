# -*- coding: utf-8 -*-
# @Author  : TAKR-Zz
# @Time    : 2022/6/30 12:55
# @Function:
import random
import time
import joblib
import tensorflow as tf
import numpy as np


def disc_Score(input):
    # print("input ", input)
    input = np.array(input).reshape(1, -1)
    input = normal_census(input)
    # print(input)
    # print("-------------------")
    max_pro = 0.0
    min_pro = 1.0
    input = input[0]
    for i in range(0, 2):
        input[8] = i
        # print(input)
        res = MLPMODEL.predict_proba([input])
        # print(res)
        # print(res)
        if max_pro < res[0][1]:
            max_pro = res[0][1]
        if min_pro > res[0][1]:
            min_pro = res[0][1]
    return max_pro - min_pro + is_disc(input)


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


def census_normal(datasets):
    '''
        [[1, 9], [0, 7], [0, 39], [0, 15], [0, 6], [0, 13], [0, 5], [0, 4], [0, 1], [0, 99], [0, 39], [0, 99], [0, 39]]
        {9: 'sex', 1: 'age', 8: 'race'}
        ['age', 'workclass', 'fnlwgt', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    '''
    newData = []
    for x in datasets:
        newEle = [(x[0] - 5) / 4, (x[1] - 3.5) / 3.5, (x[2] - 19.5) / 19.5, (x[3] - 7.5) / 7.5, (x[4] - 3) / 3,
                  (x[5] - 6.5) / 6.5, (x[6] - 2.5) / 2.5, (x[7] - 2) / 2, (x[8] - 0.5) / 0.5, (x[9] - 49.5) / 49.5,
                  (x[10] - 19.5) / 19.5, (x[11] - 49.5) / 49.5, (x[12] - 19.5) / 19.5]
        newData.append(newEle)
    return np.array(newData)


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


# print(census_normal(discSample))
# print(normal_census(census_normal(discSample)))

start = time.time()

# discSample = np.array(census_normal(discSample))
# print(discSample.shape)
# print(discSample[0])
# print(discSample[1])
end = time.time()
print("Takes :", end - start, "s")

MLPMODEL = joblib.load("model/census/MLP_unfair1.pkl")
print(MLPMODEL)


# BATCH_SIZE = 1


# discSample = tf.expand_dims(discSample, -1)

# datasets = tf.data.Dataset.from_tensor_slices(discSample)
# datasets = datasets.shuffle(40655).batch(BATCH_SIZE)

# print(datasets)


def generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16, input_shape=(4,), use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(32, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(13, use_bias=False, activation='tanh'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((13, 1)))
    return model


def discriminator_model():
    return disc_Score


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(fake_out):
    return disc_Score(fake_out)
    # print("discriminator_loss : ", fake_loss + real_loss)


def generator_loss(gen_sample):
    # print("generator_loss : ", cross_entropy(0.9 * tf.ones_like(fake_out), fake_out))
    # res = disc_Score(gen_sample)
    # res = tf.constant(res, shape=1, dtype=tf.float32)
    # return cross_entropy(tf.constant(1.0, shape=(1, 1), dtype=tf.float32),
    #                      tf.constant(res, shape=(1, 1), dtype=tf.float32))
    # return cross_entropy(2 * tf.ones_like(res), res)

    # ---------------------------------------------------------
    gen_input = np.array(gen_sample).reshape(-1, 13)
    print("-------------------------------")
    print(gen_input.shape)
    print("-------------------------------")

    dis_arr = []
    for gen in gen_input:
        max_pro = 0.0
        min_pro = 1.0

        int1 = 0
        int0 = 0
        for i in range(0, 2):
            gen[8] = i
            res = MLPMODEL.predict_proba([gen])
            # print(res)
            if max_pro < res[0][1]:
                max_pro = res[0][1]
            if min_pro > res[0][1]:
                min_pro = res[0][1]
            if MLPMODEL.predict([gen])[0] == 1:
                int1 += 1
            else:
                int0 += 1
        d = 1
        if int1 * int0 == 0:
            d = 0
        x = max_pro - min_pro + d
        dis_arr.append(x)
        # x = tf.constant(max_pro - min_pro + d, shape=(1), dtype=tf.float32)
    print("dis_arr :", dis_arr, "length : ", len(dis_arr))
    print(2 * tf.ones_like(tf.cast(dis_arr, dtype=tf.float32)))
    print(tf.cast(dis_arr, dtype=tf.float32))
    loss = cross_entropy(2 * tf.ones_like(tf.cast(dis_arr, dtype=tf.float32)), tf.cast(dis_arr, dtype=tf.float32))
    print("loss :", loss)
    # return loss
    # ---------------------------------------------------------
    # start = time.time()
    # a = 3
    # for i in range(1, 100000):
    #     a = a * i
    # print(a)
    # print("takes :", time.time() - start, " s")
    # print("gen_sample: ", gen_sample)
    # gen_sample = gen_sample * gen_sample
    # print("gen_sample * gen_sample: ", gen_sample)
    x = cross_entropy(tf.ones_like(gen_sample), gen_sample)
    # print("x:", x)
    x = tf.math.abs(x - 1)
    # print("abs x:", x)

    # quit()
    # time.sleep(10)
    return x


generator_optimizer = tf.keras.optimizers.Adam(1e-5)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

epochs = 1000
noise_dim = 4
nsamples = 20

tf.random.set_seed(2022)
z = tf.random.normal([nsamples, noise_dim])

generator = generator_model()


# discriminator = disc_Score()


# @tf.function
def train_step():
    noise = tf.random.normal([128, noise_dim])
    # print("noise :", noise)
    with tf.GradientTape() as gen_tape:
        gen_sample = generator(noise, training=True)

        # fake_out = discriminator(gen_sample)
        # gen_sample = np.array(gen_sample).reshape(1, -1)
        # fake_out = disc_Score(gen_sample)
        # print(gen_sample)
        gen_loss = generator_loss(gen_sample)
    # print("variables:", generator.trainable_variables)
    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    print("gen_loss", gen_loss)
    # print("gradient_gen", gradient_gen)
    generator_optimizer.apply_gradients(zip(gradient_gen, generator.trainable_variables))


def generator_sample(gen_model, test_noise):
    pred_image = gen_model(test_noise, training=False)
    pred_image = normal_census(pred_image)
    for i in range(nsamples):
        # print(i, test_noise[i], pred_image[i])
        print(disc_Score(pred_image[i]), end=" ")
        # print(is_disc(pred_image[i]), end=" ")
        # print(pred_image[i])
    print("")
    for i in range(nsamples):
        # print(i, test_noise[i], pred_image[i])
        # print(disc_Score(pred_image[i]), end=" ")
        print(is_disc(pred_image[i]), end=" ")
        # print(pred_image[i])
    print("")


def train(EPOCH):
    for epoch in range(EPOCH):
        train_step()
        print("Epoch {}/{}".format(epoch + 1, EPOCH))
        # generator_sample(generator, z)


train(epochs)

# noise = tf.random.normal([1, noise_dim])
# gen_sample = generator(noise, training=False)
# # fake_out = discriminator(gen_sample)
# print("noise :", noise)
# print("gen_sample", gen_sample)
# # print(fake_out)
# gen_loss = generator_loss(gen_sample)
# print("gen_loss", gen_loss)
# print(tf.ones_like(gen_loss))
#
# print("cross_entropy: ", cross_entropy(tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
#                                        tf.constant(gen_loss, shape=(1, 1), dtype=tf.float64)))
# noise = tf.random.normal([1, noise_dim])
# print(noise)
# gen_sample = generator(noise, training=True)
# print(gen_sample)
# print(np.array(gen_sample).reshape(-1))
# gen_loss = discriminator(gen_sample)
# print(gen_loss)
