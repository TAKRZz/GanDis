# -*- coding: utf-8 -*-
# @Author  : TAKR-Zz
# @Time    : 2022/6/29 11:30
# @Function:

import time
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# discSample = np.load("local_samples_MLP_census_expga.npy")
# discSample = np.load("datasets/census/new_data1.npy")
# discSample = np.load("new_local_samples.npy")
discSample = np.load("datasets/census/random_sample60000.npy")
print(discSample.shape)
discSample = tf.cast(discSample, tf.float32)
discSample = np.array(discSample)
# discSample = np.cast(discSample, np.float)
print(discSample[0])
print(discSample[-1])


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

    # mean = np.mean(datasets)

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
tf.random.set_seed(2022)

discSample = np.array(census_normal(discSample))
print(discSample.shape)
# print(discSample[0])
# print(discSample[1])
end = time.time()
print("Takes :", end - start, "s")

MLPMODEL = joblib.load("model/census/MLP_unfair1.pkl")
print(MLPMODEL)

BATCH_SIZE = 128

discSample = tf.expand_dims(discSample, -1)

datasets = tf.data.Dataset.from_tensor_slices(discSample)
datasets = datasets.shuffle(60000).batch(BATCH_SIZE)

print(datasets)


def generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, input_shape=(2,), use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(6, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(8, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(10, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(13, use_bias=False, activation='tanh'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((13, 1)))
    return model


def discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(13, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(10, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(8, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(4, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(2, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(1))
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_out, fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    # print("discriminator_loss : ", fake_loss + real_loss)
    return real_loss + fake_loss


def generator_loss(fake_out):
    # print("generator_loss : ", cross_entropy(0.9 * tf.ones_like(fake_out), fake_out))
    return cross_entropy(tf.ones_like(fake_out), fake_out)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

epochs = 5000
noise_dim = 2

nsamples = 100

z = tf.random.normal([nsamples, noise_dim])

generator = generator_model()
discriminator = discriminator_model()

gen_loss_arr = []
dis_loss_arr = []
noi_loss_arr = []

F_GEN = open("loss/loss_new/gen_loss.txt", "a")
F_DIS = open("loss/loss_new/dis_loss.txt", "a")
F_NOI = open("loss/loss_new/noi_loss.txt", "a")


@tf.function
def train_step(dataInput):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_sample = generator(noise, training=True)

        real_out = discriminator(dataInput, training=True)
        fake_out = discriminator(gen_sample, training=True)

        gen_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)

    # gen_loss_arr.append(gen_loss)
    # print("gen_loss :", gen_loss)
    # gen_loss_arr.append(gen_loss)
    # np.append(gen_loss_arr, t.eval(), axis=0)
    # tf.print("gen_loss :", gen_loss)

    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))
    return gen_loss, disc_loss


def disc_Score(input):
    max_pro = 0.0
    min_pro = 1.0
    for i in range(0, 2):
        input[8] = i
        # print(input)
        res = MLPMODEL.predict_proba([input])
        # print(res)
        if max_pro < res[0][1]:
            max_pro = res[0][1]
        if min_pro > res[0][1]:
            min_pro = res[0][1]
    return max_pro - min_pro


def is_disc(sample):
    int1 = 0
    int0 = 0
    for i in range(0, 2):
        sample[8] = i
        # print("sample :", sample, "predict :", MLPMODEL.predict([sample]))
        if MLPMODEL.predict([sample])[0] == 1:
            int1 += 1
        else:
            int0 += 1
    # print(int0, int1)
    if int1 * int0 != 0:
        return 1
    return 0


def generator_sample(gen_model, dis_model, gen_loss, test_noise, epoch):
    pred_sample = gen_model(test_noise, training=False)

    num = 0

    if epoch % 20 == 0:
        fake_out = dis_model(pred_sample, training=False)
        loss = gen_loss(fake_out)
        noi_loss_arr.append(loss.numpy())

        F_NOI0 = open("loss/RandomSample/noi_loss.txt", "a")
        F_NOI0.write("{} : {}\n".format(epoch, loss.numpy()))
        F_NOI0.close()

        pred_sample = normal_census(pred_sample)
        F_NOI = open("loss/RandomSample/noi_sample.txt", "a")
        F_NOI.write("Epoch {} :\n".format(epoch))

        for i in range(nsamples):
            # print(i, test_noise[i], pred_sample[i])
            # print(disc_Score(pred_sample[i]), end=" ")
            isdisc = is_disc(pred_sample[i])
            F_NOI.write("Is_Disc ")
            F_NOI.write("{} : {}  ".format(i + 1, isdisc))
            num += isdisc
        F_NOI.write("\n")

        F_NOI.write("Num = {}\n".format(epoch, num))
        F_NOI.close()
    else:
        pred_sample = normal_census(pred_sample)

        for i in range(nsamples):
            isdisc = is_disc(pred_sample[i])
            num += isdisc
            # print(isdisc, " ")

    # if num > 20:
    #     F_NOI = open("loss/RandomSample/noi_sample.txt", "a")
    #     F_NOI.write("Epoch {} : {}\n".format(epoch, epochs))
    #     # for i in range(nsamples):
    #     #     F_NOI.write("Predict_Proba {} : {}  ".format(i, disc_Score(pred_sample[i])))
    #     # F_NOI.write("\n")
    #     for i in range(nsamples):
    #         F_NOI.write("Is_Disc {} : {}  ".format(i, is_disc(pred_sample[i])))
    #     F_NOI.write("\n")
    #     F_NOI.close()
    # print("")
    return num


def train(dataset, EPOCH):
    best_num = -1
    best_epoch = -1
    for epoch in range(EPOCH):
        gen_loss, dis_loss = 0, 0
        for image_batch in dataset:
            gen_loss, dis_loss = train_step(image_batch)
        if epoch % 20 == 0:
            gen_loss_arr.append(gen_loss.numpy())
            dis_loss_arr.append(dis_loss.numpy())

            F_GEN = open("loss/RandomSample/gen_loss.txt", "a")
            F_DIS = open("loss/RandomSample/dis_loss.txt", "a")
            F_GEN.write("Epochs-{} : {}\n".format(epoch, gen_loss.numpy()))
            F_DIS.write("Epochs-{} : {}\n".format(epoch, dis_loss.numpy()))
            F_DIS.close()
            F_GEN.close()
        print("Epoch {} / {}".format(epoch + 1, EPOCH))
        num = generator_sample(generator, discriminator, generator_loss, z, epoch)
        if num > best_num:
            best_num = num
            best_epoch = epoch

        if epoch % 1000 == 0 and epoch > 0:
            np.save("loss/RandomSample/loss_arr{}.npy".format(epoch), np.array(gen_loss_arr))

    print("best_num :", best_num)
    print("best_epoch :", best_epoch + 1)


# print(generator.trainable_variables)
train(datasets, epochs)
F_DIS.close()
F_GEN.close()
F_NOI.close()
# print(len(gen_loss_arr))
np.save("loss/RandomSample/gen_loss.npy", np.array(gen_loss_arr))
np.save("loss/RandomSample/noi_loss.npy", np.array(noi_loss_arr))
np.save("loss/RandomSample/dis_loss.npy", np.array(dis_loss_arr))
# pred_sample = generator(z, training=False)
# fake_out = discriminator(pred_sample, training=False)
# loss = generator_loss(fake_out)
# print(loss.numpy())
