# -*- coding: utf-8 -*-
# @Author  : TAKR-Zz
# @Time    : 2022/6/27 15:12
# @Function:
import time
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

discSample = np.load("local_samples_MLP_expga.npy")
# discSample = np.load("new_local_samples.npy")
print(discSample.shape)
discSample = tf.cast(discSample, tf.float32)
discSample = np.array(discSample)


def bank_normal(dataset):
    newData = []
    for x in dataset:
        newEle = []
        '''
                    input_bounds.append([1, 9])
                    input_bounds.append([0, 11])
                    input_bounds.append([0, 2])
                    input_bounds.append([0, 3])
                    input_bounds.append([0, 1])
                    input_bounds.append([-20, 179])
                    input_bounds.append([0, 1])
                    input_bounds.append([0, 1])
                    input_bounds.append([0, 2])
                    input_bounds.append([1, 31])
                    input_bounds.append([0, 11])
                    input_bounds.append([0, 99])
                    input_bounds.append([1, 63])
                    input_bounds.append([-1, 39])
                    input_bounds.append([0, 1])
                    input_bounds.append([0, 3])
        '''
        newEle.append((x[0] - 5) / 4)
        newEle.append((x[1] - 5.5) / 5.5)
        newEle.append((x[2] - 1) / 1)
        newEle.append((x[3] - 1.5) / 1.5)
        newEle.append((x[4] - 0.5) / 0.5)
        newEle.append((x[5] - 79.5) / 99.5)
        newEle.append((x[6] - 0.5) / 0.5)
        newEle.append((x[7] - 0.5) / 0.5)
        newEle.append((x[8] - 1) / 1)
        newEle.append((x[9] - 16) / 15)
        newEle.append((x[10] - 5.5) / 5.5)
        newEle.append((x[11] - 49.5) / 49.5)
        newEle.append((x[12] - 32) / 31)
        newEle.append((x[13] - 19) / 20)
        newEle.append((x[14] - 0.5) / 0.5)
        newEle.append((x[15] - 1.5) / 1.5)

        newData.append(newEle)
    return newData


def normal_bank(dataset):
    newData = []

    for d in dataset:
        newEle = []
        x = d
        # print(d)
        '''
                    input_bounds.append([1, 9])
                    input_bounds.append([0, 11])
                    input_bounds.append([0, 2])
                    input_bounds.append([0, 3])
                    input_bounds.append([0, 1])
                    input_bounds.append([-20, 179])
                    input_bounds.append([0, 1])
                    input_bounds.append([0, 1])
                    input_bounds.append([0, 2])
                    input_bounds.append([1, 31])
                    input_bounds.append([0, 11])
                    input_bounds.append([0, 99])
                    input_bounds.append([1, 63])
                    input_bounds.append([-1, 39])
                    input_bounds.append([0, 1])
                    input_bounds.append([0, 3])
        '''
        newEle.append(int(x[0] * 4 + 5))
        newEle.append(int((x[1]) * 5.5 + 5.5))
        newEle.append(int((x[2]) + 1))
        newEle.append(int((x[3]) * 1.5 + 1.5))
        newEle.append(int((x[4]) * 0.5 + 0.5))
        newEle.append(int(x[5] * 99.5 + 79.5))
        newEle.append(int((x[6]) * 0.5 + 0.5))
        newEle.append(int((x[7]) * 0.5 + 0.5))
        newEle.append(int((x[8]) + 1))
        newEle.append(int(x[9] * 15 + 16))
        newEle.append(int((x[10]) * 5.5 + 5.5))
        newEle.append(int((x[11]) * 49.5 + 49.5))
        newEle.append(int(x[12] * 31 + 32))
        newEle.append(int(x[13] * 20 + 19))
        newEle.append(int((x[14]) * 0.5 + 0.5))
        newEle.append(int((x[15]) * 1.5 + 1.5))
        newData.append(newEle)
        # print(newEle)
    return newData


start = time.time()

discSample = np.array(bank_normal(discSample))
print(discSample.shape)
# print(discSample[0])
# print(discSample[1])
end = time.time()
print("Takes :", end - start, "s")

MLPMODEL = joblib.load("model/bank/MLP_unfair.pkl")
print(MLPMODEL)

BATCH_SIZE = 64



discSample = tf.expand_dims(discSample, -1)

datasets = tf.data.Dataset.from_tensor_slices(discSample)
datasets = datasets.shuffle(40655).batch(BATCH_SIZE)

print(datasets)


def generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(12, input_shape=(8,), use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(14, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(16, use_bias=False, activation='tanh'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((16, 1)))
    return model


def discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(32, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(32, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(1))
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_out, fake_out):
    real_loss = cross_entropy(0.9 * tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    print("discriminator_loss : ", fake_loss + real_loss)
    return real_loss + fake_loss


def generator_loss(fake_out):
    print("generator_loss : ", cross_entropy(0.9 * tf.ones_like(fake_out), fake_out))
    return cross_entropy(0.9 * tf.ones_like(fake_out), fake_out)


generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

epochs = 100
noise_dim = 8

nsamples = 20

tf.random.set_seed(2022)
z = tf.random.normal([nsamples, noise_dim])

generator = generator_model()
discriminator = discriminator_model()

print(generator.trainable_variables)


# @tf.function
def train_step(dataInput):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_sample = generator(noise, training=True)

        real_out = discriminator(dataInput, training=True)
        fake_out = discriminator(gen_sample, training=True)

        gen_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)

    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))


def disc_Score(input):
    max_pro = 0.0
    min_pro = 1.0
    for i in range(1, 10):
        input[0] = i
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
    for i in range(1, 10):
        sample[0] = i
        if MLPMODEL.predict([sample])[0] == 1:
            int1 += 1
        else:
            int0 += 1
    # print(int0, int1)
    if int1 * int0 != 0:
        return 1
    return 0

def generator_sample(gen_model, test_noise):
    pred_image = gen_model(test_noise, training=False)
    pred_image = normal_bank(pred_image)
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


def train(dataset, EPOCH):
    for epoch in range(EPOCH):
        for image_batch in dataset:
            train_step(image_batch)
        print("Epoch {}/{}".format(epoch + 1, EPOCH))
        generator_sample(generator, z)


# print(generator.trainable_variables)
train(datasets, epochs)
# print(generator.trainable_variables)

# print(generator)
