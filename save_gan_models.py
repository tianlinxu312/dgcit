import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
import argparse
import math
from datetime import datetime
import logging
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import random
from scipy import stats
from collections import defaultdict
import warnings
import cit_gan
from scipy.stats import rankdata
import xlwt
from tempfile import TemporaryFile
import scipy
import gan_utils
import pandas as pd
import ccle_data
tf.random.set_seed(42)
np.random.seed(42)

#
# The generate_samples_random function and rdc function were inspired by
# GCIT Github repository by Alexis Bellot1,2 Mihaela van der Schaar
# source: https://github.com/alexisbellot/GCIT
#


def generate_samples_random(size=1000, sType='CI', dx=1, dy=1, dz=20, nstd=0.05, alpha_x=0.05,
                            normalize=True, seed=None, dist_z='gaussian'):
    '''
    Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian or Laplace
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI, I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        we set f1 to be sin function and f2 to be cos function.
    Output:
        Samples X, Y, Z
    '''
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    num = size

    if dist_z == 'gaussian':
        cov = np.eye(dz)
        mu = np.zeros(dz)
        Z = np.random.multivariate_normal(mu, cov, num)

    elif dist_z == 'laplace':
        Z = np.random.laplace(loc=0.0, scale=1.0, size=num*dz)
        Z = np.reshape(Z, (num, dz))

    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / np.linalg.norm(Ax[:, i], ord=1)
    Ay = np.random.rand(dz, dy)
    for i in range(dy):
        Ay[:, i] = Ay[:, i] / np.linalg.norm(Ay[:, i], ord=1)

    Axy = np.ones((dx, dy)) * alpha_x

    if sType == 'CI':
        X = np.sin(np.matmul(Z, Ax) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        # X = np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num) + np.matmul(Z, Ax)
        # X = np.random.uniform(-1.0, 1.0, size=(size, 1))
        Y = np.cos(np.matmul(Z, Ay) + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    elif sType == 'I':
        X = np.sin(nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = np.cos(nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    else:
        X = np.sin(np.matmul(Z, Ax) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = np.cos(np.matmul(X, Axy) + np.matmul(Z, Ay) + nstd * np.random.multivariate_normal(np.zeros(dx),
                                                                                               np.eye(dx), num))

    if normalize:
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())

    return np.array(X), np.array(Y), np.array(Z)


def dgcit(n=500, z_dim=100, simulation='type1error', batch_size=64, n_iter=1000, train_writer=None,
          current_iters=0, nstd=1.0, z_dist='gaussian', x_dims=1, y_dims=1, a_x=0.05, M=500, k=2,
          var_idx=1, b=30, j=1000, s=None):
    # generate samples x, y, z
    # arguments: size, sType='CI', dx=1, dy=1, dz=20, nstd=1, fixed_function='linear',
    # debug=False, normalize=True, seed=None, dist_z='gaussian'
    if simulation == 'type1error':
        # generate samples x, y, z under null hypothesis - x and y are conditional independent
        x, y, z = generate_samples_random(size=n, sType='CI', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd, alpha_x=a_x,
                                          dist_z=z_dist, seed=s)

    elif simulation == 'power':
        # generate samples x, y, z under alternative hypothesis - x and y are dependent
        x, y, z = generate_samples_random(size=n, sType='dependent', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd,
                                          alpha_x=a_x, dist_z=z_dist)

    else:
        raise ValueError('Test does not exist.')

    if k == 2:
        # define training and testing subsets, training for learning the sampler and
        # testing for computing test statistic. Set 2/3 and 1/3 as default
        # x_train, y_train, z_train = x[:int(2 * n / 3), ], y[:int(2 * n / 3), ], z[:int(2 * n / 3), ]
        x_train, y_train, z_train = x[:int(n / 2), ], y[:int(n / 2), ], z[:int(n / 2), ]
        # build data pipline for test set
        x_test, y_test, z_test = x[int(n / 2):, ], y[int(n / 2):, ], z[int(n / 2):, ]
        # build data pipline for training set
        dataset1 = tf.data.Dataset.from_tensor_slices((x_train, y_train, z_train))
        testset1 = tf.data.Dataset.from_tensor_slices((x_test, y_test, z_test))
        dataset2 = tf.data.Dataset.from_tensor_slices((x_test, y_test, z_test))
        testset2 = tf.data.Dataset.from_tensor_slices((x_train, y_train, z_train))

        # Repeat n epochs
        epochs = int(n_iter)
        dataset1 = dataset1.repeat(epochs)
        batched_train1 = dataset1.shuffle(300).batch(batch_size * 2)
        # batched_training_set1 = dataset1.shuffle(300).batch(batch_size)
        batched_test1 = testset1.batch(1)

        dataset2 = dataset2.repeat(epochs)
        batched_train2 = dataset2.shuffle(300).batch(batch_size * 2)
        batched_test2 = testset2.batch(1)
        data_k = [[batched_train1, batched_test1], [batched_train2, batched_test2]]

    else:
        k = 3
        # Repeat n epochs
        epochs = int(n_iter)
        # define training and testing subsets, I1, I2,..., IK for learning the sampler and
        # testing for computing test statistic.
        x_1, y_1, z_1 = x[:int(1 * n / k), ], y[:int(1 * n / k), ], z[:int(1 * n / k), ]
        # build subset I2
        x_2, y_2, z_2 = x[int(1 * n / k):int(2 * n / k), ], y[int(1 * n / k):int(2 * n / k), ], \
                        z[int(1 * n / k):int(2 * n / k), ]
        # build subset I3
        x_3, y_3, z_3 = x[int(2 * n / k):, ], y[int(2 * n / k):, ], z[int(2 * n / k):, ]

        # build data pipline for training set I1
        train_x1 = tf.concat([x_1, x_2], axis=0)
        train_y1 = tf.concat([y_1, y_2], axis=0)
        train_z1 = tf.concat([z_1, z_2], axis=0)
        I1_dataset = tf.data.Dataset.from_tensor_slices((train_x1, train_y1, train_z1))
        # Repeat n epochs
        I1_training = I1_dataset.repeat(epochs)
        I1_training = I1_training.shuffle(100).batch(batch_size*2)
        # test-set is the one left
        I1_test = tf.data.Dataset.from_tensor_slices((x_3, y_3, z_3))
        I1_test = I1_test.batch(1)

        train_x2 = tf.concat([x_2, x_3], axis=0)
        train_y2 = tf.concat([y_2, y_3], axis=0)
        train_z2 = tf.concat([z_2, z_3], axis=0)
        I2_dataset = tf.data.Dataset.from_tensor_slices((train_x2, train_y2, train_z2))
        # Repeat n epochs
        I2_training = I2_dataset.repeat(epochs)
        I2_training = I2_training.shuffle(100).batch(batch_size*2)
        I2_test = tf.data.Dataset.from_tensor_slices((x_1, y_1, z_1))
        I2_test = I2_test.batch(1)

        train_x3 = tf.concat([x_1, x_3], axis=0)
        train_y3 = tf.concat([y_1, y_3], axis=0)
        train_z3 = tf.concat([z_1, z_3], axis=0)
        I3_dataset = tf.data.Dataset.from_tensor_slices((train_x3, train_y3, train_z3))
        # Repeat n epochs
        I3_training = I3_dataset.repeat(epochs)
        I3_training = I3_training.shuffle(100).batch(batch_size*2)
        I3_test = tf.data.Dataset.from_tensor_slices((x_2, y_2, z_2))
        I3_test = I3_test.batch(1)

        data_k = [[I1_training, I1_test], [I2_training, I2_test], [I3_training, I3_test]]
    return data_k


def main():
    # model
    model = args.model
    # number of samples
    sample_size = args.n_samples
    batch_size = args.batch_size
    # z_dims_scheme = args.z_scheme
    z_dims_scheme = [args.z_dims]
    dx = args.x_dims
    dy = args.y_dims
    # alpha_scheme = [0.01, 0.02, 0.03, 0.04, 0.05]
    var_list = np.arange(0, 137)
    alpha_scheme = [args.alpha_x]
    test = args.test
    n_test = args.n_tests
    n_iters = args.n_iters
    eps_std = args.eps_std
    dist_z = args.z_dist
    alpha_x = args.alpha_x
    m_value = args.m_value
    k_value = args.n_k
    b_value = args.b_b
    j_value = args.j_j
    lr = 0.0005

    seed_save = []
    z_dim = 100
    train_writer = None
    test_count = 0

    gen_clipping_val = 0.5
    gen_clipping_norm = 1.0
    w_clipping_val = 0.5
    w_clipping_norm = 1.0
    scaling_coef = 1.0
    sinkhorn_eps = 0.8
    sinkhorn_l = 30
    v_dist = tfp.distributions.Normal(0, scale=tf.sqrt(1.0 / 3.0))

    gx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)
    gy_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dy_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)

    # no. of random and hidden dimensions
    if z_dim <= 20:
        v_dims = int(3)
        h_dims = int(3)

    else:
        v_dims = int(50)
        h_dims = int(512)
        # v_dims = 10
        # h_dims = 128

    # create instance of G & D
    # input_dims = x_train.shape[1]

    for i in range(1, n_test+1):
        s = np.random.randint(low=0, high=100000, size=1)
        seed_save.append(s)
        data_k = dgcit(n=sample_size, z_dim=z_dim, simulation=test, batch_size=batch_size, n_iter=n_iters,
                       train_writer=train_writer, current_iters=test_count * n_test, nstd=eps_std, z_dist=dist_z,
                       x_dims=dx, y_dims=dy, a_x=alpha_x, M=m_value, k=k_value, b=b_value, j=j_value)

        k = 1
        for batched_trainingset, batched_testset in data_k:

            generator_x = cit_gan.WGanGenerator(sample_size, z_dim, h_dims, v_dims, dx, batch_size)
            generator_y = cit_gan.WGanGenerator(sample_size, z_dim, h_dims, v_dims, dy, batch_size)
            discriminator_x = cit_gan.WGanDiscriminator(sample_size, z_dim, h_dims, dx, batch_size)
            discriminator_y = cit_gan.WGanDiscriminator(sample_size, z_dim, h_dims, dy, batch_size)

            @tf.function
            def x_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p):
                gen_inputs = tf.concat([real_z, v], axis=1)
                gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
                # concatenate real inputs for WGAN discriminator (x, z)
                d_real = tf.concat([real_x, real_z], axis=1)
                d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
                fake_x = generator_x.call(gen_inputs)
                fake_x_p = generator_x.call(gen_inputs_p)
                d_fake = tf.concat([fake_x, real_z], axis=1)
                d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)

                with tf.GradientTape() as disc_tape:
                    f_real = discriminator_x.call(d_real)
                    f_fake = discriminator_x.call(d_fake)
                    f_real_p = discriminator_x.call(d_real_p)
                    f_fake_p = discriminator_x.call(d_fake_p)
                    # call compute loss using @tf.function + autograph

                    loss1 = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                                                     f_real_p, f_fake_p)
                    # disc_loss = - tf.math.minimum(loss1, 1)
                    disc_loss = - loss1
                # update discriminator parameters
                d_grads = disc_tape.gradient(disc_loss, discriminator_x.trainable_variables)
                dx_optimiser.apply_gradients(zip(d_grads, discriminator_x.trainable_variables))

            @tf.function
            def x_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p):
                gen_inputs = tf.concat([real_z, v], axis=1)
                gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
                # concatenate real inputs for WGAN discriminator (x, z)
                d_real = tf.concat([real_x, real_z], axis=1)
                d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
                with tf.GradientTape() as gen_tape:
                    fake_x = generator_x.call(gen_inputs)
                    fake_x_p = generator_x.call(gen_inputs_p)
                    d_fake = tf.concat([fake_x, real_z], axis=1)
                    d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
                    f_real = discriminator_x.call(d_real)
                    f_fake = discriminator_x.call(d_fake)
                    f_real_p = discriminator_x.call(d_real_p)
                    f_fake_p = discriminator_x.call(d_fake_p)
                    # call compute loss using @tf.function + autograph
                    gen_loss = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps,
                                                        sinkhorn_l, f_real_p, f_fake_p)
                # update generator parameters
                generator_grads = gen_tape.gradient(gen_loss, generator_x.trainable_variables)
                gx_optimiser.apply_gradients(zip(generator_grads, generator_x.trainable_variables))
                return gen_loss

            @tf.function
            def y_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p):
                gen_inputs = tf.concat([real_z, v], axis=1)
                gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
                # concatenate real inputs for WGAN discriminator (x, z)
                d_real = tf.concat([real_x, real_z], axis=1)
                d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
                fake_x = generator_y.call(gen_inputs)
                fake_x_p = generator_y.call(gen_inputs_p)
                d_fake = tf.concat([fake_x, real_z], axis=1)
                d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)

                with tf.GradientTape() as disc_tape:
                    f_real = discriminator_y.call(d_real)
                    f_fake = discriminator_y.call(d_fake)
                    f_real_p = discriminator_y.call(d_real_p)
                    f_fake_p = discriminator_y.call(d_fake_p)
                    # call compute loss using @tf.function + autograph

                    loss1 = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                                                     f_real_p, f_fake_p)
                    disc_loss = - loss1
                # update discriminator parameters
                d_grads = disc_tape.gradient(disc_loss, discriminator_y.trainable_variables)
                dy_optimiser.apply_gradients(zip(d_grads, discriminator_y.trainable_variables))

            @tf.function
            def y_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p):
                gen_inputs = tf.concat([real_z, v], axis=1)
                gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
                # concatenate real inputs for WGAN discriminator (x, z)
                d_real = tf.concat([real_x, real_z], axis=1)
                d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
                with tf.GradientTape() as gen_tape:
                    fake_x = generator_y.call(gen_inputs)
                    fake_x_p = generator_y.call(gen_inputs_p)
                    d_fake = tf.concat([fake_x, real_z], axis=1)
                    d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
                    f_real = discriminator_y.call(d_real)
                    f_fake = discriminator_y.call(d_fake)
                    f_real_p = discriminator_y.call(d_real_p)
                    f_fake_p = discriminator_y.call(d_fake_p)
                    # call compute loss using @tf.function + autograph
                    gen_loss = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps,
                                                        sinkhorn_l, f_real_p, f_fake_p)
                # update generator parameters
                generator_grads = gen_tape.gradient(gen_loss, generator_y.trainable_variables)
                gy_optimiser.apply_gradients(zip(generator_grads, generator_y.trainable_variables))
                return gen_loss

            for x_batch, y_batch, z_batch in batched_trainingset.take(n_iters):
                if x_batch.shape[0] != batch_size * 2:
                    continue
                x_batch1 = x_batch[0:batch_size, ...]
                x_batch2 = x_batch[batch_size:, ...]
                y_batch1 = y_batch[0:batch_size, ...]
                y_batch2 = y_batch[batch_size:, ...]
                z_batch1 = z_batch[0:batch_size, ...]
                z_batch2 = z_batch[batch_size:, ...]

                noise_v = v_dist.sample([batch_size, v_dims])
                noise_v = tf.cast(noise_v, tf.float64)
                noise_v_p = v_dist.sample([batch_size, v_dims])
                noise_v_p = tf.cast(noise_v_p, tf.float64)
                x_update_d(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
                loss_x = x_update_g(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
                y_update_d(y_batch1, y_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
                loss_y = y_update_g(y_batch1, y_batch2, z_batch1, z_batch2, noise_v, noise_v_p)

            generator_x.save_weights("./trained/{}/generatorX_iter{}_k{}/".format(test, i, k))
            generator_y.save_weights("./trained/{}/generatorY_iter{}_k{}/".format(test, i, k))
            print('Save {}-{} models'.format(i, k))
            k += 1
    np.savez("./trained/{}/random_seed.npz".format(test), np.asarray(seed_save))
    print('Saved random seeds!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cit-gan')
    parser.add_argument('-m', '--model', type=str, default='dgcit', choices=['dgcit', 'gcit', 'rcit'])
    parser.add_argument('-t', '--test', type=str, default='type1error',
                        choices=['type1error', 'power', 'ccle', 'brain'])
    parser.add_argument('-n', '--n_samples', type=int, default=501)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-nt', '--n_tests', type=int, default=500)  # number of p_values
    parser.add_argument('-ni', '--n_iters', type=int, default=1000)  # number of iterations to train GANs
    parser.add_argument('-dx', '--x_dims', type=int, default=1)
    parser.add_argument('-dy', '--y_dims', type=int, default=1)
    parser.add_argument('-dz', '--z_dims', type=int, default=100)
    parser.add_argument('-estd', '--eps_std', type=float, default=0.5)
    parser.add_argument('-zd', '--z_dist', type=str, default='gaussian', choices=['gaussian', 'laplace'])
    parser.add_argument('-ax', '--alpha_x', type=float, default=0.9)  # alpha before x in H1
    parser.add_argument('-zs', '--z_scheme', type=int, default=[50])
    parser.add_argument('-mv', '--m_value', type=int, default=100)
    parser.add_argument('-k', '--n_k', type=int, default=3)
    parser.add_argument('-b', '--b_b', type=int, default=100)
    parser.add_argument('-j', '--j_j', type=int, default=1000)
    args = parser.parse_args()
    main()


