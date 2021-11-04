import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
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


#
# test statistics for DGCIT
#


def t_and_sigma(psy_x_i, psy_y_i, phi_x_i, phi_y_i):
    b, n = psy_x_i.shape
    x_mtx = phi_x_i - psy_x_i
    y_mtx = phi_y_i - psy_y_i
    matrix = tf.reshape(x_mtx[None, :, :] * y_mtx[:, None, :], [-1, n])
    t_b = tf.reduce_sum(matrix, axis=1) / tf.cast(n, tf.float64)
    t_b = tf.expand_dims(t_b, axis=1)

    crit_matrix = matrix - t_b
    std_b = tf.sqrt(tf.reduce_sum(crit_matrix**2, axis=1) / tf.cast(n-1, tf.float64))
    return t_b, std_b


def test_statistics(psy_x_i, psy_y_i, phi_x_i, phi_y_i, t_b, std_b, j):
    b, n = psy_x_i.shape
    x_mtx = phi_x_i - psy_x_i
    y_mtx = phi_y_i - psy_y_i
    matrix = tf.reshape(x_mtx[None, :, :] * y_mtx[:, None, :], [-1, n])
    crit_matrix = matrix - t_b
    test_stat = tf.reduce_max(tf.abs(tf.sqrt(tf.cast(n, tf.float64)) * tf.squeeze(t_b) / std_b))

    sig = tf.reduce_sum(crit_matrix[None, :, :] * crit_matrix[:, None, :], axis=2)
    coef = std_b[None, :] * std_b[:, None] * tf.cast(n-1, tf.float64)
    sig_xy = sig / coef

    eigenvalues, eigenvectors = tf.linalg.eigh(sig_xy)
    base = tf.zeros_like(eigenvectors)
    eig_vals = tf.sqrt(eigenvalues + 1e-12)
    lamda = tf.linalg.set_diag(base, eig_vals)
    sig_sqrt = tf.matmul(tf.matmul(eigenvectors, lamda), tf.linalg.inv(eigenvectors))

    z_dist = tfp.distributions.Normal(0.0, scale=1.0)
    z_samples = z_dist.sample([b*b, j])
    z_samples = tf.cast(z_samples, tf.float64)
    vals = tf.matmul(sig_sqrt, z_samples)
    t_j = tf.reduce_max(vals, axis=0)
    return test_stat, t_j

#
# Training algorithm for DGCIT
#


def dgcit(n=500, z_dim=100, simulation='type1error', batch_size=64, n_iter=1000, train_writer=None,
          current_iters=0, nstd=1.0, z_dist='gaussian', x_dims=1, y_dims=1, a_x=0.05, M=500, k=2,
          var_idx=1, b=30, j=1000):
    # generate samples x, y, z
    # arguments: size, sType='CI', dx=1, dy=1, dz=20, nstd=1, fixed_function='linear',
    # debug=False, normalize=True, seed=None, dist_z='gaussian'
    if simulation == 'type1error':
        # generate samples x, y, z under null hypothesis - x and y are conditional independent
        x, y, z = generate_samples_random(size=n, sType='CI', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd, alpha_x=a_x,
                                          dist_z=z_dist)

    elif simulation == 'power':
        # generate samples x, y, z under alternative hypothesis - x and y are dependent
        x, y, z = generate_samples_random(size=n, sType='dependent', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd,
                                          alpha_x=a_x, dist_z=z_dist)

    elif simulation == 'ccle':
        x_drug, y, features = ccle_data.load_ccle(feature_type='mutation', normalize=False)

        ccle_selected, corrs = ccle_data.ccle_feature_filter(x_drug, y, threshold=0.05)

        features.index[ccle_selected]

        var_names = ['BRAF.MC_MUT', 'BRAF.V600E_MUT', 'HIP1_MUT', 'CDC42BPA_MUT', 'THBS3_MUT', 'DNMT1_MUT', 'PRKD1_MUT',
                     'FLT3_MUT', 'PIP5K1A_MUT', 'MAP3K5_MUT']
        idx = []

        for var in var_names:
            idx.append(features.T.columns.get_loc(var))

        x = x_drug[:, idx[5]]
        z = np.delete(x_drug, (idx[5]), axis=1).astype(np.float64)
        z_dim = z.shape[1]

        x = np.expand_dims(x, axis=1).astype(np.float64)
        y = np.expand_dims(y, axis=1)
        n = y.shape[0]
    elif simulation == 'brain':
        path = './data/ADNI-Mediation-new.csv'
        df = pd.read_csv(path, header=None)
        y = df.loc[:, 7].values
        age = df.loc[:, 5].values
        tr_measures = df.loc[:, 12:79].values
        ct_measures = df.loc[:, 80:].values
        all_data = np.concatenate((np.expand_dims(age, axis=1), tr_measures), axis=1)
        all_data = np.concatenate((all_data, ct_measures), axis=1)
        x_idx = np.argwhere(np.isnan(all_data))[:, 0]
        y_idx = np.argwhere(np.isnan(y))[:, 0]
        idx = np.concatenate([x_idx, y_idx])
        idx = np.unique(idx)
        idx_diff = np.arange(0, idx.shape[0])
        remove_idx = idx - idx_diff
        for i in remove_idx:
            all_data = np.delete(all_data, i, axis=0)
            y = np.delete(y, i, axis=0)

        all_data = np.delete(all_data, i, axis=0)
        y = np.delete(y, i, axis=0)
        x = all_data[:, var_idx]
        z = np.delete(all_data, (var_idx), axis=1).astype(np.float64)
        z_dim = z.shape[1]

        z = (z - z.min()) / (z.max() - z.min())
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        x = np.expand_dims(x, axis=1).astype(np.float64)
        y = np.expand_dims(y, axis=1)
        n = y.shape[0]

    else:
        raise ValueError('Test does not exist.')

    # split the train-test sets to k folds
    data_k = []
    idx = n // k
    epochs = int(n_iter)

    for j in range(k):
        x_train, y_train, z_train = x[j:(j+1)*idx, ], y[j:(j+1)*idx, ], z[j:(j+1)*idx, ]
        i = 0
        while i < k:
            if not i == j:
                x1, y1, z1 = x[i * idx:(i + 1) * idx, ], y[i * idx:(i + 1) * idx, ], z[i * idx:(i + 1) * idx, ]
                x_train = tf.concat([x_train, x1], axis=0)
                y_train = tf.concat([y_train, y1], axis=0)
                z_train = tf.concat([z_train, z1], axis=0)
            i += 1

        dataset = tf.data.Dataset.from_tensor_slices((x_train[idx:, ], y_train[idx:, ],
                                                      z_train[idx:, ]))
        # Repeat n epochs
        training = dataset.repeat(epochs)
        training = training.shuffle(100).batch(batch_size * 2)
        # test-set is the one left
        testing = tf.data.Dataset.from_tensor_slices((x_train[:idx, ], y_train[:idx, ],
                                                      z_train[:idx, ]))
        testing = testing.batch(1)
        data_k.append([training, testing])

    # no. of random and hidden dimensions
    if z_dim <= 20:
        v_dims = int(3)
        h_dims = int(3)

    else:
        v_dims = int(50)
        h_dims = int(512)

    v_dist = tfp.distributions.Normal(0, scale=tf.sqrt(1.0 / 3.0))
    # create instance of G & D
    lr = 0.0005
    generator_x = cit_gan.WGanGenerator(n, z_dim, h_dims, v_dims, x_dims, batch_size)
    generator_y = cit_gan.WGanGenerator(n, z_dim, h_dims, v_dims, y_dims, batch_size)
    discriminator_x = cit_gan.WGanDiscriminator(n, z_dim, h_dims, x_dims, batch_size)
    discriminator_y = cit_gan.WGanDiscriminator(n, z_dim, h_dims, y_dims, batch_size)

    gen_clipping_val = 0.5
    gen_clipping_norm = 1.0
    w_clipping_val = 0.5
    w_clipping_norm = 1.0
    scaling_coef = 1.0
    sinkhorn_eps = 0.8
    sinkhorn_l = 30

    gx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)
    gy_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dy_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)

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

    psy_x_all = []
    phi_x_all = []
    psy_y_all = []
    phi_y_all = []
    test_samples = b
    test_size = int(n/k)

    for batched_trainingset, batched_testset in data_k:
        for x_batch, y_batch, z_batch in batched_trainingset.take(n_iter):
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

            with train_writer.as_default():
                # tf.summary.scalar('Wasserstein X Discriminator Loss', x_disc_loss, step=current_iters)
                tf.summary.scalar('Wasserstein X GEN Loss', loss_x, step=current_iters)
                # tf.summary.scalar('Wasserstein Y Discriminator Loss', y_disc_loss, step=current_iters)
                tf.summary.scalar('Wasserstein Y GEN Loss', loss_y, step=current_iters)
                train_writer.flush()

            current_iters += 1

        psy_x_b = []
        phi_x_b = []
        psy_y_b = []
        phi_y_b = []

        x_samples = []
        y_samples = []
        z = []
        x = []
        y = []

        # the following code generate x_1, ..., x_400 for all B and it takes 61 secs for one test
        for test_x, test_y, test_z in batched_testset:
            tiled_z = tf.tile(test_z, [M, 1])
            noise_v = v_dist.sample([M, v_dims])
            noise_v = tf.cast(noise_v, tf.float64)
            g_inputs = tf.concat([tiled_z, noise_v], axis=1)
            # generator samples from G and evaluate from D
            fake_x = generator_x.call(g_inputs, training=False)
            fake_y = generator_y.call(g_inputs, training=False)
            x_samples.append(fake_x)
            y_samples.append(fake_y)
            z.append(test_z)
            x.append(test_x)
            y.append(test_y)

        standardise = True

        if standardise:
            x_samples = (x_samples - tf.reduce_mean(x_samples)) / tf.math.reduce_std(x_samples)
            y_samples = (y_samples - tf.reduce_mean(y_samples)) / tf.math.reduce_std(y_samples)
            x = (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
            y = (y - tf.reduce_mean(y)) / tf.math.reduce_std(y)
            z = (z - tf.reduce_mean(z)) / tf.math.reduce_std(z)

        f1 = cit_gan.CharacteristicFunction(M, x_dims, z_dim, test_size)
        f2 = cit_gan.CharacteristicFunction(M, y_dims, z_dim, test_size)
        for i in range(test_samples):
            phi_x = tf.reduce_mean(f1.call(x_samples, z), axis=1)
            phi_y = tf.reduce_mean(f2.call(y_samples, z), axis=1)
            psy_x = tf.squeeze(f1.call(x, z))
            psy_y = tf.squeeze(f2.call(y, z))

            psy_x_b.append(psy_x)
            phi_x_b.append(phi_x)
            psy_y_b.append(psy_y)
            phi_y_b.append(phi_y)
            f1.update()
            f2.update()

        psy_x_all.append(psy_x_b)
        phi_x_all.append(phi_x_b)
        psy_y_all.append(psy_y_b)
        phi_y_all.append(phi_y_b)

    # reshape
    psy_x_all = tf.reshape(psy_x_all, [k, test_samples, test_size])
    psy_y_all = tf.reshape(psy_y_all, [k, test_samples, test_size])
    phi_x_all = tf.reshape(phi_x_all, [k, test_samples, test_size])
    phi_y_all = tf.reshape(phi_y_all, [k, test_samples, test_size])
    
    t_b = 0.0
    std_b = 0.0
    for n in range(k):
        t, std = t_and_sigma(psy_x_all[n], psy_y_all[n], phi_x_all[n], phi_y_all[n])
        t_b += t
        std_b += std
    t_b = t_b / tf.cast(k, tf.float64)
    std_b = std_b / tf.cast(k, tf.float64)

    psy_x_all = tf.transpose(psy_x_all, (1, 0, 2))
    psy_y_all = tf.transpose(psy_y_all, (1, 0, 2))
    phi_x_all = tf.transpose(phi_x_all, (1, 0, 2))
    phi_y_all = tf.transpose(phi_y_all, (1, 0, 2))

    psy_x_all = tf.reshape(psy_x_all, [test_samples, test_size*k])
    psy_y_all = tf.reshape(psy_y_all, [test_samples, test_size*k])
    phi_x_all = tf.reshape(phi_x_all, [test_samples, test_size*k])
    phi_y_all = tf.reshape(phi_y_all, [test_samples, test_size*k])

    stat, critical_vals = test_statistics(psy_x_all, psy_y_all, phi_x_all, phi_y_all, t_b, std_b, j)
    comparison = [c > stat or c == stat for c in critical_vals]
    comparison = np.reshape(comparison, (-1,))
    p_value = np.sum(comparison.astype(np.float32)) / j
    return p_value


def rdc(x, y, f=np.sin, k=20, s=1 / 6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.

    Source: https://github.com/garydoranjr/rdc
    """
    x = tf.reshape(x, shape=(x.shape[0], ))
    y = tf.reshape(y, shape=(y.shape[0], ))

    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError:
                pass
        return np.median(values)

    if len(x.shape) == 1: x = tf.reshape(x, shape=(-1, 1))
    if len(y.shape) == 1: y = tf.reshape(y, shape=(-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in np.transpose(x)]) / float(x.shape[0])
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in np.transpose(y)]) / float(y.shape[0])

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s / X.shape[1]) * np.random.randn(X.shape[1], k)
    Ry = (s / Y.shape[1]) * np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:
        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0 + k, k0:k0 + k]
        Cxy = C[:k, k0:k0 + k]
        Cyx = C[k0:k0 + k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))


def permute(x):
    n = x.shape[0]
    idx = np.random.permutation(n)
    out = x[idx]
    return out

#
# test statistics and GCIT method
# Paper link: https://arxiv.org/pdf/1907.04068.pdf
#


def gcit_sinkhorn(n=1000, z_dim=100, simulation='type1error', statistic="rdc", batch_size=64, nstd=0.5,
                  z_dist='gaussian', n_iter=1000, current_iters=0, a_x=0.01, var_idx=1):
    # generate samples x, y, z
    # arguments: size, sType='CI', dx=1, dy=1, dz=20, nstd=1, fixed_function='linear',
    # debug=False, normalize=True, seed=None, dist_z='gaussian'
    x_dims = 1
    y_dims = 1
    if simulation == 'type1error':
        # generate samples x, y, z under null hypothesis - x and y are conditional independent
        x, y, z = generate_samples_random(size=n, sType='CI', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd,
                                          dist_z=z_dist)

    elif simulation == 'power':
        # generate samples x, y, z under alternative hypothesis - x and y are dependent
        x, y, z = generate_samples_random(size=n, sType='dependent', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd,
                                          dist_z=z_dist, alpha_x=a_x)

    elif simulation == 'ccle':
        x_drug, y, features = ccle_data.load_ccle(feature_type='mutation', normalize=False)

        ccle_selected, corrs = ccle_data.ccle_feature_filter(x_drug, y, threshold=0.05)

        features.index[ccle_selected]

        var_names = ['BRAF.MC_MUT', 'BRAF.V600E_MUT', 'HIP1_MUT', 'CDC42BPA_MUT', 'THBS3_MUT', 'DNMT1_MUT', 'PRKD1_MUT',
                     'FLT3_MUT', 'PIP5K1A_MUT', 'MAP3K5_MUT']
        idx = []

        for var in var_names:
            idx.append(features.T.columns.get_loc(var))

        x = x_drug[:, idx[0]]

        z = np.delete(x_drug, (idx[0]), axis=1).astype(np.float64)
        z_dim = z.shape[1]

        x = np.expand_dims(x, axis=1).astype(np.float64)
        y = np.expand_dims(y, axis=1)
        n = y.shape[0]

    elif simulation == 'brain':
        path = './data/ADNI-Mediation-new.csv'
        df = pd.read_csv(path, header=None)
        y = df.loc[:, 7].values
        age = df.loc[:, 5].values
        tr_measures = df.loc[:, 12:79].values
        ct_measures = df.loc[:, 80:].values
        all_data = np.concatenate((np.expand_dims(age, axis=1), tr_measures), axis=1)
        all_data = np.concatenate((all_data, ct_measures), axis=1)
        x_idx = np.argwhere(np.isnan(all_data))[:, 0]
        y_idx = np.argwhere(np.isnan(y))[:, 0]
        idx = np.concatenate([x_idx, y_idx])
        idx = np.unique(idx)
        idx_diff = np.arange(0, idx.shape[0])
        remove_idx = idx - idx_diff
        for i in remove_idx:
            all_data = np.delete(all_data, i, axis=0)
            y = np.delete(y, i, axis=0)

        all_data = np.delete(all_data, 0, axis=0)
        y = np.delete(y, 0, axis=0)
        x = all_data[:, var_idx]
        z = np.delete(all_data, (var_idx), axis=1).astype(np.float64)
        z_dim = z.shape[1]

        z = (z - z.min()) / (z.max() - z.min())
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        x = np.expand_dims(x, axis=1).astype(np.float64)
        y = np.expand_dims(y, axis=1)
        n = y.shape[0]
    else:
        raise ValueError('Test does not exist.')

    # define training and testing subsets, training for learning the sampler and
    # testing for computing test statistic. Set 2/3 and 1/3 as default
    x_train, y_train, z_train = x[:int(2 * n / 3), ], y[:int(2 * n / 3), ], z[:int(2 * n / 3), ]
    # build data pipline for training set
    training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, z_train))
    # Repeat n epochs
    epochs = int(n_iter)
    training_dataset = training_dataset.repeat(epochs)
    batched_training_set = training_dataset.shuffle(1000).batch(batch_size*2)
    # build data pipline for test set
    x_test, y_test, z_test = x[int(2 * n / 3):, ], y[int(2 * n / 3):, ], z[int(2 * n / 3):, ]

    # no. of random and hidden dimensions
    if z_dim <= 20:
        v_dims = int(3)
        h_dims = int(3)

    else:
        v_dims = 50
        h_dims = 512

    # v_mean = tf.zeros(v_dim, dtype=tf.float64)
    v_dist = tfp.distributions.Normal(0, scale=tf.sqrt(1.0/3.0))

    # create instance of G & D
    lr = 0.0005
    scaling_coef = 1.0
    sinkhorn_eps = 0.8
    sinkhorn_l = 30
    generator = cit_gan.WGanGenerator(n, z_dim, h_dims, v_dims, x_dims, batch_size)
    discriminator = cit_gan.WGanDiscriminator(n, z_dim, h_dims, x_dims, batch_size)

    gen_clipping_val = 0.5
    gen_clipping_norm = 1.0
    w_clipping_val = 0.5
    w_clipping_norm = 1.0

    gen_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    disc_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)

    stat = None

    # 2. Choice of statistic
    if statistic == "corr":
        stat = correlation
    if statistic == "mmd":
        stat = mmd_squared
    if statistic == "kolmogorov":
        stat = kolmogorov
    if statistic == "wilcox":
        stat = wilcox
    if statistic == "rdc":
        stat = rdc

    @tf.function
    def update_d(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        fake_x = generator.call(gen_inputs)
        fake_x_p = generator.call(gen_inputs_p)
        d_fake = tf.concat([fake_x, real_z], axis=1)
        d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)

        with tf.GradientTape() as disc_tape:
            f_real = discriminator.call(d_real)
            f_fake = discriminator.call(d_fake)
            f_real_p = discriminator.call(d_real_p)
            f_fake_p = discriminator.call(d_fake_p)
            # call compute loss using @tf.function + autograph

            loss1 = \
                gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, f_real_p, f_fake_p)
            # disc_loss = - tf.math.minimum(loss1, 1)
            disc_loss = - loss1
        # update discriminator parameters
        d_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimiser.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    @tf.function
    def update_g(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        with tf.GradientTape() as gen_tape:
            fake_x = generator.call(gen_inputs)
            fake_x_p = generator.call(gen_inputs_p)
            d_fake = tf.concat([fake_x, real_z], axis=1)
            d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
            f_real = discriminator.call(d_real)
            f_fake = discriminator.call(d_fake)
            f_real_p = discriminator.call(d_real_p)
            f_fake_p = discriminator.call(d_fake_p)
            # call compute loss using @tf.function + autograph
            gen_loss = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps,
                                                                           sinkhorn_l, f_real_p, f_fake_p)
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimiser.apply_gradients(zip(generator_grads, generator.trainable_variables))
        return gen_loss

    for x_batch, y_batch, z_batch in batched_training_set.take(n_iter):
        # for x_batch2, y_batch2, z_batch2 in batched_training_set1.take(1):
        if x_batch.shape[0] != batch_size * 2:
            continue
        x_batch1 = x_batch[0:batch_size, ...]
        x_batch2 = x_batch[batch_size:, ...]
        z_batch1 = z_batch[0:batch_size, ...]
        z_batch2 = z_batch[batch_size:, ...]

        noise_v = v_dist.sample([batch_size, v_dims])
        noise_v = tf.cast(noise_v, tf.float64)
        noise_v_p = v_dist.sample([batch_size, v_dims])
        noise_v_p = tf.cast(noise_v_p, tf.float64)
        update_d(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
        loss = update_g(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)

        if tf.abs(loss) < 0.1:
            break
        current_iters += 1

    test_samples = 1000
    rho = []
    test_size = z_test.shape[0]

    for i in range(test_samples):
        v = v_dist.sample([test_size, v_dims])
        v = tf.cast(v, tf.float64)
        g_inputs = tf.concat([z_test, v], axis=1)
        # generator samples from G and evaluate from D
        fake_data = generator.call(g_inputs, training=False)
        rho.append(stat(fake_data, y_test))

    rho = tf.stack(rho)
    stat_real = stat(x_test, y_test)
    # p-value computation as a two-sided test
    p_value = min(tf.reduce_sum(tf.cast(rho < stat_real, tf.float32)) / test_samples,
                  tf.reduce_sum(tf.cast(rho > stat_real, tf.float32)) / test_samples)
    return p_value


#
# test statistics and RCIT(regression) method
# Paper link: http://www.statslab.cam.ac.uk/~rds37/papers/Shah%20Peters%202018%20Conditional%20Independence.pdf
#


def rcit_test_stats(r):
    n = tf.cast(r.shape[0], tf.float64)
    r_sq_mean = tf.reduce_mean(r * r)
    r_mean = tf.reduce_mean(r)
    return tf.abs((tf.sqrt(n) * r_mean) / tf.sqrt((r_sq_mean - r_mean**2)))


def rcit(n=500, z_dim=100, simulation='type1error', batch_size=64, n_iter=1000, nstd=1.0, z_dist='gaussian', x_dims=1,
         y_dims=1, a_x=0.05):
    # generate samples x, y, z
    # arguments: size, sType='CI', dx=1, dy=1, dz=20, nstd=1, fixed_function='linear',
    # debug=False, normalize=True, seed=None, dist_z='gaussian'
    if simulation == 'type1error':
        # generate samples x, y, z under null hypothesis - x and y are conditional independent
        x, y, z = generate_samples_random(size=n, sType='CI', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd,
                                          alpha_x=a_x, dist_z=z_dist)

    elif simulation == 'power':
        # generate samples x, y, z under alternative hypothesis - x and y are dependent
        x, y, z = generate_samples_random(size=n, sType='dependent', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd,
                                          alpha_x=a_x, dist_z=z_dist)

    elif simulation == 'ccle':
        x_drug, y, features = ccle_data.load_ccle(feature_type='mutation', normalize=False)

        ccle_selected, corrs = ccle_data.ccle_feature_filter(x_drug, y, threshold=0.05)

        features.index[ccle_selected]

        var_names = ['BRAF.MC_MUT', 'BRAF.V600E_MUT', 'HIP1_MUT', 'CDC42BPA_MUT', 'THBS3_MUT', 'DNMT1_MUT', 'PRKD1_MUT',
                     'FLT3_MUT', 'PIP5K1A_MUT', 'MAP3K5_MUT']
        idx = []

        for var in var_names:
            idx.append(features.T.columns.get_loc(var))

        x = x_drug[:, idx[8]]
        z = np.delete(x_drug, (idx[8]), axis=1).astype(np.float64)
        z_dim = z.shape[1]

        x = np.expand_dims(x, axis=1).astype(np.float64)
        y = np.expand_dims(y, axis=1)
        n = y.shape[0]
    else:
        raise ValueError('Test does not exist.')

    # no. of random and hidden dimensions
    if z_dim <= 20:
        v_dims = int(3)
        h_dims = int(3)

    else:
        v_dims = 10
        h_dims = 128

    # distribution for noise v
    v_dist = tfp.distributions.Normal(0, scale=tf.sqrt(1.0 / 3.0))

    # create instance of G & D
    lr = 1e-4
    # input_dims = x_train.shape[1]
    f_x = cit_gan.WGanGenerator(n, z_dim, h_dims, v_dims, x_dims, batch_size)
    f_y = cit_gan.WGanGenerator(n, z_dim, h_dims, v_dims, x_dims, batch_size)

    gen_clipping_val = 0.5
    gen_clipping_norm = 1.0

    x_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    y_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)

    # train
    for i in range(n_iter):
        noise_v = v_dist.sample([n, v_dims])
        noise_v = tf.cast(noise_v, tf.float64)
        g_inputs = tf.concat([z, noise_v], axis=1)
        with tf.GradientTape() as gg:
            # generator samples from G and evaluate from D
            fake_x = f_x.call(g_inputs, training=False)
            x_loss = tf.keras.losses.MSE(x, fake_x)

        x_grads = gg.gradient(x_loss, f_x.trainable_variables)
        x_optimiser.apply_gradients(zip(x_grads, f_x.trainable_variables))

        with tf.GradientTape() as tape_y:
            # generator samples from G and evaluate from D
            fake_y = f_y.call(g_inputs, training=False)
            y_loss = tf.keras.losses.MSE(y, fake_y)

        y_grads = tape_y.gradient(y_loss, f_y.trainable_variables)
        y_optimiser.apply_gradients(zip(y_grads, f_y.trainable_variables))

    # test
    noise_v = v_dist.sample([n, v_dims])
    noise_v = tf.cast(noise_v, tf.float64)
    g_inputs = tf.concat([z, noise_v], axis=1)
    # generator samples from G and evaluate from D
    fake_x = f_x.call(g_inputs, training=False)
    fake_y = f_y.call(g_inputs, training=False)
    r_i = (x - fake_x) * (y - fake_y)

    r = tf.reshape(r_i, [-1, 1])
    t_n = rcit_test_stats(r)

    # use dummy p-values
    if t_n >= 1.645:
        p_value_1 = 0.005
    else:
        p_value_1 = 0.5
    if t_n >= 1.96:
        p_value_5 = 0.005
    else:
        p_value_5 = 0.5

    return p_value_1, p_value_5

#
# KCIT experiment results are obtained by running the original implementation.
# Paper link: https://arxiv.org/ftp/arxiv/papers/1202/1202.3775.pdf
#
