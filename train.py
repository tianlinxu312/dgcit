import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from datetime import datetime
import utils
import cit_gan
import decimal
import gan_utils
import argparse
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser(description='cit-gan')
parser.add_argument('-m', '--model', type=str, default='dgcit', choices=['dgcit', 'gcit', 'rcit'])
parser.add_argument('-t', '--test', type=str, default='type1error', choices=['type1error', 'power', 'ccle', 'brain'])
parser.add_argument('-n', '--n_samples', type=int, default=500)
parser.add_argument('-bs', '--batch_size', type=int, default=64)
parser.add_argument('-nt', '--n_tests', type=int, default=500) # number of p_values
parser.add_argument('-ni', '--n_iters', type=int, default=1000) # number of iterations to train GANs
parser.add_argument('-dx', '--x_dims', type=int, default=1)
parser.add_argument('-dy', '--y_dims', type=int, default=1)
parser.add_argument('-dz', '--z_dims', type=int, default=100)
parser.add_argument('-estd', '--eps_std', type=float, default=0.5)
parser.add_argument('-zd', '--z_dist', type=str, default='gaussian', choices=['gaussian', 'laplace'])
parser.add_argument('-ax', '--alpha_x', type=float, default=0.9) # alpha before x in H1
parser.add_argument('-zs', '--z_scheme', type=int, default=[50])
parser.add_argument('-mv', '--m_value', type=int, default=100)
parser.add_argument('-k', '--n_k', type=int, default=3)
parser.add_argument('-b', '--b_b', type=int, default=10)
parser.add_argument('-j', '--j_j', type=int, default=10)
args = parser.parse_args()


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

    saved_file = "{}-{}{}-{}-{}".format(model, datetime.now().strftime("%h"), datetime.now().strftime("%d"),
                                        datetime.now().strftime("%H"), datetime.now().strftime("%M"))
    log_dir = "./trained/{}/log".format(saved_file)
    base_path = './trained/{}/'.format(saved_file)
    train_writer = tf.summary.create_file_writer(logdir=log_dir)

    alpha = 0.1
    alpha1 = 0.05

    if test == 'type1error':
        for z_dim in z_dims_scheme:
            p_values = []
            p_values1 = []
            p_values5 = []
            test_count = 0
            for n in range(n_test):
                start_time = datetime.now()
                p_value = 0.0
                p_value1 = 0.0
                p_value5 = 0.0
                if model == 'dgcit':
                    p_value = utils.dgcit(n=sample_size, z_dim=z_dim, simulation=test, batch_size=batch_size,
                                          n_iter=n_iters, train_writer=train_writer, current_iters=test_count * n_test,
                                          nstd=eps_std, z_dist=dist_z, x_dims=dx, y_dims=dy, a_x=alpha_x, M=m_value,
                                          k=k_value, b=b_value, j=j_value)

                elif model == 'gcit':
                    p_value = utils.gcit_sinkhorn(n=sample_size, z_dim=z_dim, simulation=test, statistic="rdc",
                                                  batch_size=batch_size, n_iter=n_iters, nstd=eps_std, z_dist=dist_z)

                elif model == 'rcit':
                    p_value1, p_value5 = utils.rcit(n=sample_size, z_dim=z_dim, simulation=test, batch_size=batch_size,
                                                    n_iter=n_iters, nstd=eps_std, z_dist=dist_z, x_dims=dx, y_dims=dy,
                                                    a_x=alpha_x)
                else:
                    raise ValueError('Test does not exist.')

                test_count += 1
                print("--- The %d'th iteration take %s seconds ---" % (test_count, (datetime.now() - start_time)))

                if model == 'rcit':
                    p_values1.append(p_value1)
                    p_values5.append(p_value5)
                    fp = [pval < alpha / 2.0 for pval in p_values1]
                    final_result = tf.reduce_sum(tf.cast(fp, tf.float32)) / len(fp)
                    fp1 = [pval < alpha1 / 2.0 for pval in p_values5]
                    final_result1 = tf.reduce_sum(tf.cast(fp1, tf.float32)) / len(fp1)

                else:
                    p_values.append(p_value)
                    fp = [pval < alpha / 2.0 for pval in p_values]
                    final_result = tf.reduce_sum(tf.cast(fp, tf.float32)) / len(fp)
                    fp1 = [pval < alpha1 / 2.0 for pval in p_values]
                    final_result1 = tf.reduce_sum(tf.cast(fp1, tf.float32)) / len(fp1)

                print('Type 1 error: {} for z dimension {} with significance level {}'.format(final_result, z_dim,
                                                                                              alpha))
                print('Type 1 error: {} for z dimension {} with significance level {}'.format(final_result1, z_dim,
                                                                                              alpha1))

            if model == 'rcit':
                filename1 = '{}_z_dims{}_z_distribution_{}x_dim_{}sig0.05.npz'.format(test, z_dim, dist_z, dx)
                np.savez(os.path.join(base_path, filename1), np.asarray(p_values5))
                filename2 = '{}_z_dims{}_z_distribution_{}x_dim_{}sig0.1.npz'.format(test, z_dim, dist_z, dx)
                np.savez(os.path.join(base_path, filename2), np.asarray(p_values1))
            else:
                filename = '{}_z_dims{}_z_distribution_{}x_dim_{}.npz'.format(test, z_dim, dist_z, dx)
                np.savez(os.path.join(base_path, filename), np.asarray(p_values))

    elif test == 'power':
        for al in alpha_scheme:
            for z_dim in z_dims_scheme:
                p_values = []
                p_values1 = []
                p_values5 = []
                test_count = 0
                for n in range(n_test):
                    start_time = datetime.now()
                    p_value = 0.0
                    p_value1 = 0.0
                    p_value5 = 0.0
                    if model == 'dgcit':
                        p_value = utils.dgcit(n=sample_size, z_dim=z_dim, simulation=test, batch_size=batch_size,
                                              n_iter=n_iters, train_writer=train_writer,
                                              current_iters=test_count * n_test, nstd=eps_std, z_dist=dist_z,
                                              x_dims=dx, y_dims=dy, a_x=alpha_x, M=m_value, k=k_value, b=b_value, j=j_value)
                    elif model == 'gcit':
                        p_value = utils.gcit_sinkhorn(n=sample_size, z_dim=z_dim, simulation=test, statistic="rdc",
                                                      batch_size=batch_size, n_iter=n_iters,
                                                      nstd=eps_std, z_dist=dist_z)

                    elif model == 'rcit':
                        p_value1, p_value5 = utils.rcit(n=sample_size, z_dim=z_dim, simulation=test,
                                                        batch_size=batch_size, n_iter=n_iters, nstd=eps_std,
                                                        z_dist=dist_z, x_dims=dx, y_dims=dy, a_x=al)
                    else:
                        raise ValueError('Test does not exist.')

                    test_count += 1
                    print("--- The %d'th iteration take %s seconds ---" % (test_count, (datetime.now() - start_time)))

                    if model == 'rcit':
                        p_values1.append(p_value1)
                        p_values5.append(p_value5)
                        fp = [pval < alpha / 2.0 for pval in p_values1]
                        final_result = tf.reduce_sum(tf.cast(fp, tf.float32)) / len(fp)
                        fp1 = [pval < alpha1 / 2.0 for pval in p_values5]
                        final_result1 = tf.reduce_sum(tf.cast(fp1, tf.float32)) / len(fp1)

                    else:
                        p_values.append(p_value)
                        fp = [pval < alpha / 2.0 for pval in p_values]
                        final_result = tf.reduce_sum(tf.cast(fp, tf.float32)) / len(fp)
                        fp1 = [pval < alpha1 / 2.0 for pval in p_values]
                        final_result1 = tf.reduce_sum(tf.cast(fp1, tf.float32)) / len(fp1)

                    print('Power: {} for z dimension {} and alpha {} with significance level {}'.format(final_result,
                                                                                                        z_dim, al,
                                                                                                        alpha))
                    print(
                        'Power: {} for z dimension {} and alpha {} with significance level {}'.format(final_result1,
                                                                                                      z_dim, al,
                                                                                                      alpha1))

                if model == 'rcit':
                    filename1 = '{}_z_dims{}_alpha{}_z_distribution_{}x_dim_{}sig0.05.npz'.format(test, z_dim, al,
                                                                                                  dist_z, dx)
                    np.savez(os.path.join(base_path, filename1), np.asarray(p_values5))
                    filename2 = '{}_z_dims{}_alpha{}_z_distribution_{}x_dim_{}sig0.1.npz'.format(test, z_dim, al,
                                                                                                 dist_z, dx)
                    np.savez(os.path.join(base_path, filename2), np.asarray(p_values1))
                else:
                    filename = '{}_z_dims{}_alpha{}_z_distribution_{}x_dim_{}sig0.1.npz'.format(test, z_dim, al,
                                                                                                dist_z, dx)
                    np.savez(os.path.join(base_path, filename), np.asarray(p_values))
    elif test == 'ccle':
        if model == 'dgcit':
            p_value = utils.dgcit(n=sample_size, simulation=test, batch_size=batch_size, n_iter=n_iters,
                                  train_writer=train_writer, nstd=eps_std, z_dist=dist_z, x_dims=dx, y_dims=dy,
                                  a_x=alpha_x, M=m_value, k=k_value, b=b_value, j=j_value)
            print(p_value)

        elif model == 'gcit':
            p_value = utils.gcit_sinkhorn(n=sample_size, simulation=test, statistic="rdc", batch_size=batch_size,
                                          n_iter=n_iters, nstd=eps_std, z_dist=dist_z, train_writer=train_writer)
            print(p_value)

        elif model == 'rcit':
            p_value1, p_value5 = utils.rcit(n=sample_size, simulation=test, batch_size=batch_size, n_iter=n_iters,
                                            nstd=eps_std, z_dist=dist_z, x_dims=dx, y_dims=dy)
            print(p_value1, p_value5)

    elif test == 'brain':
        p_vals = []
        for var in var_list:
            if model == 'dgcit':
                p_value = utils.dgcit(n=sample_size, simulation=test, batch_size=batch_size, n_iter=n_iters,
                                      train_writer=train_writer, nstd=eps_std, z_dist=dist_z, x_dims=dx,
                                      y_dims=dy, a_x=alpha_x, M=m_value, k=k_value, var_idx=var, b=b_value, j=j_value)
                p_vals.append(p_value)
                print('P value {} for {} dataset {} for current variable number {}'.format(p_value, test, model, var))

            elif model == 'gcit':
                p_value = utils.gcit_sinkhorn(n=sample_size, simulation=test, statistic="rdc", batch_size=batch_size,
                                              n_iter=n_iters, nstd=eps_std, z_dist=dist_z, var_idx=var)
                p_vals.append(p_value)
                print('P value {} for {} dataset {} for current variable number {}'.format(p_value, test, model, var))
            filename = '{}_dataset_{}.npz'.format(test, model)
            np.savez(os.path.join(base_path, filename), np.asarray(p_vals))
    else:
        raise ValueError('Test does not exist.')


if __name__ == '__main__':
    main()
