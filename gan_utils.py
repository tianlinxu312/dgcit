# Utilites related to Sinkhorn computations and training for TensorFlow 2.0
import tensorflow as tf


def cost_xy(x, y, scaling_coef):
    '''
    L2 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, x dims]
    :param y: y is tensor of shape [batch_size, y dims]
    :param scaling_coef: a scaling coefficient for distance between x and y
    :return: cost matrix: a matrix of size [batch_size, batch_size] where
    '''
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)
    return tf.reduce_sum((x - y)**2, -1) * scaling_coef


def benchmark_sinkhorn(x, y, scaling_coef, epsilon=1.0, L=10):
    '''
    :param x: a tensor of shape [batch_size, sequence length]
    :param y: a tensor of shape [batch_size, sequence length]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :param epsilon: (float) entropic regularity constant
    :param L: (int) number of iterations
    :return: V: (float) value of regularized optimal transport
    '''
    n_data = x.shape[0]
    # Note that batch size of x can be different from batch size of y
    m = 1.0 / tf.cast(n_data, tf.float64) * tf.ones(n_data, dtype=tf.float64)
    n = 1.0 / tf.cast(n_data, tf.float64) * tf.ones(n_data, dtype=tf.float64)
    m = tf.expand_dims(m, axis=1)
    n = tf.expand_dims(n, axis=1)

    c_xy = cost_xy(x, y, scaling_coef)  # shape: [batch_size, batch_size]

    k = tf.exp(-c_xy / epsilon) + 1e-09  # add 1e-09 to prevent numerical issues
    k_t = tf.transpose(k)

    a = tf.expand_dims(tf.ones(n_data, dtype=tf.float64), axis=1)
    b = tf.expand_dims(tf.ones(n_data, dtype=tf.float64), axis=1)

    for i in range(L):
        b = m / tf.matmul(k_t, a)  # shape: [m,]
        a = n / tf.matmul(k, b)  # shape: [m,]

    return tf.reduce_sum(a * k * tf.reshape(b, (1, -1)) * c_xy)


def benchmark_loss(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l, xp=None, yp=None):
    '''
    :param x: real data of shape [batch size, sequence length]
    :param y: fake data of shape [batch size, sequence length]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and several values for monitoring the training process)
    '''
    if yp is None:
        yp = y
    if xp is None:
        xp = x
    x = tf.reshape(x, [x.shape[0], -1])
    y = tf.reshape(y, [y.shape[0], -1])
    xp = tf.reshape(xp, [xp.shape[0], -1])
    yp = tf.reshape(yp, [yp.shape[0], -1])
    loss_xy = benchmark_sinkhorn(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xx = benchmark_sinkhorn(x, xp, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_yy = benchmark_sinkhorn(y, yp, scaling_coef, sinkhorn_eps, sinkhorn_l)

    loss = loss_xy - 0.5 * loss_xx - 0.5 * loss_yy

    return loss
