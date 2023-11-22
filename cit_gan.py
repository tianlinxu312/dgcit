import tensorflow as tf
import logging

logging.getLogger('tensorflow').disabled = True
tf.keras.backend.set_floatx('float32')
tf.random.set_seed(42)


class WGanGenerator(tf.keras.Model):
    '''
    class for WGAN generator
    Args:
        inputs, noise and confounding factor [v, z], of shape [batch size, z_dims + v_dims]
    return:
       fake samples of shape [batch size, x_dims]
    '''
    def __init__(self, n_samples, z_dims, h_dims, v_dims, x_dims, batch_size):
        super(WGanGenerator, self).__init__()
        self.n_samples = n_samples
        self.hidden_dims = h_dims
        self.batch_size = batch_size
        self.dz = z_dims
        self.dx = x_dims
        self.dv = v_dims

        self.input_dim = self.dz + self.dv
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, self.dx]

        self.w1 = self.xavier_var_creator(self.input_shape1)
        self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

        self.w3 = self.xavier_var_creator(self.input_shape3)
        self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, inputs, training=None, mask=None):
        # inputs are concatenations of z and v
        z = tf.reshape(tensor=inputs, shape=[-1, self.input_dim])
        h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
        # h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        out = tf.math.sigmoid(tf.matmul(h1, self.w3) + self.b3)
        return out


class WGanDiscriminator(tf.keras.Model):
    '''
    class for WGAN discriminator
    Args:
        inputss: real and fake samples of shape [batch size, x_dims]
    return:
       features f_x of shape [batch size, features]
    '''
    def __init__(self, n_samples, z_dims, h_dims, v_dims, batch_size):
        super(WGanDiscriminator, self).__init__()
        self.n_samples = n_samples
        self.hidden_dims = h_dims
        self.batch_size = batch_size

        self.input_dim = z_dims + v_dims
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, 1]

        self.w1 = self.xavier_var_creator(self.input_shape1)
        self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

        self.w3 = self.xavier_var_creator(self.input_shape3)
        self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, inputs, training=None, mask=None):
        # inputs are concatenations of z and v
        z = tf.reshape(tensor=inputs, shape=[self.batch_size, -1])
        z = tf.cast(z, tf.float64)
        h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
        # h2 = tf.nn.sigmoid(tf.matmul(h1, self.w2) + self.b2)
        # out = tf.nn.sigmoid(tf.matmul(h1, self.w3) + self.b3)
        out = tf.matmul(h1, self.w3) + self.b3
        return out


class MINEDiscriminator(tf.keras.layers.Layer):
    '''
    class for MINE discriminator for benchmark GCIT
    '''

    def __init__(self, in_dims, output_activation='linear'):
        super(MINEDiscriminator, self).__init__()
        self.output_activation = output_activation
        self.input_dim = in_dims

        self.w1a = self.xavier_var_creator()
        self.w1b = self.xavier_var_creator()
        self.b1 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

        self.w2a = self.xavier_var_creator()
        self.w2b = self.xavier_var_creator()
        self.b2 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

        self.w3 = self.xavier_var_creator()
        self.b3 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

    def xavier_var_creator(self):
        xavier_stddev = 1.0 / tf.sqrt(self.input_dim / 2.0)
        init = tf.random.normal(shape=[self.input_dim, ], mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(self.input_dim, ), trainable=True)
        return var

    def mine_layer(self, x, x_hat, wa, wb, b):
        return tf.math.tanh(wa * x + wb * x_hat + b)

    def call(self, x, x_hat):
        h1 = self.mine_layer(x, x_hat, self.w1a, self.w1b, self.b1)
        h2 = self.mine_layer(x, x_hat, self.w2a, self.w2b, self.b2)
        out = self.w3 * (h1 + h2) + self.b3
        return out, tf.exp(out)


class CharacteristicFunction:
    '''
    class to construct a function that represents the characteristic function
    '''

    def __init__(self, size, x_dims, z_dims, test_size):
        self.n_samples = size
        self.hidden_dims = 20
        self.test_size = test_size

        self.input_dim = z_dims + x_dims
        self.z_dims = z_dims
        self.x_dims = x_dims
        self.input_shape1x = [self.x_dims, self.hidden_dims]
        self.input_shape1z = [self.z_dims, self.hidden_dims]
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, 1]

        self.w1x = self.xavier_var_creator(self.input_shape1x)
        self.b1 = tf.squeeze(self.xavier_var_creator([self.hidden_dims, 1]))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        xavier_stddev = tf.sqrt(2.0 / (input_shape[0]))
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def update(self):
        self.w1x = self.xavier_var_creator(self.input_shape1x)
        self.b1 = tf.squeeze(self.xavier_var_creator([self.hidden_dims, 1]))
        self.w2 = self.xavier_var_creator(self.input_shape2)

    def call(self, x, z):
        # inputs are concatenations of z and v
        x = tf.reshape(tensor=x, shape=[self.test_size, -1, self.x_dims])
        z = tf.reshape(tensor=z, shape=[self.test_size, -1, self.z_dims])
        # we asssume parameter b for z to be 0
        h1 = tf.nn.sigmoid(tf.matmul(x, self.w1x) + self.b1)
        out = tf.nn.sigmoid(tf.matmul(h1, self.w2))
        return out

