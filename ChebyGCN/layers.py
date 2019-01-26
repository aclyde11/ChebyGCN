## Based on
## Michael Defferrard, Xavier Bresson, Pierre Vandergheynst, Convolutional Neural Networks on Graphs with Fast Localized
## Spectral Filtering, Neural Information Processing Systems (NIPS), 2016.

import tensorflow as tf

from keras import backend as K
from keras.layers import Layer
import numpy as np
import scipy


class GraphConvolution(Layer):

    def __init__(self, filter_size, pooling, poly_k, L=[], bias_per_vertex=False,
                 pool_type='max', activation=None, **kwargs):

#        self.F_0 = input_channels
        self.F_1 = filter_size
#        self.M_0 = input_nodes
        self.K = poly_k
        self.p_1 = pooling
        self.L = L
        self.output_dim = ()
        self.bias_per_vertex = bias_per_vertex

        if activation is None:
            self.activation = tf.nn.relu
        else:
            self.activation = activation

        if pool_type == 'max':
            self.poolf = tf.nn.max_pool
        elif pool_type == 'average' or pool_type == 'avg':
            self.poolf = tf.nn.avg_pool
        else:
            raise ValueError('pool_type not set to "max" or "avg"')

        super(GraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.F_0 = input_shape[2]
        self.M_0 = input_shape[1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.F_0 * self.K, self.F_1),
                                      initializer='uniform',
                                      trainable=True)

        if self.bias_per_vertex:
            self.bias = self.add_weight(name='bias', shape=(1, self.M_0, self.F_1), initializer='uniform',
                                        trainable=True)
        else:
            self.bias = self.add_weight(name='bias', shape=(1, 1, self.F_1), initializer='uniform', trainable=True)
        print "OUTPUT: "
        print (input_shape[0], self.M_0 / self.p_1, self.F_1)
        super(GraphConvolution, self).build(input_shape)  # Be sure to call this at the end

    def rescale_L(self, L, lmax=2):
        """Rescale the Laplacian eigenvalues in [-1,1]."""
        M, M = L.shape
        I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
        L /= lmax / 2
        L -= I
        return L

    def chebyshev5(self, x, L, Fout, K_coeff):
        N, M, Fin = K.int_shape(x)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = self.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K_coeff x M x Fin*N

        if K_coeff > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K_coeff):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K_coeff, M, Fin, -1])  # K_coeff x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K_coeff
        x = tf.reshape(x, [-1, Fin * K_coeff])  # N*M x Fin*K_coeff
        # Filter: Fin*Fout filters of order K_coeff, i.e. one filterbank per feature pair.
        #  W = self._weight_variable([Fin*K_coeff, Fout], regularization=False)
        x = tf.matmul(x, self.kernel)  # N*M x Fout
        x = tf.reshape(x, [-1, M, Fout])  # N x M x Fout
        return x

    def pool(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = self.poolf(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            # tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def call(self, x):
        if len(x.get_shape()) != 3:
            x = tf.expand_dims(x, 2)
        x = self.chebyshev5(x, self.L, self.F_1, self.K)
        x = self.activation(x + self.bias)
        x = self.pool(x, self.p_1)
        print x.get_shape()
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.M_0 / self.p_1, self.F_1)
