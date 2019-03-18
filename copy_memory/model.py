import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import sys
sys.path.append("..")
from tcn import TemporalConvNet

class TCN(tf.keras.Model):
    def __init__(self, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

        self.temporalCN = TemporalConvNet(num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = tf.keras.layers.Dense(output_size, kernel_initializer=init)

    def call(self, x, training=True):
        y = self.temporalCN(x, training=training)
        y = self.linear(y)
        return y