# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe 
tf.enable_eager_execution()

# Layer normalization can be used in Word PTB task.

class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size], 
                dtype=tf.float64, initializer=tf.ones_initializer())  # 全1初始化
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size], 
                dtype=tf.float64, initializer=tf.zeros_initializer()) # 全0初始化
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)   # (batchsize,length,1)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)  # (batchsize,length,1)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)   # (batchsize,length,dims)
        return norm_x * self.scale + self.bias
