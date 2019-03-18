# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe 
tf.enable_eager_execution()

layers = tf.keras.layers

class TemporalBlock(tf.keras.Model):
	def __init__(self, dilation_rate, nb_filters, kernel_size, 
				       padding, dropout_rate=0.0): 
		super(TemporalBlock, self).__init__()
		init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
		assert padding in ['causal', 'same']

		# block1
		self.conv1 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
				                   dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
		self.batch1 = layers.BatchNormalization(axis=-1)
		self.ac1 = layers.Activation('relu')
		self.drop1 = layers.Dropout(rate=dropout_rate)
		
		# block2
		self.conv2 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
						           dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
		self.batch2 = layers.BatchNormalization(axis=-1)		
		self.ac2 = layers.Activation('relu')
		self.drop2 = layers.Dropout(rate=dropout_rate)

		# 为了防止维度不一致使用 1*1 卷积在channel处进行匹配  
		self.downsample = layers.Conv1D(filters=nb_filters, kernel_size=1, 
									    padding='same', kernel_initializer=init)
		self.ac3 = layers.Activation('relu')


	def call(self, x, training):
		prev_x = x
		x = self.conv1(x)
		x = self.batch1(x)
		x = self.ac1(x)
		x = self.drop1(x) if training else x

		x = self.conv2(x)
		x = self.batch2(x)
		x = self.ac2(x)
		x = self.drop2(x) if training else x

		if prev_x.shape[-1] != x.shape[-1]:    # match the dimention
			prev_x = self.downsample(prev_x)
		assert prev_x.shape == x.shape

		return self.ac3(prev_x + x)            # skip connection

# test:
# x = tf.convert_to_tensor(np.random.random((100, 10, 50)))   # batch, seq_len, dim
# model = TemporalBlock(dilation_rate=3, nb_filters=70, kernel_size=3, padding='causal', dropout_rate=0.1)
# y = model(x)
# print(y.shape)


class TemporalConvNet(tf.keras.Model):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2):
    	# num_channels is a list contains hidden sizes of Conv1D
        super(TemporalConvNet, self).__init__()
        assert isinstance(num_channels, list)

        # 初始化 model
        model = tf.keras.Sequential()

        # The model contains "num_levels" TemporalBlock
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i                  # exponential growth
            model.add(TemporalBlock(dilation_rate, num_channels[i], kernel_size, 
                      padding='causal', dropout_rate=dropout))
        self.network = model

    def call(self, x, training):
        return self.network(x, training=training)

# test:
# x = tf.convert_to_tensor(np.random.random((100, 10, 50)))   # batch, seq_len, dim
# model = TemporalConvNet(num_channels=[70, 80])
# y = model(x)
# print(y.shape)

