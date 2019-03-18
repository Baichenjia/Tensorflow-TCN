import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import sys
sys.path.append("..")
from tcn import TemporalConvNet

layers = tf.keras.layers

class TCN(tf.keras.Model):
    def __init__(self, 
                 output_size, num_channels, kernel_size, dropout,
                 embedding_dim, 
                 sequence_length,
                 emb_dropout=0.1):        # embedding的dropout
                 
        super(TCN, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.vocab_size = output_size         # vocab_size equals to output_size

        # Embedding层
        self.embedding = layers.Embedding(self.vocab_size,
                    embedding_dim, input_length=sequence_length)
        self.drop1 = layers.Dropout(rate=emb_dropout)

        # TCN
        self.temporalCN = TemporalConvNet(num_channels, 
                            kernel_size=kernel_size, dropout=dropout)

        # Linear
        self.drop2 = layers.Dropout(rate=emb_dropout)
        self.linear = layers.Dense(self.vocab_size)  # output size is vocab_size
        

    def call(self, x, training=True):
        # Embedding
        # assert x.shape == (16, self.sequence_length)
        x = self.embedding(x)
        x = self.drop1(x) if training else x
        # assert x.shape == (16, self.sequence_length, self.embedding_dim)

        # TCN
        x = self.temporalCN(x, training=training)
        x = self.drop2(x) if training else x
        x = self.linear(x)
        # assert x.shape == (16, self.sequence_length, self.vocab_size)
        return x

