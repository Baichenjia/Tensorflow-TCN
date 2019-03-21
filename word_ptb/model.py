import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import sys
sys.path.append("..")
from tcn import TemporalConvNet

layers = tf.keras.layers

# 目的是使 embedding 层和 最后的线性输出层共享权重
class EmbeddingSharedWeights(tf.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""
    def __init__(self, vocab_size, hidden_size):
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
    
    def build(self, _):
        with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
            self.shared_weights = tf.get_variable("weights", 
                [self.vocab_size, self.hidden_size], 
                initializer=tf.random_normal_initializer(0., self.hidden_size ** -0.5))
        self.built = True

    def call(self, x):
        """Get token embeddings of x.
        x: An int64 tensor with shape [batch_size, length]
        embeddings: float32 tensor with shape [batch_size, length, embedding_size]
        """
        with tf.name_scope("embedding"):
            embeddings = tf.gather(self.shared_weights, x)
            embeddings *= self.hidden_size ** 0.5
        return embeddings

    def linear(self, x):
        """
            Computes logits by running x through a linear layer.
            Args: x - A float32 tensor with shape [batch_size, length, hidden_size]
            Returns: float32 tensor with shape [batch_size, length, vocab_size].
        """
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])


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
        self.embedding = EmbeddingSharedWeights(self.vocab_size, embedding_dim)
        # self.embedding = layers.Embedding(self.vocab_size,
        #             embedding_dim, input_length=sequence_length)
        self.drop = layers.Dropout(rate=emb_dropout)

        # TCN
        self.temporalCN = TemporalConvNet(num_channels, 
                            kernel_size=kernel_size, dropout=dropout)

    def call(self, x, training=True):
        # Embedding
        x = self.embedding(x)
        x = self.drop(x) if training else x
        # assert x.shape == (16, self.sequence_length, self.embedding_dim)

        # TCN
        x = self.temporalCN(x, training=training)
        x = self.embedding.linear(x)
        # assert x.shape == (16, self.sequence_length, self.vocab_size)
        return x

