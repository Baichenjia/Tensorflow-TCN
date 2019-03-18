# -*- coding: utf-8 -*-

TRAIN_PATH = "data/ptb.train.txt"
VALID_PATH = "data/ptb.valid.txt"

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

class Datasets():

    def __init__(self):
        # load the Penn Treebank dataset
        self.word2idx = {}
        self.idx2word = []

        # 存储的是所有数据构成的一个Numpy，句子之间以<eos>分隔
        self.train = self.tokenize(TRAIN_PATH)
        self.eval = self.tokenize(VALID_PATH)

        self.vocab_size = len(self.idx2word)

    def tokenize(self, path):

        total_len = 0
        with tf.gfile.Open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word not in self.idx2word:
                        self.idx2word.append(word)
                        self.word2idx[word] = len(self.idx2word) - 1
                total_len += len(words)
        print("total_len =", total_len)

        token = np.zeros(total_len, dtype=np.int64)
        i = 0
        with tf.gfile.Open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    token[i] = self.word2idx[word]
                    i += 1
        return token

    def _divide_into_batches(self, data, batch_size):
        # 变成训练的格式，训练数据 shape=(batch_size, n_batches)
        nbatch = data.shape[0] // batch_size 
        print("nbatch:", nbatch)
        data = data[:nbatch * batch_size]
        data = data.reshape(batch_size, -1).astype(np.int32)
        return data

# if __name__ == '__main__':
#     dataset = Datasets()
#     print(dataset.vocab_size)
#     dataset._divide_into_batches(dataset.train, batch_size=16)
