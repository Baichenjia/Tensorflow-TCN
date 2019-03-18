# -*- coding: utf-8 -*-

from Datasets import Datasets
import argparse
from model import TCN
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Sequence Modeling - The Word PTB')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size (default: 16)')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (default: 0.0)')
parser.add_argument('--emb_dropout', type=float, default=0.25, help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=0.4, help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=3, help='kernel size (default: 7)')
parser.add_argument('--emsize', type=int, default=600, help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=4, help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=80, help='sequence length (default: 400)')
parser.add_argument('--validseqlen', type=int, default=40,help='valid sequence length (default: 40)')
parser.add_argument('--lr', type=float, default=2., help='initial learning rate (default: 4e-3)')
parser.add_argument('--nhid', type=int, default=600, help='number of hidden units per layer (default: 30)')
args = parser.parse_args()

# Parameter
batch_size = args.batch_size
seq_len = args.seq_len
epochs = args.epochs
clip = args.clip
lr = args.lr
print("Args:\n", args)

# Dataset
corpus = Datasets()
train_data = corpus._divide_into_batches(corpus.train, batch_size=batch_size)
eval_data = corpus._divide_into_batches(corpus.eval, batch_size=batch_size)

print("train_data.shape:", train_data.shape)
print("eval_data.shape:", eval_data.shape)


# Build model
print("Building model...")
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout
emsize = args.emsize
emb_dropout = args.emb_dropout
vocab_size = corpus.vocab_size
model = TCN(output_size=vocab_size, 
            num_channels=channel_sizes, 
            kernel_size=kernel_size, 
            dropout=dropout,
            embedding_dim=emsize,
            sequence_length=seq_len,
            emb_dropout=emb_dropout)

# 优化
learning_rate = tf.Variable(lr, name="learning_rate")
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# 
best_loss = None
for epoch in range(epochs):
    
    # range把原来的sequence截成多部分，分别训练
    for batch, i in enumerate(range(0, train_data.shape[1]-args.seq_len, args.validseqlen)):
        # 获取数据
        train_seq = tf.convert_to_tensor(train_data[:, i:i+args.seq_len])
        train_target = tf.convert_to_tensor(train_data[:, i+1: i+1+args.seq_len])
        assert train_seq.shape == train_target.shape == (batch_size, args.seq_len)
 
        with tf.GradientTape() as tape:
            outputs = model(train_seq, training=True)
            assert outputs.shape == (batch_size, args.seq_len, vocab_size)
            #  Discard the effective history part
            eff_history = args.seq_len - args.validseqlen
            labels = tf.reshape(train_target[:, eff_history:], [-1])
            logits = tf.reshape(outputs[:, eff_history:, :], [-1, vocab_size])
            # loss
            loss_np = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            assert loss_np.shape == (batch_size * args.validseqlen, )
            loss = tf.reduce_mean(loss_np)
        
        if batch % 100 == 0:
            print("Batch:", batch, ", Train loss:", loss.numpy())
    
        gradient = tape.gradient(loss, model.trainable_variables)
        if clip != -1:
            gradient, _ = tf.clip_by_global_norm(gradient, clip)  
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    # Eval
    eval_losses = []

    for batch, i in enumerate(range(0, eval_data.shape[1]-args.seq_len, args.validseqlen)):
        # 获取数据
        eval_seq = tf.convert_to_tensor(eval_data[:, i:i + args.seq_len])
        eval_target = tf.convert_to_tensor(eval_data[:, i + 1:i + 1 + args.seq_len])
        assert eval_seq.shape == eval_target.shape == (batch_size, args.seq_len)
        # 
        outputs = model(eval_seq, training=False)
        assert outputs.shape == (batch_size, args.seq_len, vocab_size)
        #  Discard the effective history part
        eff_history = args.seq_len - args.validseqlen
        eval_loss_np = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=eval_target[:, eff_history:], logits=outputs[:, eff_history:, :])
        assert eval_loss_np.shape == (batch_size, args.validseqlen)
        eval_loss = tf.reduce_mean(eval_loss_np)
        eval_losses.append(eval_loss.numpy())

    eval_loss = np.mean(eval_losses)
    print("Epoch:", epoch, ", Eval loss:", eval_loss, ", Eval perplexity:", np.exp(eval_loss))
    
    # 当验证集损失不再下降时调整学习率
    if not best_loss or eval_loss < best_loss:
        model.save_weights("weights/model_weight.h5")
        best_loss = eval_loss
    else:   # 缩小学习率
        learning_rate.assign(learning_rate / 2.0)
        print("changing learning rate to", learning_rate.numpy())
    print("-------\n\n")
    
