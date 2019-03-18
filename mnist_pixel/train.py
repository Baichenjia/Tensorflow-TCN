import argparse
from model import TCN
from utils import data_generator
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Sequence Modeling - The Mnist Pixel Problem')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size (default: 32)')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1, help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7, help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8, help='# of levels (default: 8)')
parser.add_argument('--lr', type=float, default=2e-3, help='initial learning rate (default: 2e-3)')
parser.add_argument('--nhid', type=int, default=25, help='number of hidden units per layer (default: 25)')
parser.add_argument('--permute', type=bool, default=True, help='use permuted MNIST (default: false)')
args = parser.parse_args()

# Parameter
n_classes = 10
batch_size = args.batch_size
input_channels = 1
seq_length = int(28*28/input_channels)
epochs = args.epochs
clip = args.clip
lr = args.lr
print("Args:\n", args)

# dataset
print("Producing data...")
# shape=(60000, 784, 1) (60000, 10),  (10000, 784, 1) (10000, 10)
(x_train, y_train), (x_test, y_test) = data_generator(permute=args.permute)
print("train_data.shape:", x_train.shape, ", train_labels.shape:", y_train.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)

test_data, test_labels = tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test)

# build model
# Note: We use a very simple setting here (assuming all levels have the same # of channels.
print("Building model...")
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
model = TCN(n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

# optimizer
optimizer = tf.train.AdamOptimizer(lr)

# run 
for epoch in range(epochs):
    for batch, (train_x, train_y) in enumerate(train_dataset):
        # assert train_x.shape == (batch_size, seq_length, 1)
        # assert train_y.shape == (batch_size,)
        # loss
        with tf.GradientTape() as tape:
            y = model(train_x, training=True)
            # assert y.shape == (batch_size, 10)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_y, logits=y))
        # gradient
        gradient = tape.gradient(loss, model.trainable_variables)
        if clip != -1:
            gradient, _ = tf.clip_by_global_norm(gradient, clip)  
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        if batch % 100 == 0:
            print("Batch:", batch, ", Train loss:", loss.numpy())
        
    # Eval Acc
    eval_labels =  model(test_data, training=False)
    eval_acc = np.mean(np.argmax(eval_labels.numpy(), axis=1) == test_labels.numpy())
    print("Epoch:", epoch, ", Eval acc:", eval_acc*100, "%\n---\n")

    # Save
    if not args.permute:
        model.save_weights("Sequential-weights/model_weight.h5")
    else:
        model.save_weights("Permuted-weights/model_weight.h5")


# Eval Acc

# print(test_data.shape, test_labels.shape)
# test_data = test_data[:100, :, :]
# test_labels = test_labels[:100]

# print("Before:")
# eval_labels =  model(test_data, training=False)
# eval_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=test_labels, logits=eval_labels))
# print("eval_loss:", eval_loss)

# print("\nAfter:")
# model.load_weights("weights/model_weight.h5")
# eval_labels =  model(test_data, training=False)
# eval_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=test_labels, logits=eval_labels))
# print("eval_loss:", eval_loss)

# print(eval_labels.numpy())
# print(eval_labels.shape)

# print("\n-----")
# print(np.argmax(eval_labels.numpy(), axis=1))
# print(test_labels.numpy())

# eval_acc = np.mean(np.argmax(eval_labels.numpy(), axis=1) == test_labels.numpy())
# print("Eval acc:", eval_acc*100, "%\n---\n")
      

