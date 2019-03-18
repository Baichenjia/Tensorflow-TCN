import argparse
from model import TCN
from utils import data_generator
import tensorflow as tf
tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1, help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=8, help='kernel size (default: 8)')
parser.add_argument('--levels', type=int, default=9, help='# of levels (default: 9)')
parser.add_argument('--seq_len', type=int, default=600, help='sequence length (default: 600)')
parser.add_argument('--lr', type=float, default=2e-3, help='initial learning rate (default: 2e-3)')
parser.add_argument('--nhid', type=int, default=30, help='number of hidden units per layer (default: 30)')
args = parser.parse_args()

# Parameter
n_classes = 1
batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs
clip = args.clip
lr = args.lr
print("Args:\n", args)

# dataset
print("Producing data...")
train_data, train_labels = data_generator(100000, seq_length)
print("train_data.shape:", train_data.shape, ", train_labels.shape:", train_labels.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(50000).batch(batch_size)

test_data, test_labels = data_generator(1000, seq_length)
test_data, test_labels = tf.convert_to_tensor(test_data), tf.convert_to_tensor(test_labels)

# build model
# Note: We use a very simple setting here (assuming all levels have the same # of channels.
print("Building model...")
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
model = TCN(n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

# Optimizer
optimizer = tf.train.AdamOptimizer(lr)

# Run 
for epoch in range(epochs):
    for batch, (train_x, train_y) in enumerate(train_dataset):
        # loss
        with tf.GradientTape() as tape:
            y = model(train_x, training=True)  # y.shape == (batch_size, 1)
            loss = tf.reduce_mean(tf.square(y - train_y))
        # gradient
        gradient = tape.gradient(loss, model.trainable_variables)
        if clip != -1:
            gradient, _ = tf.clip_by_global_norm(gradient, clip)  
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        if batch % 100 == 0:
            print("Batch:", batch, ", Train loss:", loss.numpy())
    # Eval
    eval_loss = tf.reduce_mean(tf.square(test_labels - model(test_data, training=False)))
    print("Epoch:", epoch, ", Eval loss:", eval_loss.numpy(), "\n---\n")
    # save
    model.save_weights("weights/model_weight.h5")


