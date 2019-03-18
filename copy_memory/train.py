import argparse
from model import TCN
from utils import data_generator
import tensorflow as tf
tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Sequence Modeling - The Copy Memory')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size (default: 32)')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=1.0, help='gradient clip, -1 means no clip (default: 1.0)')
parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit (default: 50)')
parser.add_argument('--iters', type=int, default=100, help='number of iters per epoch (default: 100)')
parser.add_argument('--ksize', type=int, default=8, help='kernel size (default: 8)')
parser.add_argument('--levels', type=int, default=8, help='# of levels (default: 8)')
parser.add_argument('--blank_len', type=int, default=1000, metavar='N',help='The size of the blank (i.e. T) (default: 1000)')
parser.add_argument('--seq_len', type=int, default=10, help='sequence length')
parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate (default: 5e-4)')
parser.add_argument('--nhid', type=int, default=10, help='number of hidden units per layer (default: 10)')
args = parser.parse_args()

# Parameter
batch_size = args.batch_size
seq_len = args.seq_len
epochs = args.epochs
clip = args.clip
lr = args.lr
T = args.blank_len
n_steps = T + 2 * seq_len
n_classes = 10
n_train = 10000
n_test = 1000
print("Args:\n", args)

# Dataset
print("Producing data...")
train_data, train_labels = data_generator(T, seq_len, n_train)
print("train_data.shape:", train_data.shape, ", train_labels.shape:", train_labels.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(n_train).batch(batch_size)

test_data, test_labels = data_generator(T, seq_len, n_test)
test_data, test_labels = tf.convert_to_tensor(test_data), tf.convert_to_tensor(test_labels)

# Build model
# Note: We use a very simple setting here (assuming all levels have the same # of channels.
print("Building model...")
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
model = TCN(n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

# Optimizer
optimizer = tf.train.RMSPropOptimizer(lr)

# RUN 
for epoch in range(epochs):
    for batch, (train_x, train_y) in enumerate(train_dataset):
        # print(train_x.shape, train_y.shape)
        # assert train_x.shape == (batch_size, n_steps, 1)
        # assert train_y.shape == (batch_size, n_steps)
        # loss
        with tf.GradientTape() as tape:
            y = model(train_x, training=True)
            # assert y.shape == (batch_size, n_steps, 10)
            loss_np = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_y, logits=y)
            # assert loss_np.shape == (batch_size, n_steps)
            loss = tf.reduce_mean(loss_np)
        
        # gradient
        gradient = tape.gradient(loss, model.trainable_variables)
        if clip != -1:
            gradient, _ = tf.clip_by_global_norm(gradient, clip)  
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        if batch % 100 == 0:
            print("Batch:", batch, ", Train loss:", loss.numpy())
    # Eval
    eval_logits = model(test_data, training=False)
    eval_loss_list = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=test_labels, logits=eval_logits)
    # assert eval_loss_list.shape == (batch_size, n_steps)
    eval_loss = tf.reduce_mean(eval_loss_list)
    print("Epoch:", epoch, ", Eval loss:", eval_loss.numpy(), "\n---\n")
    
    # Save
    model.save_weights("weights/model_weight.h5")


