
## Sequential MNIST & Permuted Sequential MNIST

### Overview

MNIST is a handwritten digit classification dataset (Lecun et al., 1998) that is frequently used to  test deep learning models. In particular, sequential MNIST is frequently used to test a recurrent network’s ability to retain information from the distant past (see paper for references). In this task, each MNIST image (28 x 28) is presented to the model as a 784 × 1 sequence for digit classification. 


### Data
Using `tf.keras.datasets.mnist.load_data()` to load data, then process it to 

- `X_train.shape=(60000, 784, 1)`, `y_train.shape=(60000, 10)`
- `X_test.shape(10000, 784, 1) `, `y_test.shape=(10000, 10)`

Using `args.permute` to choose the "Sequential Mnist" or "Permuted Mnist".


### Result

- With Tensorflow 1.13 and eager mode.
- Training for 10 epochs and take about 1 hour.
- Using 100000 data for training, and 1000 data for evaluation.
- Using GTX 1080Ti.
- Netowork weights saved in `Permuted-weights/model_weight.h5` and `Sequential-weights/model_weight.h5`

- Sequential MNIST LOG
Epoch: 0 , Eval acc: 94.86 %
Epoch: 1 , Eval acc: 97.28999999999999 %
Epoch: 2 , Eval acc: 97.13000000000001 %
Epoch: 3 , Eval acc: 96.72 %
Epoch: 4 , Eval acc: 97.77 %
Epoch: 5 , Eval acc: 97.22 %
Epoch: 6 , Eval acc: 98.0 %
Epoch: 7 , Eval acc: 98.06 %
Epoch: 8 , Eval acc: 98.24000000000001 %
Epoch: 9 , Eval acc: 97.66 %
Epoch: 10 , Eval acc: 98.15 %
Epoch: 11 , Eval acc: 98.02 %
Epoch: 12 , Eval acc: 97.71 %
Epoch: 13 , Eval acc: 98.22 %
Epoch: 14 , Eval acc: 98.19 %
Epoch: 15 , Eval acc: 98.0 %
Epoch: 16 , Eval acc: 98.22999999999999 %
Epoch: 17 , Eval acc: 98.14 %
Epoch: 18 , Eval acc: 98.28 %
Epoch: 19 , Eval acc: 98.4 %

- Permuted MNIST LOG
Epoch: 0 , Eval acc: 92.54 %
Epoch: 1 , Eval acc: 93.4 %
Epoch: 2 , Eval acc: 94.73 %
Epoch: 3 , Eval acc: 95.28999999999999 %
Epoch: 4 , Eval acc: 95.47 %
Epoch: 5 , Eval acc: 94.89999999999999 %
Epoch: 6 , Eval acc: 95.38 %
Epoch: 7 , Eval acc: 95.91 %
Epoch: 8 , Eval acc: 95.93 %
Epoch: 9 , Eval acc: 95.89 %
Epoch: 10 , Eval acc: 96.13000000000001 %
Epoch: 11 , Eval acc: 96.1 %
Epoch: 12 , Eval acc: 96.48 %
Epoch: 13 , Eval acc: 95.88 %
Epoch: 14 , Eval acc: 95.72 %
Epoch: 15 , Eval acc: 96.2 %


### Note

- Because a TCN's receptive field depends on depth of the network and the filter size, we need to make sure these the model we use can cover the sequence length 784. 

- While this is a sequence model task, we only use the very last output (i.e. at time T=784) for the eventual classification.
