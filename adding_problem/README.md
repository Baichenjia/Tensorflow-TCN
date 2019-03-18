
## The Adding Problem

### Overview

In this task, each input consists of a length-T sequence of depth 2, with all values randomly chosen randomly in [0, 1] in dimension 1. The second dimension consists of all zeros except for two elements, which are marked by 1. The objective is to sum the two random values whose second dimensions are marked by 1. One can think of this as computing the dot product of two dimensions.

Simply predicting the sum to be 1 should give an MSE of about 0.1767. 

### Result

- With Tensorflow 1.13 and eager mode.
- Training for 10 epochs and take about 1 hour.
- Using 100000 data for training, and 1000 data for evaluation.
- Using GTX 1080Ti.
- Netowork weights saved in `weights/model_weight.h5`

- LOG
Epoch: 0 , Eval loss: 0.000813658985754942 
Epoch: 1 , Eval loss: 0.0007132544483624387 
Epoch: 2 , Eval loss: 0.0003603468718604334 
Epoch: 3 , Eval loss: 0.00014354334453869918 
Epoch: 4 , Eval loss: 5.22902180092326e-05 
Epoch: 5 , Eval loss: 0.0002829899716703299 
Epoch: 6 , Eval loss: 0.0003108931334010413 
Epoch: 7 , Eval loss: 0.00024285050328540635 
Epoch: 8 , Eval loss: 2.6982756251534838e-05 
Epoch: 9 , Eval loss: 0.00010913188533177809 

### Data Generation

See `data_generator` in `utils.py`.

### Note

Because a TCN's receptive field depends on depth of the network and the filter size, we need
to make sure these the model we use can cover the sequence length T. 