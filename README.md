# Tensorflow TCN

**The explanation and graph in this README.md refers to [Keras-TCN](https://github.com/philipperemy/keras-tcn).**

*Temporal Convolutional Network* with tensorflow 1.13 (eager execution)

   * [Tensorflow TCN](#tensorflow-tcn)
      * [Why Temporal Convolutional Network?](#why-temporal-convolutional-network)
      * [API](#api)
         * [Arguments](#arguments)
         * [Input shape](#input-shape)
         * [Output shape](#output-shape)
         * [Receptive field](#receptive-field)
      * [Run](#run)
      * [Tasks](#tasks)
         * [Adding Task](#adding-task)
         * [Copy Memory Task](#copy-memory-task)
         * [Sequential & Permuted MNIST](#sequential-mnist)
         * [PennTreebank](#penntreebank)
      * [References](#references)

## Why Temporal Convolutional Network?

- TCNs exhibit longer memory than recurrent architectures with the same capacity.
- Constantly performs better than LSTM/GRU architectures on a vast range of tasks (Seq. MNIST, Adding Problem, Copy Memory, Word-level PTB...).
- Parallelism, flexible receptive field size, stable gradients, low memory requirements for training, variable length inputs...

<p align="center">
  <img src="misc/Dilated_Conv.png">
<b>Visualization of a stack of dilated causal convolutional layers (Wavenet, 2016)</b><br><br>
</p>

## API

### Arguments
`tcn = TemporalConvNet(num_channels, kernel_size, dropout)`

- `num_channels`: list. For example, if `num_channels=[30,40,50,60,70,80]`, the temporal convolution model has 6 levels, the `dilation_rate` of each level is $$[2^0,2^1,2^2,2^3,2^4,2^5]$$, and filters of each level are `30,40,50,60,70,80`.
- `kernel_size`: Integer. The size of the kernel to use in each convolutional layer.
- `dilations`: List. A dilation list. Example is: [1, 2, 4, 8, 16, 32, 64].
- `dropout`: Float between 0 and 1. Fraction of the input units to drop. The dropout layers is activated in training, and deactivated in testing. Using `y = tcn(x, training=True/False)` to control.


### Input shape

3D tensor with shape `(batch_size, timesteps, input_dim)`.

### Output shape

It depends on the task (cf. below for examples):

- Regression (Many to one) e.g. adding problem
- Classification (Many to many) e.g. copy memory task
- Classification (Many to one) e.g. sequential mnist task

### Receptive field

- Receptive field = **nb_stacks_of_residuals_blocks * kernel_size * last_dilation**.
- If a TCN has only one stack of residual blocks with a kernel size of 2 and dilations [1, 2, 4, 8], its receptive field is 2 * 1 * 8 = 16. The image below illustrates it:

<p align="center">
  <img src="https://user-images.githubusercontent.com/40159126/41830054-10e56fda-7871-11e8-8591-4fa46680c17f.png">
  <b>ks = 2, dilations = [1, 2, 4, 8], 1 block</b><br><br>
</p>

- If the TCN has now 2 stacks of residual blocks, wou would get the situation below, that is, an increase in the receptive field to 32:

<p align="center">
  <img src="https://user-images.githubusercontent.com/40159126/41830618-a8f82a8a-7874-11e8-9d4f-2ebb70a31465.jpg">
  <b>ks = 2, dilations = [1, 2, 4, 8], 2 blocks</b><br><br>
</p>


- If we increased the number of stacks to 3, the size of the receptive field would increase again, such as below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/40159126/41830628-ae6e73d4-7874-11e8-8ecd-cea37efa33f1.jpg">
  <b>ks = 2, dilations = [1, 2, 4, 8], 3 blocks</b><br><br>
</p>



## Run

Each task has a separate folder. Enter each folder you can usually find `utils.py`, `model.py` and `train.py`. The `utils.py` file generate data, and `model.py` build the model. You should run `train.py` to train. The parameters in `train.py` are set by `argparse`. The pre trained models are saved in `weights/`.

```bash
cd adding_problem/
python train.py # run adding problem task

cd copy_memory/
python train.py # run copy memory task

cd mnist_pixel/
python train.py # run sequential mnist pixel task

cd word_ptb/
python train.py # run PennTreebank word-level language model task
```
The training detail of each task is in README.md in each folder.

## Tasks

### Adding Task

The task consists of feeding a large array of decimal numbers to the network, along with a boolean array of the same length. The objective is to sum the two decimals where the boolean array contain the two 1s.

<p align="center">
  <img src="misc/Adding_Task.png">
  <b>Adding Problem Task</b><br><br>
</p>


### Copy Memory Task

The copy memory consists of a very large array:
- At the beginning, there's the vector x of length N. This is the vector to copy.
- At the end, N+1 9s are present. The first 9 is seen as a delimiter.
- In the middle, only 0s are there.

The idea is to copy the content of the vector x to the end of the large array. The task is made sufficiently complex by increasing the number of 0s in the middle.

<p align="center">
  <img src="misc/Copy_Memory_Task.png">
  <b>Copy Memory Task</b><br><br>
</p>


### Sequential MNIST

The idea here is to consider MNIST images as 1-D sequences and feed them to the network. This task is particularly hard because sequences are `28*28 = 784` elements. In order to classify correctly, the network has to remember all the sequence. Usual LSTM are unable to perform well on this task.

<p align="center">
  <img src="misc/Sequential_MNIST_Task.png">
  <b>Sequential MNIST</b><br><br>
</p>

### PennTreebank

In word-level language modeling tasks, each element of the sequence is a word, where the model is expected to predict the next incoming word in the text. We evaluate the temporal convolutional network as a word-level language model on PennTreebank.

## References
- https://github.com/philipperemy/keras-tcn (TCN for keras)
- https://github.com/locuslab/TCN/ (TCN for Pytorch)
- https://arxiv.org/pdf/1803.01271.pdf (An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling)
- https://arxiv.org/pdf/1609.03499.pdf (Wavenet paper)