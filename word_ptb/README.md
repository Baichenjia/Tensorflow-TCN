
## Word-level Language Modeling

- **This task can not achieve the performance described in the original paper, and needs to be further solved.**

### Overview

In word-level language modeling tasks, each element of the sequence is a word, where the model is expected to predict the next incoming word in the text. We evaluate the temporal convolutional network as a word-level language model on PennTreebank

- The learning rate should de halved when the eval loss no longer falling.

### Data

- **PennTreebank**: A frequently studied, but still relatively
small language corpus. When used as a word-level language corpus,
PTB contains 888K words for training, 70K for validation,
and 79K for testing, with a vocabulary size of 10K.


### Result

- With Tensorflow 1.13 and eager mode.
- Training for 10 epochs and take about 1 hour.
- Using 100000 data for training, and 1000 data for evaluation.
- Using GTX 1080Ti.
- Netowork weights saved in `weights/model_weight.h5` 

- LOG




### Note

- Just like in a recurrent network implementation where it is common to repackage 
hidden units when a new sequence begins, we pass into TCN a sequence `T` consisting 
of two parts: 1) effective history `L1`, and 2) valid sequence `L2`:

```
Sequence [---------T--------->] = [--L1--> ------L2------>]
```

In the forward pass, the whole sequence is passed into TCN, but only the `L2` portion is used for training. This ensures that the training data are also provided with sufficient history. The size of `T` and `L2` can be adjusted via flags `seq_len` and `validseqlen`. A similar setting was used in character-level language modeling experiments.
