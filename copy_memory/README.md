
## Copying Memory Task

### Overview

In this task, each input sequence has length T+20. The first 10 values are chosen randomly among the digits 1-8, with the rest being all zeros, except for the last 11 entries that are filled with the digit ‘9’ (the first ‘9’ is a delimiter). The goal is to generate an output of same length that is zero everywhere, except the last 10 values after the delimiter, where the model is expected to repeat the 10 values it encountered at the start of the input.

### Result

- With Tensorflow 1.13 and eager mode.
- Training for 10 epochs and take about 1 hour.
- Using 100000 data for training, and 1000 data for evaluation.
- Using GTX 1080Ti.
- Netowork weights saved in `weights/model_weight.h5`

- LOG


Epoch: 0 , Eval loss: 0.032604238335429206 
Epoch: 1 , Eval loss: 0.020700338041302873 
Epoch: 2 , Eval loss: 0.019680310168485968 
Epoch: 3 , Eval loss: 0.015000858444147133 
Epoch: 4 , Eval loss: 0.010173443584815055 
Epoch: 5 , Eval loss: 0.005700644494324414 
Epoch: 6 , Eval loss: 0.005150207891503893 
Epoch: 7 , Eval loss: 0.004951838016467049 
Epoch: 8 , Eval loss: 0.009987736097279602 
Epoch: 9 , Eval loss: 0.006308756551210753 
Epoch: 10 , Eval loss: 0.0010778146339530616 
Epoch: 11 , Eval loss: 0.005314206800982766 
Epoch: 12 , Eval loss: 0.004991700869702015 
Epoch: 13 , Eval loss: 0.03572527119360213 
Epoch: 14 , Eval loss: 0.00019343361915390617 
Epoch: 15 , Eval loss: 0.010225045236323383 
Epoch: 16 , Eval loss: 0.00386027262917956 
Epoch: 17 , Eval loss: 0.00012344072127766197 
Epoch: 18 , Eval loss: 5.709481540974582e-05 
Epoch: 19 , Eval loss: 4.568465843339097e-05 
Epoch: 20 , Eval loss: 2.5620241496367407e-05 
Epoch: 21 , Eval loss: 2.288924360678626e-05 
Epoch: 22 , Eval loss: 0.018589317467219785 
Epoch: 23 , Eval loss: 1.809761427546778e-05 
Epoch: 24 , Eval loss: 5.449527173681429e-05 
Epoch: 25 , Eval loss: 2.24093048244839e-05 
Epoch: 26 , Eval loss: 0.0007487799625961211 
Epoch: 27 , Eval loss: 4.496355157952741e-05 
Epoch: 28 , Eval loss: 2.594544378875287e-05 
Epoch: 29 , Eval loss: 4.047355657361625e-05 
Epoch: 30 , Eval loss: 3.252689501081529e-05 
Epoch: 31 , Eval loss: 1.0559955733250008e-05 
Epoch: 32 , Eval loss: 9.485499666082105e-06 
Epoch: 33 , Eval loss: 6.933585311322369e-06 
Epoch: 34 , Eval loss: 3.4169526349911936e-05 
Epoch: 35 , Eval loss: 1.2170086162198213e-05 
Epoch: 36 , Eval loss: 6.287420564321654e-06 
Epoch: 37 , Eval loss: 8.799355339669319e-06 
Epoch: 38 , Eval loss: 7.124628394670952e-06 
Epoch: 39 , Eval loss: 8.288686373886261e-06 
Epoch: 40 , Eval loss: 6.745277902638558e-06 
Epoch: 41 , Eval loss: 2.4829093642346727e-06 
Epoch: 42 , Eval loss: 1.7841864086249153e-06 
Epoch: 43 , Eval loss: 7.4030886712981755e-06 
Epoch: 44 , Eval loss: 0.04310435864654208 
Epoch: 45 , Eval loss: 1.1521112303697698e-05 
Epoch: 46 , Eval loss: 0.000623339198711661 
Epoch: 47 , Eval loss: 1.1846899383127256e-06 
Epoch: 48 , Eval loss: 6.346768470445379e-06 
Epoch: 49 , Eval loss: 9.008760160348695e-06 


### Data Generation

See `data_generator` in `utils.py`.

### Note

- Because a TCN's receptive field depends on depth of the network and the filter size, we need to make sure these the model we use can cover the sequence length T+20. 

- Using the `--seq_len` flag, one can change the # of values to recall (the typical setup is 10).


