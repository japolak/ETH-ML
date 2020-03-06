# task4

Sleep Staging

## Problem description

In this task we will perform sequence classification. We will categorize temporally coherent and uniformly distributed short sections of a long time-series. In particular, for each 4 seconds of a lengthy EEG/EMG measurement of brain activity recorded during sleep, we will assign one of the 3 classes corresponding to the sleep stage present within the evaluated epoch.

## Data

3 channel of the data were concatenate, forming a `X_train` of shape `(64800, 1536)`. The signals were measured in 3 channels at the same time with the same measure frequency.

### Baseline model:

```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
reshape_1 (Reshape)          (None, 512, 3)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 512, 32)           4608      
_________________________________________________________________
lstm_2 (LSTM)                (None, 32)                8320      
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 99        
=================================================================
Total params: 13,027
Trainable params: 13,027
Non-trainable params: 0
_________________________________________________________________
```

### Conv model

[https://iopscience.iop.org/article/10.1088/1741-2552/aadc1f/pdf](https://iopscience.iop.org/article/10.1088/1741-2552/aadc1f/pdf)

8 channels of EEG

```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
reshape_1 (Reshape)          (None, 8, 900, 1)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 900, 3)         33        
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 900, 3)         30        
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 3, 450, 3)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 450, 5)         80        
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 450, 5)         80        
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 1, 150, 5)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 150, 7)         182       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 1, 37, 7)          0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 1, 1, 10)          2600      
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 1, 1, 3)           33        
_________________________________________________________________
dense_1 (Dense)              (None, 1, 1, 3)           12        
=================================================================
Total params: 3,050
Trainable params: 3,050
Non-trainable params: 0
_________________________________________________________________
```
