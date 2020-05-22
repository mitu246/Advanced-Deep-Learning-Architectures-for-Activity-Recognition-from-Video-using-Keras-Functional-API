import keras
import tensorflow as tf
import math
import pandas as pd
import numpy as np
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
import re
import os
import random
import shutil
from keras.models import load_model
from tensorflow.keras.utils import plot_model

# Model Architecture:
from tensorflow.keras.layers import AveragePooling2D, concatenate, Activation, TimeDistributed, Conv2D, Dense, \
    MaxPooling2D, Flatten, LSTM, Dropout, BatchNormalization, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import Input


def inception_block(Input):
    base_1 = TimeDistributed(Conv2D(64, (1, 1), strides=2, padding='same', activation='relu'))(Input)
    base_2 = TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(Input)
    base_2 = TimeDistributed(Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'))(base_2)
    base_3 = TimeDistributed(AveragePooling2D(3, strides=2, padding='same'))(Input)
    base_3 = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(base_3)
    base_4 = TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(Input)
    base_4 = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(base_4)
    base_4 = TimeDistributed(Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'))(base_4)

    output = concatenate([base_1, base_2, base_3, base_4], axis=-1)

    return output


x = Input(shape=(10, 250, 250, 1))

## 1st Block:
cnlst_1 = TimeDistributed(Conv2D(128, (3, 3), strides=2, padding='same', activation='relu'))(x)
cnlst_1 = TimeDistributed(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))(cnlst_1)
cnlst_1 = TimeDistributed(MaxPooling2D((2, 2), strides=2, padding='same'))(cnlst_1)

# Batch Normalization:
cnlst_1 = TimeDistributed(BatchNormalization())(cnlst_1)

# Inception_Block_1:
incep_block_1 = inception_block(cnlst_1)

## 2nd Block:
cnlst_2 = TimeDistributed(Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'))(incep_block_1)
cnlst_2 = TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))(cnlst_2)
cnlst_2 = TimeDistributed(MaxPooling2D((2, 2), strides=1))(cnlst_2)

# Batch Normalization:
cnlst_2 = TimeDistributed(BatchNormalization())(cnlst_2)

## Flatten:
flat = TimeDistributed(Flatten())(cnlst_2)

# LSTM:
lstm = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.3)(flat)

## 2nd Model:
conv_block_1 = TimeDistributed(Conv2D(128, (5, 5), strides=(2, 2), activation='relu', padding="same"))(x)
conv_block_1 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(BatchNormalization())(conv_block_1)

conv_block_1 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(conv_block_1)
conv_block_1 = TimeDistributed(BatchNormalization())(conv_block_1)

conv_block_1 = TimeDistributed(Flatten())(conv_block_1)
conv_block_1 = Dropout(0.2)(conv_block_1)

conv_block_1 = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.3)(conv_block_1)  # used 64 units

# Concatenate:
concat_layer = concatenate([lstm, conv_block_1], axis=-1)

# Dense Block:
dense = Dense(256, activation='relu')(concat_layer)
dense = Dense(128, activation='relu')(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dense(32, activation='relu')(dense)
dense = Dropout(0.2)(dense)
output_layer = Dense(1, activation='sigmoid')(dense)

model_test_new = Model(x, output_layer)
print(model_test_new.summary())

