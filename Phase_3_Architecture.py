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
# Checking if just the Inception module model works:
# 2nd Model:
# Checking if just the Inception module model works:
# 2nd Model:
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling2D, concatenate, Activation, TimeDistributed, \
    Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Dropout, BatchNormalization, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras import regularizers

l2_regu = regularizers.l2(0.001)


def incep_a(inputi):
    b1 = TimeDistributed(AveragePooling2D((3, 3), strides=1, padding='same'))(inputi)
    b1 = TimeDistributed(Conv2D(96, (1, 1), strides=1, activation='relu', padding='same'))(b1)

    b2 = TimeDistributed(Conv2D(96, (1, 1), activation='relu', padding='same'))(inputi)

    b3 = TimeDistributed(Conv2D(64, (1, 1), activation='relu', padding='same'))(inputi)
    b3 = TimeDistributed(Conv2D(96, (3, 3), activation='relu', padding='same'))(b3)

    b4 = TimeDistributed(Conv2D(64, (1, 1), activation='relu', padding='same'))(inputi)
    b4 = TimeDistributed(Conv2D(96, (3, 3), activation='relu', padding='same'))(b4)
    b4 = TimeDistributed(Conv2D(96, (3, 3), activation='relu', padding='same'))(b4)

    concatu = concatenate([b1, b2, b3, b4], axis=-1)

    return concatu


def reduction_a(inputi):
    b1 = TimeDistributed(MaxPooling2D((3, 3), strides=2, padding='same'))(inputi)

    b2 = TimeDistributed(Conv2D(384, (3, 3), strides=2, activation='relu', padding='same'))(inputi)

    b3 = TimeDistributed(Conv2D(192, (1, 1), strides=1, activation='relu', padding='same'))(inputi)
    b3 = TimeDistributed(Conv2D(224, (3, 3), strides=1, activation='relu', padding='same'))(b3)
    b3 = TimeDistributed(Conv2D(256, (3, 3), strides=2, activation='relu', padding='same'))(b3)

    concatu = concatenate([b1, b2, b3], axis=-1)
    return concatu


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
cnlst_1 = TimeDistributed(
    Conv2D(128, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=l2_regu))(x)
cnlst_1 = TimeDistributed(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))(cnlst_1)
cnlst_1 = TimeDistributed(MaxPooling2D((2, 2), strides=2, padding='same'))(cnlst_1)

# Batch Normalization:
cnlst_1 = TimeDistributed(BatchNormalization())(cnlst_1)

# Inception_Block_1:
incep_block_1 = inception_block(cnlst_1)

## 2nd Block:
cnlst_2 = TimeDistributed(Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=l2_regu))(
    incep_block_1)
cnlst_2 = TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))(cnlst_2)
cnlst_2 = TimeDistributed(MaxPooling2D((2, 2), strides=1))(cnlst_2)
cnlst_2 = TimeDistributed(Dropout(0.2))(cnlst_2)
cnlst_2 = TimeDistributed(BatchNormalization())(cnlst_2)

# Incep Block 2:
incep_block_2 = inception_block(cnlst_2)

cnlst_3 = TimeDistributed(Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'))(incep_block_2)
cnlst_3 = TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))(cnlst_3)
cnlst_3 = TimeDistributed(MaxPooling2D((2, 2), strides=1, padding='same'))(cnlst_3)
cnlst_3 = TimeDistributed(Dropout(0.2))(cnlst_3)
cnlst_3 = TimeDistributed(BatchNormalization())(cnlst_3)

incep_A = incep_a(cnlst_3)
incep_A = TimeDistributed(Dropout(0.2))(incep_A)
reduc_A = reduction_a(incep_A)
batch_n_a = TimeDistributed(BatchNormalization())(reduc_A)

cnlst_4 = TimeDistributed(Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'))(batch_n_a)
cnlst_4 = TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))(cnlst_4)
cnlst_4 = TimeDistributed(MaxPooling2D((2, 2), strides=1, padding='same'))(cnlst_4)
cnlst_4 = TimeDistributed(Dropout(0.3))(cnlst_4)
cnlst_4 = TimeDistributed(BatchNormalization())(cnlst_4)

incep_A_1 = incep_a(cnlst_4)
incep_A_1 = TimeDistributed(Dropout(0.2))(incep_A_1)
reduc_A_1 = reduction_a(incep_A_1)
batch_n_a_1 = TimeDistributed(BatchNormalization())(reduc_A_1)

incep_final_1 = inception_block(batch_n_a_1)
maxpool_final_1 = TimeDistributed(MaxPooling2D((2, 2), strides=2, padding='same'))(incep_final_1)
incep_final_2 = inception_block(maxpool_final_1)
batch_norm_final_1 = TimeDistributed(BatchNormalization())(incep_final_2)
maxpool_final_2 = TimeDistributed(MaxPooling2D((2, 2), strides=2, padding='same'))(batch_norm_final_1)

# Batch Normalization:
batch_norm_new = TimeDistributed(BatchNormalization())(maxpool_final_2)

## Flatten:
flat = TimeDistributed(Flatten())(batch_norm_new)

# LSTM:
# lstm=LSTM(32,return_sequences=False,dropout=0.2,recurrent_dropout=0.3)(flat)


## 2nd Model:
conv_block_1 = TimeDistributed(Conv2D(128, (7, 7), strides=(3, 3), activation='relu', padding="same"))(x)
conv_block_1 = TimeDistributed(Conv2D(256, (5, 5), strides=(2, 2), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), strides=2, padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), strides=2, padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(BatchNormalization())(conv_block_1)
conv_block_1 = TimeDistributed(Dropout(0.2))(conv_block_1)

conv_block_1 = TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(128, (5, 5), strides=(2, 2), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(
    Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="same", kernel_regularizer=l2_regu))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), strides=2, padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(BatchNormalization())(conv_block_1)
conv_block_1 = TimeDistributed(Dropout(0.2))(conv_block_1)

conv_block_1 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(conv_block_1)
conv_block_1 = TimeDistributed(BatchNormalization())(conv_block_1)
conv_block_1 = TimeDistributed(Dropout(0.2))(conv_block_1)

conv_block_1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same"))(conv_block_1)
conv_block_1 = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(conv_block_1)
conv_block_1 = TimeDistributed(BatchNormalization())(conv_block_1)
conv_block_1 = TimeDistributed(Dropout(0.2))(conv_block_1)

conv_block_1 = TimeDistributed(Flatten())(conv_block_1)
conv_block_1_outu = Dropout(0.2)(conv_block_1)

concatu = concatenate([conv_block_1_outu, flat])

lstm_layer = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.3)(concatu)  # used 64 units

batch_norm_new = BatchNormalization()(lstm_layer)
densu_1 = Dense(1024, activation='relu')(batch_norm_new)

# Model 3:
conv_3D = Conv3D(128, (3, 3, 3), strides=2, padding='same')(x)
conv_3D = Conv3D(64, (3, 3, 3), strides=1, padding='same')(conv_3D)
conv_3D = MaxPooling3D((2, 2, 2), strides=2, padding='same')(conv_3D)
conv_3D = Conv3D(64, (3, 3, 3), strides=2, padding='same')(conv_3D)
conv_3D = Conv3D(32, (3, 3, 3), strides=1, padding='same')(conv_3D)
conv_3D = MaxPooling3D((2, 2, 2), padding='same')(conv_3D)
conv_3D = BatchNormalization()(conv_3D)
conv_3D = Dropout(0.3)(conv_3D)
conv_3D = Conv3D(128, (3, 3, 3), strides=2, padding='same', kernel_regularizer=l2_regu)(conv_3D)
conv_3D = MaxPooling3D((2, 2, 2), padding='same')(conv_3D)
conv_3D = Conv3D(64, (3, 3, 3), strides=1, padding='same')(conv_3D)
conv_3D = MaxPooling3D((2, 2, 2), padding='same')(conv_3D)
conv_3D = Conv3D(32, (3, 3, 3), padding='same')(conv_3D)
conv_3D = MaxPooling3D((2, 2, 2), strides=2, padding='same')(conv_3D)
conv_3D = BatchNormalization()(conv_3D)
conv_3D = Flatten()(conv_3D)

densu_2 = Dense(1024, activation='relu')(conv_3D)

# Concatenate:
concat_layer = concatenate([densu_2, densu_1], axis=-1)

densu_3 = Dense(512, activation='relu')(concat_layer)
densu_3 = Dropout(0.2)(densu_3)
densu_3 = Dense(256, activation='relu')(densu_3)
densu_3 = Dense(128, activation='relu')(densu_3)
dense_final = Dropout(0.2)(densu_3)

output_layer = Dense(1, activation='sigmoid')(dense_final)

model_test_new = Model(x, output_layer)
print(model_test_new.summary())
