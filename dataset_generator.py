import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
import glob 
import os
import json

DATA_ROOT = "openmic-2018"
if not os.path.exists(DATA_ROOT):
    raise ValueError("Did you forget to set the root?")

OPENMIC = np.load(os.path.join(DATA_ROOT, "openmic-2018.npz"), allow_pickle = True)
X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']

split_train = pd.read_csv(os.path.join(DATA_ROOT, 'partitions/split01_train.csv'), header=None)
split_test = pd.read_csv(os.path.join(DATA_ROOT, 'partitions/split01_test.csv'), header=None)
train_set = set(split_train[0].values) #len: 14915
test_set = set(split_test[0].values) #len: 5085

idx_train, idx_test = [], []

for idx, n in enumerate(sample_key):
    if n in train_set:
        idx_train.append(idx)
    else:
        idx_test.append(idx)

X_train = X[idx_train]
Y_true_train = Y_true[idx_train]
Y_mask_train = Y_mask[idx_train]
X_test = X[idx_test]
Y_true_test = Y_true[idx_test]
Y_mask_test = Y_mask[idx_test]
for sample_num, sample in enumerate(Y_true_train):
    for label_num, label in enumerate(sample):
        if Y_mask_train[sample_num][label_num] == False:
            Y_true_train[sample_num][label_num] = -1.

def train_generator():
    for i in range(len(idx_train)):
        for j in range(10):
            yield (X_train[i][j], Y_true_train[i])

