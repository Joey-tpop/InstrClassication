import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
import glob 
import os
import json
from tensorflow.keras.callbacks import ModelCheckpoint
import dataset_generator
from dataset_generator import train_generator

class basic_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.FFL = [
            keras.layers.Dense(256),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(512),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(256),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(64),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(20, activation = "sigmoid")
        ]
    def call(self, inputs):
        x = inputs
        for layer in self.FFL:
            x = layer(x)
        return x

def custom_BCE(y_true, y_pred):
    mask = tf.math.not_equal(y_true, -1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    loss = -tf.reduce_mean(y_true_masked * tf.math.log(y_pred_masked) + (1 - y_true_masked) * tf.math.log(1 - y_pred_masked))
    return loss

class customAcc(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.correct = self.add_weight(name='correct', initializer='zeros', dtype = tf.int32)
        self.total = self.add_weight(name='total', initializer='zeros', dtype = tf.int32)
    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.math.not_equal(y_true, -1)
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        y_true_masked = tf.cast(y_true_masked > 0.5, tf.int32)
        y_pred_masked = tf.cast(y_pred_masked > 0.5, tf.int32)
        correct_values = tf.cast(tf.equal(y_true_masked, y_pred_masked), tf.int32)
        self.correct.assign_add(tf.reduce_sum(correct_values))
        self.total.assign_add(tf.size(y_true_masked))

    def result(self):
        return self.correct / self.total

    def reset_states(self):
        self.correct.assign(0)
        self.total.assign(0)

dataset = tf.data.Dataset.from_generator(train_generator, 
    output_signature = (tf.TensorSpec(shape = (128,), dtype = tf.int32),
                        tf.TensorSpec(shape = (20,), dtype = tf.float32)))
dataset = dataset.batch(32)


model = basic_model()
model.compile(tf.keras.optimizers.RMSprop(learning_rate=0.01),
              loss= custom_BCE,
              metrics=customAcc())
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
model.fit(dataset, epochs = 3, callbacks = [checkpoint])