# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:31:45 2024

@author: Ronan
"""

import keras_tuner as kt
import tensorflow as tf
import numpy as np


class HyperModel_bay(kt.HyperModel):
    
    def build(self, hp):  
        hp_units_1 = hp.Int("units_1", min_value=4, max_value=512, step=2,sampling='log')
        hp_lr = hp.Float("lr", min_value=1e-3, max_value=3e-2, sampling="linear")
       
        rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
        r2 = tf.keras.metrics.R2Score(class_aggregation="uniform_average", num_regressors=0, name="r2_score", dtype=None)
        
        opti = tf.keras.optimizers.Adam(learning_rate=hp_lr)
        f_loss = tf.keras.losses.MeanSquaredError()
        
        
        myModel = tf.keras.Sequential([
            tf.keras.layers.Dense(hp_units_1, activation="relu",name='fc1',kernel_initializer="glorot_uniform", input_shape=(11,)),
            tf.keras.layers.Dense(32, activation="relu",name='fc2',),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1)
            ])
        
        myModel.compile(optimizer=opti, loss=f_loss, metrics=[rmse,r2])
        return myModel
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=256,
            **kwargs,)
    
    
class HyperModel_grid(kt.HyperModel):
    
    def build(self, hp):  
        hp_units_1 = hp.Choice("units_1", [(2**k) for k in range(2,10)])
        hp_lr = hp.Choice("lr",list(np.linspace(1e-3,3e-2,15)))
       
        rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
        r2 = tf.keras.metrics.R2Score(class_aggregation="uniform_average", num_regressors=0, name="r2_score", dtype=None)
        
        opti = tf.keras.optimizers.Adam(learning_rate=hp_lr)
        f_loss = tf.keras.losses.MeanSquaredError()
        
        
        myModel = tf.keras.Sequential([
            tf.keras.layers.Dense(hp_units_1, activation="relu",name='fc1',kernel_initializer="glorot_uniform", input_shape=(11,)),
            tf.keras.layers.Dense(32, activation="relu",name='fc2',),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1)
            ])
        
        myModel.compile(optimizer=opti, loss=f_loss, metrics=[rmse,r2])
        return myModel
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=256,
            **kwargs,)