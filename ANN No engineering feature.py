# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:32:43 2024

@author: Ronan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#import the different ANN libraries
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import keras_tuner as kt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


my_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\2_Initial Data\\"
name_training_set = "training_data.csv"
saving_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\"
models_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\1_Models\\"
path_tensorboard="G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\2_TensorBoard\\"


def convert_tf(file_path_excel):
    file_pd = pd.read_excel(file_path_excel)
    try:
        file_np = file_pd.values
    except : 
        raise "Error"
    return tf.convert_to_tensor(file_np, dtype=tf.float32)

X_train_unscaled = convert_tf(my_path+"X_train_no_feature.xlsx")
y_train_unscaled = convert_tf(my_path+"Y_train_no_feature.xlsx")

X_test_unscaled = convert_tf(my_path+"X_test_no_feature.xlsx")
y_test_unscaled = convert_tf(my_path+"Y_test_no_feature.xlsx")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train_unscaled)
y_train_scaled = scaler_y.fit_transform(y_train_unscaled)

X_test_scaled = scaler_X.fit_transform(X_test_unscaled)
y_test_scaled = scaler_y.fit_transform(y_test_unscaled)


def ANN_3layer_model(lr):
    rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
    r2 = tf.keras.metrics.R2Score(class_aggregation="uniform_average", num_regressors=0, name="r2_score", dtype=None)
    opti = tf.keras.optimizers.Adam(learning_rate=lr)
    f_loss = tf.keras.losses.MeanSquaredError()
    
    myModel = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='sigmoid',name='fc1', input_shape=(8,)),
        tf.keras.layers.Dense(32, activation='sigmoid',name='fc2',),
        tf.keras.layers.Dense(1)
        ])
    
    myModel.compile(optimizer=opti, loss=f_loss, metrics=[rmse,r2])    
    
    return myModel

def k_fold_assess():
    num_epoch = 300
    split_num = 4 #usualy between 5 and 10
    k_fold = np.arange(1,split_num+1,1)
    i=0
    n_random = 42
    list_loss_val = np.zeros([split_num,num_epoch])
    list_loss_train = np.zeros([split_num,num_epoch])

    list_RMSE_val = np.zeros([split_num,num_epoch])
    list_MAE_val = np.zeros([split_num,num_epoch])
    list_r2_val = np.zeros([split_num,num_epoch])
    
    kf = KFold(n_splits=split_num,random_state=n_random,shuffle=True)
    
    for train_index,test_index in kf.split(X_train_scaled):
            X_train,X_val,y_train,y_val = X_train_scaled[train_index],X_train_scaled[test_index],y_train_scaled[train_index],y_train_scaled[test_index]
            model=ANN_3layer_model(1e-2)
            hist = model.fit(X_train,y_train, epochs=num_epoch, validation_data=(X_val,y_val),verbose=1).history
            list_loss_train[i,:] = hist['loss']
            list_loss_val[i,:] = hist['val_loss']
            list_r2_val[i,:] = hist['val_r2_score']
            list_RMSE_val[i,:] = hist['val_root_mean_squared_error']
    
    plt.figure(figsize=(16,9))
    
    plt.plot([k for k in range(num_epoch)],np.mean(list_loss_train,axis=0),label="Training Dataset")
    plt.plot([k for k in range(num_epoch)],np.mean(list_loss_val,axis=0),label="Validation Dataset")
    plt.tick_params(labelsize = 15)
    plt.title("Evolution of the loss function over cv process",fontsize=30)
    plt.xlabel("Epochs",fontsize=15)
    plt.ylabel("Mean value for Loss Function",fontsize=15)
    plt.legend(fontsize=12)
    plt.grid()
    plt.tight_layout()
    # plt.savefig(saving_path+"No extra feature model cv.png",bbox_inches="tight")
    plt.show()
    
    print("R2 score validation: ",np.mean(list_r2_val,axis=0).max())
    print("RMSE validation: ",np.mean(list_RMSE_val,axis=0).min())

def assess_model():
    num_epoch = 300
    X_train,X_val,y_train,y_val = train_test_split(X_train_scaled,y_train_scaled, test_size=0.3, random_state=42)
    model=ANN_3layer_model(1e-2)
    hist = model.fit(X_train,y_train, epochs=num_epoch, validation_data=(X_val,y_val),verbose=1).history
    
        
    plt.figure(figsize=(16,9))
    plt.plot([k for k in range(num_epoch)],hist["loss"],label="Training Dataset")
    plt.plot([k for k in range(num_epoch)],hist["val_loss"],label="Validation Dataset")
    plt.tick_params(labelsize = 15)
    plt.title("Evolution of the loss function over epochs",fontsize=30)
    plt.xlabel("Epochs",fontsize=15)
    plt.ylabel("Loss Function",fontsize=15)
    plt.legend(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.tick_params(labelsize = 15)
    plt.savefig(saving_path+"No extra feature model",bbox_inches="tight")
    plt.show()
    
    plt.figure(figsize=(16,9))
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    plt.scatter(y_test_unscaled,y_pred)
    perfect_pred = np.linspace(min(y_test_unscaled),max(y_test_unscaled),1000)
    plt.plot(perfect_pred,perfect_pred,'k')
    plt.title("Evaluation of the ANN model",fontsize=30)
    plt.ylabel("Predicted values",fontsize=15)
    plt.xlabel("Actual Values",fontsize=15)
    plt.tick_params(labelsize = 15)
    plt.grid()
    plt.show()
    
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
    r2 = r2_score(y_test_unscaled,y_pred)
    mae = mean_absolute_error(y_test_unscaled,y_pred)
    
    print("R2:", round(r2,2)*100,"%")
    print("Root mean Squared Error:", round(rmse,4))
    print("Mean Absolute Error:", round(mae,2))
    
assess_model()
    
    








