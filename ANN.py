# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:35:23 2024

@author: Ronan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from lime import lime_tabular

import time as t

import tensorflow as tf
import keras_tuner as kt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import HyperModels
from matplotlib import cm


my_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\"
steven_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\0_Steven Model\\"
alm_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\1_Alm and Hamre Model\\"

saving_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\"
models_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\1_Models\\"
figure_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\Figures\\1_Final Try\\"


#########################################################################################################
################################# Loading the input datasets ############################################

def convert_tf(file_path_excel):
    file_pd = pd.read_excel(file_path_excel)
    try:
        file_np = file_pd.values
    except : 
        raise "Error"
    return tf.convert_to_tensor(file_np, dtype=tf.float32)

###################################### Steven Dataset ###################################################

X_train_unscaled_s = convert_tf(steven_path+"X_train_stev.xlsx")
y_train_unscaled_s = convert_tf(steven_path+"Y_train_stev.xlsx")

X_test_unscaled_s = convert_tf(steven_path+"X_test_stev.xlsx")
y_test_unscaled_s = convert_tf(steven_path+"Y_test_stev.xlsx")

scaler_X_s = StandardScaler()
scaler_y_s = StandardScaler()

X_train_scaled_s = scaler_X_s.fit_transform(X_train_unscaled_s)
y_train_scaled_s = scaler_y_s.fit_transform(y_train_unscaled_s)

X_test_scaled_s = scaler_X_s.fit_transform(X_test_unscaled_s)
y_test_scaled_s = scaler_y_s.fit_transform(y_test_unscaled_s)

###################################### Alm and Hamre Dataset ###################################################

X_train_unscaled_a = convert_tf(alm_path+"X_train_alm.xlsx")
y_train_unscaled_a = convert_tf(alm_path+"Y_train_alm.xlsx")

X_test_unscaled_a = convert_tf(alm_path+"X_test_alm.xlsx")
y_test_unscaled_a = convert_tf(alm_path+"Y_test_alm.xlsx")

scaler_X_a = StandardScaler()
scaler_y_a = StandardScaler()

X_train_scaled_a = scaler_X_a.fit_transform(X_train_unscaled_a)
y_train_scaled_a = scaler_y_a.fit_transform(y_train_unscaled_a)

X_test_scaled_a = scaler_X_a.fit_transform(X_test_unscaled_a)
y_test_scaled_a = scaler_y_a.fit_transform(y_test_unscaled_a)


#########################################################################################################
################################# Creating the basic ANN model ##########################################

def ANN_3layer_model(lr):
    rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None) #root mean square error for the model
    r2 = tf.keras.metrics.R2Score(class_aggregation="uniform_average", num_regressors=0, name="r2_score", dtype=None) # correlation coeff 
    opti = tf.keras.optimizers.Adam(learning_rate=lr) #optimizer is ADAM
    f_loss = tf.keras.losses.MeanSquaredError() # loss function is MSE
    
    myModel = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='sigmoid',name='fc1', input_shape=(11,)), #first layer
        tf.keras.layers.Dense(32, activation='sigmoid',name='fc2',), #second layer
        tf.keras.layers.Dense(1)
        ])
    
    myModel.compile(optimizer=opti, loss=f_loss, metrics=[rmse,r2])    
    
    return myModel


#########################################################################################################
################################# Impact of the learning rate ###########################################
list_lr = [1e-3,5e-3,1e-2,5e-2,1e-1,5e-1]

def impact_lr(list_lr):
    num_epoch = 200
    X_train,X_val,y_train,y_val = train_test_split(X_train_scaled_s,y_train_scaled_s, test_size=0.3, random_state=42) #separate training and validation data
    batch_size = 16 #hyperparameter
    
    list_loss = [] #used to save the losses of training dataset
    list_val_loss = [] #used to save the losses of validation dataset
    for lr in list_lr:
        model = ANN_3layer_model(lr) #creating the Network
        hist = model.fit(X_train,y_train, epochs=num_epoch, validation_data=(X_val,y_val),batch_size=batch_size,verbose=0).history #fitting the model
        list_loss.append(min(hist["loss"])) #saving loss
        list_val_loss.append(min(hist["val_loss"])) #saving loss
    
    plt.Figure(figsize=(16,9))
    plt.plot(list_lr,list_loss,label="Training Dataset")
    plt.plot(list_lr,list_val_loss,label="Validation Dataset")
    plt.grid()
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Learning Rate",fontsize=15)
    plt.ylabel("Loss Function",fontsize=15)
    plt.title("Impact of the learning rate on the loss function",fontsize=18)
    plt.savefig(figure_path+"Learning Rate Impact.png",bbox_inches="tight")
    plt.show()

# impact_lr(list_lr)


#########################################################################################################
############################### Impact of the number of units ###########################################

list_units = [2**k for k in range(4,10)]

def impact_units(list_units):
    num_epoch = 200
    X_train,X_val,y_train,y_val = train_test_split(X_train_scaled_s,y_train_scaled_s, test_size=0.3, random_state=42) #separate training and validation data
    batch_size = 16

    list_loss = [] #used to save the losses of training dataset
    list_val_loss = []  #used to save the losses of validation dataset
    
    for units in list_units:
        
        rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
        r2 = tf.keras.metrics.R2Score(class_aggregation="uniform_average", num_regressors=0, name="r2_score", dtype=None)
        opti = tf.keras.optimizers.Adam(learning_rate=1e-2)
        f_loss = tf.keras.losses.MeanSquaredError()
        
        myModel = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='sigmoid',name='fc1', input_shape=(11,)),#first layer with varying number of neurons
            tf.keras.layers.Dense(16, activation='sigmoid',name='fc2',),
            tf.keras.layers.Dense(1)
            ])
        
        myModel.compile(optimizer=opti, loss=f_loss, metrics=[rmse,r2])  
        
        start_time = t.time() #to save the computational time for each number of units
        hist = myModel.fit(X_train,y_train, epochs=num_epoch, validation_data=(X_val,y_val),batch_size=batch_size,verbose=0).history
        end_time = t.time()
        
        execution_time = end_time - start_time
        
        time = pd.read_excel(saving_path+"4_Computational Cost\\Time.xlsx") #saving the data for latter plot
        new_row = pd.DataFrame({
        "Model": ["ANN stev"],
        "Execution Type": ["Single run"],
        "N Trials": [1],
        "Time [s]": [execution_time],
        "units":[units],
        "Figure":["Impact of units"]})
        
        time = pd.concat([new_row,time],ignore_index=True)
        time.to_excel(saving_path+"4_Computational Cost\\Time.xlsx",index=False)
        
        list_loss.append(min(hist["loss"])) #saving best iteration
        list_val_loss.append(min(hist["val_loss"])) #saving best iteration
        
    fig = plt.Figure(figsize=(16,9))
    plt.plot(list_units,list_loss,label="Training Dataset")
    plt.plot(list_units,list_val_loss,label="Validation Dataset")
    plt.xscale("log")
    plt.grid()
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Number of neurons",fontsize=15)
    plt.ylabel("Loss Function",fontsize=15)
    plt.title("Impact of the number of neurons on the loss function",fontsize=18)
    plt.savefig(figure_path+"Architecture Impact.png",bbox_inches="tight")
    plt.show()        

# impact_units(list_units)

#########################################################################################################
############################ Identification of overfitting issue ########################################

def overfitting():
    num_epoch = 500 #using an important number of epochs to foster overfitting
    X_train,X_val,y_train,y_val = train_test_split(X_train_scaled_s,y_train_scaled_s, test_size=0.3, random_state=42) #separate training and validation data
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
    plt.savefig(figure_path+"Overfitting.png",bbox_inches="tight")
    plt.show()

# overfitting()

#########################################################################################################
############################ Hyperparameters Tuning of the model ########################################


class MyHyperModel(kt.HyperModel):
    
    def build(self, hp):
        #introducing all the hyperparameters to tune in the model
        hp_units_1 = hp.Int("units_1", min_value=16, max_value=512, step=2,sampling="log")
        hp_units_2 = hp.Int("units_2", min_value=16, max_value=512, step=2,sampling="log")
        
        hp_lr = hp.Float("lr", min_value=1e-3, max_value=5e-2)
        hp_acti_1 = hp.Choice("activation_1", ["relu", "sigmoid","tanh"])
        hp_acti_2 = hp.Choice("activation_2", ["relu", "sigmoid","tanh"])
        
        hp_drop= hp.Float("dropout",min_value=0, max_value=1)    
        hp_init = hp.Choice("weight_init",["he_normal","glorot_uniform"])
        hp_L2_1 = hp.Float("L2_1",min_value=0, max_value=1)
        hp_L2_2 = hp.Float("L2_2",min_value=0, max_value=1)

        #classic definition of the Newtork, Optimizer, Performance indicators
        rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
        r2 = tf.keras.metrics.R2Score(class_aggregation="uniform_average", num_regressors=0, name="r2_score", dtype=None)
        mae = tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error', dtype=None)
        
        opti = tf.keras.optimizers.Adam(learning_rate=hp_lr)
        f_loss = tf.keras.losses.MeanSquaredError()
        
        myModel = tf.keras.Sequential([
            tf.keras.layers.Dense(hp_units_1, activation=hp_acti_1,name='fc1',kernel_initializer=hp_init,kernel_regularizer=tf.keras.regularizers.l2(hp_L2_1), input_shape=(11,)),
            tf.keras.layers.Dense(hp_units_2, activation=hp_acti_2,name='fc2',kernel_regularizer=tf.keras.regularizers.l2(hp_L2_2),),
            tf.keras.layers.Dropout(hp_drop),
            tf.keras.layers.Dense(1)
            ])
        
        myModel.compile(optimizer=opti, loss=f_loss, metrics=[rmse,r2,mae])
        return myModel

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int("batch_size",min_value=16, max_value=512, step=2,sampling="log"),
            **kwargs,)

def ANN_tuning(X_train,y_train,model_name):
    X_train, X_val, y_train, y_val = train_test_split(np.array(X_train), np.array(y_train),test_size=0.25,random_state=100)
    
    #number of trials to do before providing results
    n_trial = 500
    
    #defining the Bayesian Tuner to minimise the loss of the validation set
    tuner = kt.BayesianOptimization(
        hypermodel = MyHyperModel(),
        objective="val_loss",
        max_trials=n_trial,
        alpha=0.0001,
        beta=2.6,
        overwrite=True,
        directory=models_path+"Tuning\\",
        project_name="ANN",) 
    
    #callback to prevent from overfitting
    early_stop = tf.keras.callbacks.EarlyStopping('val_loss', patience=60,restore_best_weights=True)
    
    start_time = t.time() #to save the computational time of the tuning
    tuner.search(X_train, y_train, epochs=175, callbacks=[early_stop],validation_data=(X_val, y_val),verbose=0)
    end_time = t.time()
    
    execution_time = end_time - start_time
    
    time = pd.read_excel(saving_path+"4_Computational Cost\\Time.xlsx") #saving the data for latter plot
    new_row = pd.DataFrame({
    "Model": ["ANN "+model_name],
    "Execution Type": ["HP Tuning - Bay"],
    "N Trials": [n_trial],
    "Time [s]": [execution_time],
    "Figure":["Tuning Time"]})
    time = pd.concat([new_row,time],ignore_index=True)
    time.to_excel(saving_path+"4_Computational Cost\\Time.xlsx",index=False)
    
    
    best_param = tuner.get_best_hyperparameters(num_trials=10) #saving the 10 best combinations of HP
    columns = list(best_param[0].values.keys()) #saving it in an excel file
    df = pd.DataFrame(index=[k for k in range(10)],columns = columns)
    for k in range(10):
        for key in columns:
            df.loc[k,key]=best_param[k][key]
    df.to_excel(models_path+"best_hp_ANN_"+model_name+".xlsx")
    return tuner

# stev_tuner = ANN_tuning(X_train_scaled_s,y_train_scaled_s,"stev")
# alm_tuner = ANN_tuning(X_train_scaled_a,y_train_scaled_a,"alm")

#########################################################################################################
############################### Assessment of a given model #############################################

def dicho_P(liste,p):
    left = 0
    right = len(liste) - 1
    
    # Initialiser les indices des valeurs les plus proches de 0.5
    closest_index_low = None
    closest_index_high = None
    
    # Recherche binaire pour trouver les indices des valeurs les plus proches de 0.5
    while left <= right:
        mid = (left + right) // 2
        mid_value = liste[mid]
        
        # Mettre à jour les indices des valeurs les plus proches si nécessaire
        if mid_value < p:
            closest_index_low = mid
            left = mid + 1
        elif mid_value > p:
            closest_index_high = mid
            right = mid - 1
        else:
            closest_index_low = mid
            closest_index_high = mid
            break  
    
    # Gérer le cas où aucune valeur exacte de 0.5 n'a été trouvée
    if closest_index_low is None:
        closest_index_low = right
    if closest_index_high is None:
        closest_index_high = left
    
    return closest_index_low, closest_index_high


def assess_model(myModel,y_scaler,X_test_scaled,y_test_unscaled,model_name):
    y_pred_norm = myModel.predict(X_test_scaled) #uses the model to predict normalized blowcount/m
    y_pred = y_scaler.inverse_transform(y_pred_norm) #uses the scaler to predict absolute values for blowcount/m
    
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred)) #assesses the RMSE
    r2 = r2_score(y_test_unscaled,y_pred) #assesses the correlation coeff
    mae = mean_absolute_error(y_test_unscaled,y_pred) #assesses the MAE
    
    print("R2:", round(r2,2)*100,"%")
    print("Root mean Squared Error:", round(rmse,4))
    print("Mean Absolute Error:", round(mae,2))
    
    plt.figure(figsize=(16,9)) #plotting the actual vs. predicted values
    plt.scatter(y_test_unscaled,y_pred)
    perfect_pred = np.linspace(min(y_test_unscaled),max(y_test_unscaled),1000)
    plt.plot(perfect_pred,perfect_pred,'k')
    plt.title("Evaluation of the ANN model",fontsize=30)
    plt.ylabel("Predicted values",fontsize=15)
    plt.xlabel("Actual Values",fontsize=15)
    plt.grid()
    plt.savefig(figure_path+"results_ANN_"+model_name+".png")
    plt.show()
    
    ratio = np.array(y_pred/y_test_unscaled)[:,0] #divides the predicted by actual values
    sorted_ratio = np.sort(ratio) #sort the list to make a probabilist analysis
    n = len(ratio)
    cfd = np.zeros(n) #empty list to save cumulative proba 
    for i in range(n):
        cfd[i]=i/n
        
    p50 = dicho_P(cfd,0.5) #finds the index for the 50th quantile
    p10 = dicho_P(cfd,0.1) #finds the index for the 10th quantile
    p90 = dicho_P(cfd,0.9) #finds the index for the 90th quantile
    
    #displays the results
    print("P50 = ",(sorted_ratio[p50[0]]+sorted_ratio[p50[1]])/2)
    print("Confidence Interval at 80%: [",(sorted_ratio[p10[0]]+sorted_ratio[p10[1]])/2,";",(sorted_ratio[p90[0]]+sorted_ratio[p90[1]])/2,"]")
        
    plt.figure(figsize=(16,9))
    sns.histplot(ratio, bins=15, color='blue',stat='probability',binrange=(0, 2),label="Distribution of the ratio")
    plt.plot(sorted_ratio,cfd,'k',label = "CDF")
    plt.xlim(0,2)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("$\\frac{y_{predicted}}{y_{measured}}$",fontsize=15)
    plt.ylabel("Probability of occurance",fontsize=15)
    plt.title("Empirical Probabilities for the ratio: $\\frac{y_{predicted}}{y_{measured}}$",fontsize=30)
    plt.grid()
    plt.legend()
    plt.savefig(figure_path+"proba_ANN_"+model_name+".png")
    plt.show()


def plot_residuals(model,y_scaler,X_test_scaled,y_test_unscaled,model_name):   
    y_pred_norm = model.predict(X_test_scaled)  #uses the model to predict normalized blowcount/m
    y_pred = y_scaler.inverse_transform(y_pred_norm.reshape(-1,1)) #uses the scaler to predict absolute values for blowcount/m
    residuals = y_test_unscaled - y_pred #calculates the residuals
    
    y_pred_standardized = (y_pred - np.mean(y_pred)) / np.std(y_pred) #normalises the predicted values
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals) #normalises the residuals
    
    #displays the Residuals vs. Predicted Values
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred_standardized, y=standardized_residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
    plt.xlabel('Standardized Predicted Values')
    plt.ylabel('Standardized Residuals')
    plt.title('Residuals vs. Predicted Values')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.grid()
    plt.savefig(figure_path+"residuals_dist_ANN_"+model_name,bbox_inches="tight")
    plt.show()

    #displays the residuals distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(standardized_residuals, kde=True, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid()
    plt.savefig(figure_path+"residuals_hist_ANN_"+model_name,bbox_inches="tight")
    plt.show()
    
    
#########################################################################################################
############################### Explanation of a given model #############################################
def explanation(X_train,ann_kt_name,model_name,index):
    myModel = tf.keras.models.load_model(ann_kt_name) #loads the model
    explainer = lime_tabular.LimeTabularExplainer(X_train, mode="regression", feature_names= pd.read_excel(steven_path+"X_train_stev.xlsx")) #uses lime to explain the impact of each feature 
    explanation = explainer.explain_instance(X_train[index], myModel.predict, num_features=11)
    fig = explanation.as_pyplot_figure() #displays the results

    fig.savefig(figure_path+"feature_importance_lime_ann"+model_name+".png",bbox_inches="tight")


#########################################################################################################
################################# Creating the complex ANN model ########################################

def ANN_3layer_complex_model(lr):
    rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
    r2 = tf.keras.metrics.R2Score(class_aggregation="uniform_average", num_regressors=0, name="r2_score", dtype=None)
    opti = tf.keras.optimizers.Adam(learning_rate=lr)
    f_loss = tf.keras.losses.MeanSquaredError()
    
    myModel = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu',name='fc1',kernel_initializer="glorot_uniform",kernel_regularizer=tf.keras.regularizers.l2(0.56622), input_shape=(11,)),
        tf.keras.layers.Dense(16, activation='relu',name='fc2',kernel_regularizer=tf.keras.regularizers.l2(0.4226)),
        tf.keras.layers.Dropout(0.06609),
        tf.keras.layers.Dense(1)
        ])
    
    myModel.compile(optimizer=opti, loss=f_loss, metrics=[rmse,r2])    
    
    return myModel

#########################################################################################################
#################################### Assessement of Steven Method #######################################

# num_epoch = 200
# myModel = ANN_3layer_complex_model(0.0034645)
# early_stop = tf.keras.callbacks.EarlyStopping('val_loss', patience=60,restore_best_weights=True)
X_train,X_val,y_train,y_val = train_test_split(X_train_scaled_s,y_train_scaled_s, test_size=0.3, random_state=42)
# myModel.fit(X_train,y_train, epochs=num_epoch, validation_data=(X_val,y_val),callbacks=[early_stop],batch_size=16,verbose=0)
# myModel.save("ann_complex_stev.h5")
# myModel = tf.keras.models.load_model('ann_complex_stev.h5')
# assess_model(myModel,scaler_y_s,X_test_scaled_s,y_test_unscaled_s,"stev")
# plot_residuals(myModel,scaler_y_s,X_test_scaled_s,y_test_unscaled_s,"stev")
explanation(X_train,"ann_complex_stev.h5","stev",75)



#########################################################################################################
################################ Assessement of Alm and Hamre Method ####################################

# num_epoch = 350
# myModel = ANN_3layer_model(0.029326)
# early_stop = tf.keras.callbacks.EarlyStopping('val_loss', patience=60,restore_best_weights=True)
# X_train,X_val,y_train,y_val = train_test_split(X_train_scaled_a,y_train_scaled_a, test_size=0.3, random_state=42)
# myModel.fit(X_train,y_train, epochs=num_epoch, validation_data=(X_val,y_val),callbacks=[early_stop],batch_size=128,verbose=0)
# myModel.save("ann_complex_alm.h5")
# myModel = tf.keras.models.load_model('ann_complex_alm.h5')
# assess_model(myModel,scaler_y_a,X_test_scaled_a,y_test_unscaled_a,"alm")
# plot_residuals(myModel,scaler_y_a,X_test_scaled_a,y_test_unscaled_a,"alm")

#########################################################################################################
############################### Presentation of the Bayesian Approach ###################################

def surf_opti():
    X_train, X_val, y_train, y_val = train_test_split(np.array(X_train_scaled_s), np.array(y_train_scaled_s),test_size=0.25,random_state=100)  #separates training and validation data
    early_stop = tf.keras.callbacks.EarlyStopping('val_loss', patience=60,restore_best_weights=True) #early stop to prevent from overfitting
    
    #defines the grid search tunner with 120 points (15x8) number of nodes in the mesh 
    tuner_grid = kt.GridSearch(
        HyperModels.HyperModel_grid(),
        objective="val_loss",
        max_trials=120,
        project_name="grid_search",
        seed=42,)
    
    start_time = t.time() #to save the execution time
    tuner_grid.search(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stop],verbose=0) #tuning HP
    end_time = t.time()
    
    execution_time = end_time-start_time
    
    time = pd.read_excel(saving_path+"4_Computational Cost\\Time.xlsx") #saving time for latter plot
    new_row = pd.DataFrame({
    "Model": ["ANN "+"stev"],
    "Execution Type": ["HP Tuning - GS"],
    "N Trials": [120],
    "Time [s]": [execution_time],
    "Figure":["Advantage of Bayesian Opti"]})
    
    time = pd.concat([new_row,time],ignore_index=True)
    time.to_excel(saving_path+"4_Computational Cost\\Time.xlsx",index=False)
    
    #saving the results for each node of the space
    trials = tuner_grid.oracle.trials
    score_grid = []
    unit_grid = []
    lr_grid = []
    for trial_id,trial in trials.items():
        hp = trial.hyperparameters.values
        unit_grid.append(hp["units_1"])
        lr_grid.append(hp["lr"])
        score_grid.append(trial.score)
    
    #defines the bayesian tunner with 5 points
    tuner_bay = kt.BayesianOptimization(
        HyperModels.HyperModel_bay(),
        objective="val_loss",
        max_trials=5,
        alpha=0.0001,
        overwrite=True,
        beta=2.6,
        project_name="bayesian_search",
        seed=42,)
    
    start_time = t.time() #to save the execution time
    tuner_bay.search(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stop],verbose=0) #tuning HP
    end_time = t.time() #to save the execution time
    
    execution_time = end_time-start_time #to save the execution time
    
    time = pd.read_excel(saving_path+"4_Computational Cost\\Time.xlsx") #saving for latter plot 
    new_row = pd.DataFrame({
    "Model": ["ANN "+"stev"],
    "Execution Type": ["HP Tuning - Bay"],
    "N Trials": [10],
    "Time [s]": [execution_time],
    "Figure":["Advantage of Bayesian Opti"]})
    
    time = pd.concat([new_row,time],ignore_index=True)
    time.to_excel(saving_path+"4_Computational Cost\\Time.xlsx",index=False)
    
    #saving the results for each node of the space
    trials = tuner_bay.oracle.trials
    score_bay = []
    unit_bay = []
    lr_bay = []
    for trial_id,trial in trials.items():
        hp = trial.hyperparameters.values
        unit_bay.append(hp["units_1"])
        lr_bay.append(hp["lr"])
        score_bay.append(trial.score)
    
    #displays the results
    X = np.linspace(1e-3,3e-2,15)
    Y = np.array([(2**k) for k in range(2,10)])
    
    Z = np.zeros([8,15])
    
    #defines the Z values for the 3D plot
    for i in range(15):
        for j in range(8):
            for k in range(120):
                if X[i]==lr_grid[k] and Y[j]==unit_grid[k] :
                    Z[j,i] = score_grid[k]
    
    X,Y = np.meshgrid(X,Y) #defines coordinates
    fig=plt.figure(figsize=(16,9))
    ax=fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm,alpha=0.5) #plot surface (grid search)
    fig.colorbar(surf, shrink=0.5, aspect=5)
     
    ax.plot(lr_bay,unit_bay,score_bay,'r') #plot line (bayesian opti)
    ax.set_xlabel("Learning Rate",fontsize=15)
    ax.set_ylabel("unit",fontsize=15)
    ax.set_zlabel("Score",fontsize=15)
    plt.title("Hyperparameter Tuning",fontsize=30)
    plt.savefig( "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\surface.png")
    plt.show()
    
    print("Bay Opti Score:",min(score_bay))
    print("Grid Search Opti Score:",min(score_grid))
    

# surf_opti()


    