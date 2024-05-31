# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:19:10 2024

@author: Ronan
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import time as t

import shap

from catboost import CatBoostRegressor
from catboost import cv
import optuna
import catboost as cb

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer


my_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\"
steven_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\0_Steven Model\\"
alm_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\1_Alm and Hamre Model\\"

saving_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\"
models_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\1_Models\\"
figure_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\Figures\\1_Final Try\\"

#########################################################################################################
################################# Loading the input datasets ############################################

###################################### Steven Dataset ###################################################

X_train_unscaled_s = pd.read_excel(steven_path+"X_train_stev.xlsx")
y_train_unscaled_s = pd.read_excel(steven_path+"Y_train_stev.xlsx")

X_test_unscaled_s = pd.read_excel(steven_path+"X_test_stev.xlsx")
y_test_unscaled_s = pd.read_excel(steven_path+"Y_test_stev.xlsx")

###################################### Alm and Hamre Dataset ##############################################

X_train_unscaled_a = pd.read_excel(alm_path+"X_train_alm.xlsx")
y_train_unscaled_a = pd.read_excel(alm_path+"Y_train_alm.xlsx")

X_test_unscaled_a = pd.read_excel(alm_path+"X_test_alm.xlsx")
y_test_unscaled_a = pd.read_excel(alm_path+"Y_test_alm.xlsx")

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


def assess_model(myModel,X_test_unscaled,y_test_unscaled,model_name):
    y_pred = myModel.predict(X_test_unscaled)
    
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
    r2 = r2_score(y_test_unscaled,y_pred)
    mae = mean_absolute_error(y_test_unscaled,y_pred)
    
    print("R2:", round(r2,2)*100,"%")
    print("Root mean Squared Error:", round(rmse,4))
    print("Mean Absolute Error:", round(mae,2))
    
    plt.figure(figsize=(16,9))
    plt.scatter(y_test_unscaled["Blowcount [Blows/m]"],y_pred)
    perfect_pred = np.linspace(min(y_test_unscaled["Blowcount [Blows/m]"]),max(y_test_unscaled["Blowcount [Blows/m]"]),1000)
    plt.plot(perfect_pred,perfect_pred,'k')
    plt.title("Evaluation of the Cat Boosting model",fontsize=30)
    plt.ylabel("Predicted values",fontsize=15)
    plt.xlabel("Actual Values",fontsize=15)
    plt.grid()
    plt.savefig(figure_path+"results_CB_"+model_name+".png")
    plt.show()
    
    ratio = np.array(y_pred/y_test_unscaled["Blowcount [Blows/m]"])
    sorted_ratio = np.sort(ratio)
    n = len(ratio)
    cfd = np.zeros(n)
    for i in range(n):
        cfd[i]=i/n
        
    p50 = dicho_P(cfd,0.5)
    p10 = dicho_P(cfd,0.1)
    p90 = dicho_P(cfd,0.9)
    
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
    plt.savefig(figure_path+"proba_CB_"+model_name+".png")
    plt.show()


def plot_tuning(grid_results,model_name):
    
    cv_results = grid_results['cv_results'] #saves the cross-validation results of the tuning from grid_searc
    test_rmse_mean = cv_results['test-RMSE-mean']  #saves the mean RMSE on test dataset
    test_rmse_std = cv_results['test-RMSE-std'] #saves the std of RMSE on test dataset
    train_rmse_mean = cv_results['train-RMSE-mean'] #saves the mean RMSE on train dataset
    train_rmse_std = cv_results['train-RMSE-std'] #saves the std of RMSE on train dataset
    
    fig,((ax0, ax1)) = plt.subplots(1,2,figsize=(25,9)) #displays the results
    
    ax0.plot(test_rmse_mean,label="Test")
    ax0.plot(train_rmse_mean,label="Train")
    ax0.set_xlabel("Iterations")
    ax0.set_ylabel("Mean RMSE")
    ax0.set_title('Evolution of the Mean RMSE')
    ax0.grid()
    ax0.legend()

    ax1.plot(test_rmse_std,label="Test")
    ax1.plot(train_rmse_std,label="Train")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Std of RMSE")
    ax1.set_title('Evolution of the Std of RMSE')
    ax1.grid()
    ax1.legend()
    
    plt.suptitle("Evolution of the RMSE along the cv process",fontsize=30)
    plt.tight_layout()
    plt.savefig(figure_path+"results_CB_RMSE_"+model_name,bbox_inches="tight")


def plot_residuals(model,X_test_unscaled,y_test_unscaled,model_name):
    model.fit(X_train, y_train, verbose=False,eval_set=(X_val, y_val),early_stopping_rounds=100)
    
    y_pred = model.predict(X_test_unscaled)
    residuals = y_test_unscaled["Blowcount [Blows/m]"] - y_pred
    
    y_pred_standardized = (y_pred - np.mean(y_pred)) / np.std(y_pred)
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred_standardized, y=standardized_residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.grid()
    plt.savefig(figure_path+"residuals_dist_cb_"+model_name,bbox_inches="tight")
    plt.show()

    # Residuals distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(standardized_residuals, kde=True, bins=30)
    plt.xlabel('Standardized Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid()
    plt.savefig(figure_path+"residuals_hist_cb_"+model_name,bbox_inches="tight")
    plt.show()


    
#########################################################################################################
############################### Explanation of a given model #############################################
def explanation(X_train,model,model_name):
    shap.initjs() #initiate the core of shap model
    explainer = shap.TreeExplainer(model) #defines the explainer for the model
    shap_values = explainer(X_train) #trains the explainer on the training dataset
    fig = shap.summary_plot(shap_values) #plots the impact of each variable in a graph
    plt.savefig(figure_path+"feature_importance_cb_shap_"+model_name,bbox_inches="tight")
    fig_2 = shap.dependence_plot("Qsi [kN]", shap_values.values, X_train) #plots the dependance of Qsi with the output
    plt.savefig(figure_path+"Qsi_cb_shap_"+model_name,bbox_inches="tight")



#########################################################################################################
################################# Basic Cat Boosting Model ##############################################

def basic_model_stev(): #assesses the results for a basic model (Stevens method)
    X_train,X_val,y_train,y_val = train_test_split(X_train_unscaled_s,y_train_unscaled_s, test_size=0.3, random_state=42)
    model = CatBoostRegressor(iterations=1700, learning_rate=1e-3, depth=12, loss_function='RMSE',verbose=0,task_type="GPU")
    model.fit(X_train, y_train, verbose=False,eval_set=(X_val, y_val),early_stopping_rounds=60)
    assess_model(model,X_test_unscaled_s,y_test_unscaled_s,"stev")
    
def basic_model_alm():  #assesses the results for a basic model (Alm and Hamre method)
    X_train,X_val,y_train,y_val = train_test_split(X_train_unscaled_a,y_train_unscaled_a, test_size=0.3, random_state=42)
    model = CatBoostRegressor(iterations=1700, learning_rate=0.1, depth=7, loss_function='RMSE',verbose=0,task_type="GPU")
    model.fit(X_train, y_train, verbose=False,eval_set=(X_val, y_val),early_stopping_rounds=60)
    assess_model(model,X_test_unscaled_a,y_test_unscaled_a,"alm")    
    
# basic_model_stev()
# basic_model_alm()

#########################################################################################################
########################### Definition of data for HP Tuning ###########################################

X_train, X_val, y_train, y_val = train_test_split(X_train_unscaled_s, y_train_unscaled_s, test_size=0.3, random_state=42)


#########################################################################################################
########################### HyperParameters Tuning GridSearch ###########################################

def grid_tuning(model_name):
    #defines the model with un tunned parameters
    model = CatBoostRegressor(iterations=1700, learning_rate=0.1, depth=7, loss_function='RMSE',verbose=0,task_type="GPU")

    grid = {
    'learning_rate': [0.03,0.001,0.01],
    'depth':[4, 6, 8],
    'l2_leaf_reg': [1, 5,10],
    'bagging_temperature': [0.05,0.25,0.75],
    'iterations':[1250,1500,1750],
    'random_strength':[0.1,0.5,0.8] } #defines the space of research 
    
    start_time = t.time() #saves the execution time
    grid_search_results = model.grid_search(grid, X_train,y_train, shuffle=False,partition_random_seed=3)
    end_time = t.time()
    
    execution_time = end_time-start_time
    
    time = pd.read_excel(saving_path+"4_Computational Cost\\Time.xlsx") #saves times for latter plot
    new_row = pd.DataFrame({
    "Model": ["CB "+model_name],
    "Execution Type": ["HP Tuning - Grid"],
    "N Trials": [729],
    "Time [s]": [execution_time],
    "Figure":["Tuning Time"]})
    time = pd.concat([new_row,time],ignore_index=True)
    time.to_excel(saving_path+"4_Computational Cost\\Time.xlsx",index=False)
    
    #save the best hyperparameters as an excel file
    df = pd.DataFrame(index=[0],columns=list(grid_search_results['params'].keys()))
    for key in df.columns:
        df[key]=grid_search_results['params'][key]
    df.to_excel(models_path+"tuning_cat_grid_"+model_name+".xlsx")
    plot_tuning(grid_search_results,"grid_"+model_name)
    return grid_search_results

# grid_stev = grid_tuning("stev")
# grid_alm = grid_tuning("alm")

#########################################################################################################
########################### HyperParameters Tuning Bayesian Opti ########################################

def objective(trial): #objective function to optimize with bayesian process
    #defines the space of search for bayesian research
    params = {
        "iterations": 3,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-2),
        "depth": trial.suggest_int("depth", 5, 15),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0, 1),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1)}
    
    model = cb.CatBoostRegressor(**params, silent=True) #defines the model without verbose
    model.fit(X_train, y_train) 
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False) #return the value of objective function (RMSE of validation dataset)
    return rmse

def bayesian_tuning(objective,model_name):
    n_trial = 500 #defines the number of iteration
    study = optuna.create_study(direction='minimize') #creates the study 
    
    start_time = t.time() #saves the execution time
    study.optimize(objective, n_trials=n_trial)
    end_time = t.time()
    
    execution_time = end_time-start_time
    
    time = pd.read_excel(saving_path+"4_Computational Cost\\Time.xlsx") #save time for latter plot
    new_row = pd.DataFrame({
    "Model": ["CB "+model_name],
    "Execution Type": ["HP Tuning - Bay"],
    "N Trials": [n_trial],
    "Time [s]": [execution_time],
    "Figure":["Tuning Time"]})
    time = pd.concat([new_row,time],ignore_index=True)
    time.to_excel(saving_path+"4_Computational Cost\\Time.xlsx",index=False)
    
    best_hp = study.best_params #saves the best HP to an excel file
    df = pd.DataFrame(index=[0],columns=list(best_hp.keys()))
    for key in df.columns:
        df[key]=best_hp[key]
    df.to_excel(models_path+"tuning_cat_bay"+model_name+".xlsx")
    return study

# bay_stev = bayesian_tuning(objective, "stev")
# bay_alm = bayesian_tuning(objective, "alm")

#########################################################################################################


#########################################################################################################
################################# Complex Cat Boosting Model ##############################################

def complex_model_stev(): #assesses a model with tuned HP and displays results
    X_train,X_val,y_train,y_val = train_test_split(X_train_unscaled_s,y_train_unscaled_s, test_size=0.3, random_state=42)
    model = CatBoostRegressor(iterations=1700, learning_rate=0.1, depth=7, loss_function='RMSE',l2_leaf_reg=3,random_strength=0.1,bagging_temperature=0.1,verbose=0,task_type="GPU")
    model.fit(X_train, y_train, verbose=False,eval_set=(X_val, y_val),early_stopping_rounds=60)
    # assess_model(model,X_test_unscaled_s,y_test_unscaled_s,"stev")
    # plot_residuals(model,X_test_unscaled_s,y_test_unscaled_s,"stev")
    explanation(X_train_unscaled_s,model,"stev")

    
def complex_model_alm(): #assesses a model with tuned HP and displays results
    X_train,X_val,y_train,y_val = train_test_split(X_train_unscaled_a,y_train_unscaled_a, test_size=0.3, random_state=42)
    model = CatBoostRegressor(iterations=1700, learning_rate=0.1, depth=7, loss_function='RMSE',l2_leaf_reg=3,random_strength=0.1,bagging_temperature=0.1,verbose=0,task_type="GPU")
    model.fit(X_train, y_train, verbose=False,eval_set=(X_val, y_val),early_stopping_rounds=60)
    # assess_model(model,X_test_unscaled_a,y_test_unscaled_a,"alm")  
    # plot_residuals(model,X_test_unscaled_a,y_test_unscaled_a,"alm")
    explanation(X_train_unscaled_a,model,"alm")

    
complex_model_stev()
# complex_model_alm()

