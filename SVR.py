# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:57:01 2024

@author: Ronan
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import time as t

import shap

import seaborn as sns

from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

my_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\"
steven_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\0_Steven Model\\"
alm_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\1_Alm and Hamre Model\\"

saving_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\"
models_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\1_Models\\"
figure_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\Figures\\1_Final Try\\"

#########################################################################################################
################################# Loading the input datasets ############################################

#########################################################################################################
################################# Loading the input datasets ############################################

###################################### Steven Dataset ###################################################

X_train_unscaled_s =pd.read_excel(steven_path+"X_train_stev.xlsx")
y_train_unscaled_s = pd.read_excel(steven_path+"Y_train_stev.xlsx")

X_test_unscaled_s = pd.read_excel(steven_path+"X_test_stev.xlsx")
y_test_unscaled_s = pd.read_excel(steven_path+"Y_test_stev.xlsx")

scaler_X_s = StandardScaler()
scaler_y_s = StandardScaler()

X_train_scaled_s = scaler_X_s.fit_transform(X_train_unscaled_s)
y_train_scaled_s = scaler_y_s.fit_transform(y_train_unscaled_s)
y_train_scaled_s = y_train_scaled_s.ravel()


X_test_scaled_s = scaler_X_s.fit_transform(X_test_unscaled_s)
y_test_scaled_s = scaler_y_s.fit_transform(y_test_unscaled_s)
y_test_scaled_s = y_test_scaled_s.ravel()


###################################### Alm and Hamre Dataset ###################################################

X_train_unscaled_a = pd.read_excel(alm_path+"X_train_alm.xlsx")
y_train_unscaled_a = pd.read_excel(alm_path+"Y_train_alm.xlsx")

X_test_unscaled_a = pd.read_excel(alm_path+"X_test_alm.xlsx")
y_test_unscaled_a = pd.read_excel(alm_path+"Y_test_alm.xlsx")

scaler_X_a = StandardScaler()
scaler_y_a = StandardScaler()

X_train_scaled_a = scaler_X_a.fit_transform(X_train_unscaled_a)
y_train_scaled_a = scaler_y_a.fit_transform(y_train_unscaled_a)
y_train_scaled_a = y_train_scaled_a.ravel()


X_test_scaled_a = scaler_X_a.fit_transform(X_test_unscaled_a)
y_test_scaled_a = scaler_y_a.fit_transform(y_test_unscaled_a)
y_test_scaled_a = y_test_scaled_a.ravel()

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
    y_pred_norm = myModel.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_norm.reshape(-1,1))
    
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
    r2 = r2_score(y_test_unscaled,y_pred)
    mae = mean_absolute_error(y_test_unscaled,y_pred)
    
    print("R2:", round(r2,2)*100,"%")
    print("Root mean Squared Error:", round(rmse,4))
    print("Mean Absolute Error:", round(mae,2))
    
    plt.figure(figsize=(16,9))
    plt.scatter(y_test_unscaled['Blowcount [Blows/m]'],y_pred)
    perfect_pred = np.linspace(min(y_test_unscaled['Blowcount [Blows/m]']),max(y_test_unscaled['Blowcount [Blows/m]']),1000)
    plt.plot(perfect_pred,perfect_pred,'k')
    plt.title("Evaluation of the SVR model",fontsize=30)
    plt.ylabel("Predicted values",fontsize=15)
    plt.xlabel("Actual Values",fontsize=15)
    plt.tick_params(labelsize=15)
    plt.grid()
    plt.savefig(figure_path+"results_SVR_"+model_name+".png",bbox_inches="tight")
    plt.show()
    
    ratio = np.array(y_pred/y_test_unscaled)[:,0]
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
    plt.title("Empirical Probabilities for the ratio: $\\frac{y_{predicted}}{y_{measured}}$ - SVR",fontsize=30)
    plt.grid()
    plt.legend()
    plt.savefig(figure_path+"proba_SVR_"+model_name+".png")
    plt.show()


def plot_tuning(grid_results,model_name):
    
    cv_results = grid_results['cv_results']
    test_rmse_mean = cv_results['test-RMSE-mean']
    test_rmse_std = cv_results['test-RMSE-std']
    train_rmse_mean = cv_results['train-RMSE-mean']
    train_rmse_std = cv_results['train-RMSE-std']
    
    fig,((ax0, ax1)) = plt.subplots(1,2,figsize=(25,9))
    
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


def plot_residuals(model,y_scaler,X_test_scaled,y_test_unscaled,model_name):   
    y_pred_norm = model.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_norm.reshape(-1,1))
    residuals = y_test_unscaled - y_pred
    
    y_pred_standardized = (y_pred - np.mean(y_pred)) / np.std(y_pred)
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred_standardized, y=standardized_residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
    plt.xlabel('Standardized Predicted Values')
    plt.ylabel('Standardized Residuals')
    plt.title('Residuals vs. Predicted Values')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.grid()
    plt.savefig(figure_path+"residuals_dist_svr_"+model_name,bbox_inches="tight")
    plt.show()

    # Residuals distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(standardized_residuals, kde=True, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid()
    plt.savefig(figure_path+"residuals_hist_svr_"+model_name,bbox_inches="tight")
    plt.show()

#########################################################################################################
############################### Explanation of a given model #############################################
def explanation(X_train,model,model_name):
    shap.initjs() #initiates the core of the shap module
    X_train_summary = shap.kmeans(X_train, 25) 
    explainer = shap.KernelExplainer(model.predict,X_train_summary)
    shap_values = explainer(X_train)
    fig = shap.summary_plot(shap_values)
    plt.savefig(figure_path+"feature_importance_svr_shap_"+model_name,bbox_inches="tight")
    # fig_2 = shap.dependence_plot("Qsi [kN]", shap_values.values, X_train)
    # plt.savefig(figure_path+"Qsi_svr_shap_"+model_name,bbox_inches="tight")

#########################################################################################################
################################# Creating the basic SVR model ##########################################

# svr = SVR(kernel='rbf', C=1.0, gamma='scale', coef0=0.0, degree=3)
# svr.fit(X_train_scaled_s, y_train_scaled_s)

#########################################################################################################
########################### HyperParameters Tuning Bayesian Opti ########################################


def tuning(model,X_train_scaled,y_train_scaled,model_name):
    max_iter = 500 #defines the number of iteration
    search_spaces = {
        'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
        'C': Real(1e-3, 1e3, prior='log-uniform'),
        'epsilon': Real(1e-3, 1, prior='log-uniform'),
        'gamma': Real(1e-4, 1, prior='log-uniform'),
        'degree': Integer(2, 5),
        'coef0': Real(0, 1)
    } #defines the space of research
    
    opt = BayesSearchCV(
    estimator=SVR(),
    search_spaces=search_spaces,
    n_iter=max_iter,  # Number of iterations
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
    ) #creates the optimizer
    
    np.int = int #to avoid error du to incompatibility of module version
    
    start_time = t.time()  #saves the execution time  
    opt.fit(X_train_scaled,y_train_scaled)
    end_time = t.time()
    
    execution_time = end_time-start_time
    
    time = pd.read_excel(saving_path+"4_Computational Cost\\Time.xlsx")
    new_row = pd.DataFrame({
    "Model": ["SVR "+model_name],
    "Execution Type": ["HP Tuning - Bay"],
    "N Trials": [max_iter],
    "Time [s]": [execution_time],
    "Figure":["Tuning Time"]})
    time = pd.concat([new_row,time],ignore_index=True)
    time.to_excel(saving_path+"4_Computational Cost\\Time.xlsx",index=False) #saves the time for latter plot

    #saves the best HP in a excel file
    best_params = opt.best_params_

    df = pd.DataFrame(index=[0],columns=list(best_params.keys()))
    for key in df.columns:
        df[key]=best_params[key]
    df.to_excel(models_path+"tuning_svr_"+model_name+".xlsx")
    return best_params

# stev_params = tuning(svr,X_train_scaled_s, y_train_scaled_s,"stev")
# alm_params = tuning(svr,X_train_scaled_a, y_train_scaled_a,"alm")


#########################################################################################################
#################################### Assessement of Steven Method #######################################

myModel =  SVR(kernel='rbf', C=1.0, gamma='scale', coef0=0.0,epsilon=1e-1, degree=3)
myModel.fit(X_train_scaled_s,y_train_scaled_s)
# assess_model(myModel,scaler_y_s,X_test_scaled_s,y_test_unscaled_s,"stev")
# plot_residuals(myModel,scaler_y_s,X_test_scaled_s,y_test_unscaled_s,"stev")
# explanation(X_test_scaled_s[0:200,:],myModel,"stev")

#########################################################################################################
################################ Assessement of Alm and Hamre Method ####################################

# myModel =  SVR(kernel='rbf', C=1.0, gamma='scale', coef0=0.0,epsilon=1e-1, degree=3)
# myModel.fit(X_train_scaled_a,y_train_scaled_a)
# assess_model(myModel,scaler_y_a,X_test_scaled_a,y_test_unscaled_s,"alm")
# plot_residuals(myModel,scaler_y_a,X_test_scaled_a,y_test_unscaled_a,"alm")
# explanation(X_train_scaled_a[0:200,:],myModel,"alm")

