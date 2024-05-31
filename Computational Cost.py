# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:09:48 2024

@author: Ronan
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

saving_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\"
figure_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\Figures\\1_Final Try\\"
time = pd.read_excel(saving_path+"4_Computational Cost\\Time.xlsx")

def ANN_units_impact(time):
    fig = plt.figure(figsize=(16,9))
    units = time[time["Figure"]=="Impact of units"]["units"]
    time_cost = time[time["Figure"]=="Impact of units"]["Time [s]"]
    plt.plot(units,time_cost)
    plt.xlabel("Number of units used",fontsize=15)
    plt.ylabel("Execution Time [s]",fontsize=15)
    plt.title("Impact of the neuron number on time cost", fontsize=20)
    plt.xlim([10,600])
    plt.xscale("log")
    plt.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(figure_path+"Impact neuron number.png")
    plt.show()
    
# ANN_units_impact(time)

def tuning_time(time):
    fig = plt.figure(figsize=(16,9))
    models = time[time["Figure"]=="Tuning Time"]["Model"].values
    time_cost = time[time["Figure"]=="Tuning Time"]["Time [s]"].values
    
    indices = np.argsort(time_cost)[::-1]
    
    plt.bar(models[indices], time_cost[indices], color='skyblue')
    plt.xlabel('Method used',fontsize=15)
    plt.ylabel('Mean Execution Time [s] / iteration',fontsize=15)
    plt.title('Computational Cost',fontsize=30)
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.tick_params(labelsize=10)
    plt.savefig(figure_path+"Tuning Time.png")
    plt.show()

tuning_time(time)

def bay_grid_time(time):
    fig = plt.figure(figsize=(16,9))
    models = time[time["Figure"]=="Advantage of Bayesian Opti"]["Execution Type"]
    time_cost = time[time["Figure"]=="Advantage of Bayesian Opti"]["Time [s]"]
    
    plt.bar(models, time_cost, color='skyblue')
    plt.xlabel('Method used',fontsize=15)
    plt.ylabel('Execution Time [s]',fontsize=15)
    plt.title('Comparison of execution time Bayesian vs. Grid Search',fontsize=30)
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.tick_params(labelsize=10)
    plt.savefig(figure_path+"Bayesian over GS")
    plt.show()
    

# bay_grid_time(time)
