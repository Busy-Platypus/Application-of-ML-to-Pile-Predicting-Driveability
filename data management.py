# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:14:02 2024

@author: Ronan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

my_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\3_Inputs\\0_Data\\0_Working file\\"

name_training_set = "training_data.csv"
name_test_set = "test_data.csv"

saving_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\Figures\\"
figure_path = "G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\Figures\\1_Final Try\\"

#save the data as an excel file without nan
def clean_csv(path):
    data = pd.read_csv(path,sep=",")
    empty_data = data.isna().sum()
    cleaned_data = data.dropna()
    cleaned_data.to_excel(path.split(".")[0]+"_cleaned.xlsx")


#########################################################################################################
###################### Adding the blowcounts to test dataset ############################################
def complete_test():
    solution_path = my_path+"2_Initial Data\\test_solutions.xlsx"
    test_path = my_path+"2_Initial Data\\test_data_cleaned.xlsx"
    
    solution_file = pd.read_excel(solution_path)
    test_file =  pd.read_excel(test_path)
    
    test_file = pd.merge(solution_file,test_file,on="ID",how="inner")
    test_file.to_excel(my_path+"2_Initial Data\\test_complete_data_cleaned.xlsx",index=False)
    
# complete_test()



#########################################################################################################
###################################Adding Features#######################################################
def steven_shaft_res(file_path):
    file = pd.read_excel(file_path)
    df = pd.DataFrame()
    
    int_angle = 25 #API friction angle for medium dense sand
    f_max = 81.3  #Maximum unit shaft friction [MPa] for medium dense sand
    K = 0.7 # Coefficient of lateral pressure taken as 0.7 for Stevens Method
    
    for location in file['Location ID'].unique(): #calculation for each pile
        temp = pd.DataFrame()  #creation of a temporary variable
        location_data = file[file['Location ID']==location].copy()
        temp["bvo [kPa]"] = 9 * location_data["z [m]"] #calculation of the effective overburden pressure 
        temp["f no lim [kPa]"] = K * np.tan(int_angle*np.pi/180) * temp["bvo [kPa]"] #uncapped value for the unit shaft friction
        temp["fsi [kPa]"] = temp["f no lim [kPa]"].clip(upper=f_max) #capped value for the unit skin friction
        
        location_data["Qsi [kN]"] = 0.5*np.pi*2.48*temp["fsi [kPa]"] #shaft resistance at the given location
        df = pd.concat([df,location_data])
        df = df.dropna()
        df.to_excel(file_path,index=False)
    return df

def stev_base_res(file_path):
    file = pd.read_excel(file_path)
    df = pd.DataFrame()
    N = 20 #bearing capacity factor for medium dense sand
    qb_lim = 5e3 #unit end bearing resistance limit for medium dense sand
    D = 2.48 #diameter of the pile 
    for location in file['Location ID'].unique():
        temp = pd.DataFrame() 
        location_data = file[file['Location ID']==location].copy()
        
        temp["qb [MPa]"] = N * 9 * location_data["z [m]"] #unit end base resistance
        t = location_data["Bottom wall thickness [mm]"].min()*1e-3 #bottom wall thickness for the pile
        A = 0.25*np.pi*(D**2-(D-2*t)**2) #calculation of the annulus area

        temp["Qb [kN]"] = A*temp["qb [MPa]"] #uncapped end bearing resistance 
        location_data["Qb [kN]"] = temp["Qb [kN]"].clip(upper=A*qb_lim) #capped end bearing resistance
        df = pd.concat([df,location_data])
    df.to_excel(file_path,index=False)
    return df


def alm_shaft_res(file_path):
    
    file = pd.read_excel(file_path)
    df = pd.DataFrame()
    int_angle = 29 #volume friction angle in degree
    
    for location in file['Location ID'].unique():
        temp = pd.DataFrame() 
        location_data = file[file['Location ID']==location].copy()
        
        temp["bvo [kPa]"] = 9* location_data["z [m]"] #effective overburden pressure 
        temp["qc [MPa]"] = location_data["qc [MPa]"] 
        temp["h"] = location_data["z [m]"].max()-location_data["z [m]"] #distance from depth to pile toe
        temp["z [m]"] = location_data["z [m]"]
        temp["k"]= 0.0125 * (location_data["qc [MPa]"]/ (temp["bvo [kPa]"]/1000))**0.5 #shape factor for degradation
        temp["Tfmax"]= 0.0132 * location_data["qc [MPa]"] * 1000 * ((temp["bvo [kPa]"]/100) **0.13) * np.tan(int_angle*np.pi/180) #unit initial pile side friction
        temp["Tres"] = 0.2*temp["Tfmax"] #unit residual pile side friction
        temp["fsi [kPa]"]=temp["Tres"]+(temp["Tfmax"]-temp["Tres"])*np.exp(-temp["k"]*temp["h"]) #unit pile side friction
        
        temp.to_excel("temp.xlsx")
        location_data["Qsi [kN]"]=0.5*np.pi*2.48*temp["fsi [kPa]"] #pile side friction 

        df = pd.concat([df,location_data])
        df = df.dropna()
        df.to_excel(file_path,index=False)
        
    return df

def alm_base_res(file_path):
    file = pd.read_excel(file_path)
    df = pd.DataFrame()
    D = 2.48
    for location in file['Location ID'].unique():
        temp = pd.DataFrame() 
        location_data = file[file['Location ID']==location].copy()
        temp["q tip [kPa]"] = location_data["qc [MPa]"]*1000
        temp["qb [kPa]"] = 0.15*temp["q tip [kPa]"]*(temp["q tip [kPa]"]/(location_data["z [m]"]*9))**0.2 #unit end  bearing resistance in kPa 
        location_data["qb [kPa]"]= temp["qb [kPa]"]
        temp["t [mm]"] = location_data["Bottom wall thickness [mm]"]*1e-3 #bottom wall thickness of the pile
        temp["A [m2]"] = 0.25*np.pi*(D**2-(D-2*temp['t [mm]'])**2) #anulus area of the pile
        
        location_data["Qb [kN]"] = temp["A [m2]"]*temp["qb [kPa]"] #end bearing resistance
        df = pd.concat([df,location_data])
    df.to_excel(file_path,index=False)
    return df
    

def tot_res_stev(file_path):
    file = pd.read_excel(file_path)
    df = pd.DataFrame()
    for location in file['Location ID'].unique():
        location_data = file[file['Location ID']==location].copy()
        for depth in location_data["z [m]"]:
            location_data.loc[location_data["z [m]"]==depth,["Qt [kN]"]]=location_data[location_data["z [m]"]==depth]["Qb [kN]"]
            +2*location_data[location_data["z [m]"]<=depth]["Qsi [kN]"].sum() #Soil resistance to Driving - Stevens Method - Upper Bound Corring
        df = pd.concat([df,location_data])
    df.to_excel(file_path,index=False)
    return df   

def tot_res_alm(file_path):
    file = pd.read_excel(file_path)
    df = pd.DataFrame()
    for location in file['Location ID'].unique():
        location_data = file[file['Location ID']==location].copy()
        for depth in location_data["z [m]"]:
            location_data.loc[location_data["z [m]"]==depth,["Qt [kN]"]]=location_data[location_data["z [m]"]==depth]["Qb [kN]"]
            +location_data[location_data["z [m]"]<=depth]["Qsi [kN]"].sum() # Soil resistance to Driving - Alm and Hamre 
        df = pd.concat([df,location_data])
    df.to_excel(file_path,index=False)
    return df    

def blow_ener(file_path):
    file = pd.read_excel(file_path)
    file["Blow/m/ENTHRU"] = file["Blowcount [Blows/m]"]/file["Normalised ENTRHU [-]"]
    file.to_excel(file_path, index=False)
    return file

def qt_ener(file_path):
    file = pd.read_excel(file_path)
    file["Qt/ener [kN]"] = file["Qt [kN]"]/file["Normalised ENTRHU [-]"]
    file.to_excel(file_path, index=False)
    return file

## Adding the Shaft Resistance According to the Stevens method
# steven_shaft_res(my_path+'0_Steven Model\\training_data_features_stev.xlsx')
# steven_shaft_res(my_path+'0_Steven Model\\test_data_features_stev.xlsx')

## Adding the Base Resistance According to the Stevens method
# stev_base_res(my_path+'0_Steven Model\\training_data_features_stev.xlsx')
# stev_base_res(my_path+'0_Steven Model\\test_data_features_stev.xlsx')

## Adding the Total Resistance to Stevens model
# tot_res_stev(my_path+'0_Steven Model\\training_data_features_stev.xlsx')
# tot_res_stev(my_path+'0_Steven Model\\test_data_features_stev.xlsx')

## Adding the Shaft Resistance According to the Alm and Hamre's method
# alm_shaft_res(my_path+'1_Alm and Hamre Model\\training_data_features_alm.xlsx')
# alm_shaft_res(my_path+'1_Alm and Hamre Model\\test_data_features_alm.xlsx')

## Adding the Base Resistance According to the Alm and Hamre's method
# alm_base_res(my_path+'1_Alm and Hamre Model\\training_data_features_alm.xlsx')
# alm_base_res(my_path+'1_Alm and Hamre Model\\test_data_features_alm.xlsx')

## Adding the Total Resistanceto Alm and Hamre model
# tot_res_alm(my_path+'1_Alm and Hamre Model\\training_data_features_alm.xlsx')
# tot_res_alm(my_path+'1_Alm and Hamre Model\\test_data_features_alm.xlsx')

## Adding blowcount / ener
# blow_ener(my_path+'0_Steven Model\\training_data_features_stev.xlsx')
# blow_ener(my_path+'1_Alm and Hamre Model\\training_data_features_alm.xlsx')


## Adding total resistance / ener
# qt_ener(my_path+'0_Steven Model\\training_data_features_stev.xlsx')
# qt_ener(my_path+'1_Alm and Hamre Model\\training_data_features_alm.xlsx')


#########################################################################################################
###################################Compare the p capacity models#########################################


def compare_qsi(location):
    stev_path = my_path + '0_Steven Model\\training_data_features_stev.xlsx'
    alm_path  = my_path+'1_Alm and Hamre Model\\training_data_features_alm.xlsx'
    
    stev_file = pd.read_excel(stev_path)
    alm_file = pd.read_excel(alm_path)
    
    plt.figure(figsize=(9,13))
    #plotting the values for a given location
    plt.plot(stev_file[stev_file["Location ID"]==location]["Qsi [kN]"],stev_file[stev_file["Location ID"]==location]["z [m]"],label="Stevens Method")
    plt.plot(alm_file[alm_file["Location ID"]==location]["Qsi [kN]"],alm_file[alm_file["Location ID"]==location]["z [m]"],label="Alm and Hamre Method")
    plt.grid()
    plt.legend(fontsize=15)
    plt.xlabel("$Q_{si}$ [kN]",fontsize=15)
    plt.ylabel("z [m]",fontsize=15)
    plt.ylim([35,0])
    plt.title("$Q_{s,i}$ comparison of Steven with Alm and Hamre Method",fontsize=20)
    plt.savefig(figure_path+"qsi method comparison.png",bbox_inches="tight")
    plt.show()
        
def compare_qb(location):
    stev_path = my_path + '0_Steven Model\\training_data_features_stev.xlsx'
    alm_path  = my_path+'1_Alm and Hamre Model\\training_data_features_alm.xlsx'
    
    stev_file = pd.read_excel(stev_path)
    alm_file = pd.read_excel(alm_path)
    #plotting the values for a given location   
    plt.figure(figsize=(9,13))
    plt.plot(stev_file[stev_file["Location ID"]==location]["Qb [kN]"],stev_file[stev_file["Location ID"]==location]["z [m]"],label="Stevens Method")
    plt.plot(alm_file[alm_file["Location ID"]==location]["Qb [kN]"],alm_file[alm_file["Location ID"]==location]["z [m]"],label="Alm and Hamre Method")
    plt.grid()
    plt.legend(fontsize=15)
    plt.xlabel("$Q_{b}$ [kN]",fontsize=15)
    plt.ylabel("z [m]",fontsize=15)
    plt.ylim([35,0])
    plt.title("$Q_{b}$ comparison of Steven with Alm and Hamre Method",fontsize=20)
    plt.savefig(figure_path+"qb method comparison.png",bbox_inches="tight")
    plt.show()
    
def compare_qt(location):
    stev_path = my_path + '0_Steven Model\\training_data_features_stev.xlsx'
    alm_path  = my_path+'1_Alm and Hamre Model\\training_data_features_alm.xlsx'
    
    stev_file = pd.read_excel(stev_path)
    alm_file = pd.read_excel(alm_path)
    #plotting the values for a given location
    plt.figure(figsize=(9,13))
    plt.plot(stev_file[stev_file["Location ID"]==location]["Qt [kN]"],stev_file[stev_file["Location ID"]==location]["z [m]"],label="Stevens Method")
    plt.plot(alm_file[alm_file["Location ID"]==location]["Qt [kN]"],alm_file[alm_file["Location ID"]==location]["z [m]"],label="Alm and Hamre Method")
    plt.grid()
    plt.legend(fontsize=15)
    plt.xlabel("$Q_{t}$ [kN]",fontsize=15)
    plt.ylabel("z [m]",fontsize=15)
    plt.ylim([35,0])
    plt.title("$Q_{t}$ comparison of Steven with Alm and Hamre Method",fontsize=20)
    plt.savefig(figure_path+"qt method comparison.png",bbox_inches="tight")
    plt.show()
    
# compare_qsi("AA")
# compare_qb("AA")
# compare_qt("AA")
#########################################################################################################
###################################Creating Input and Output#############################################

##Splitting Input and Output dataset
drop_list = ["ID","Location ID","Blowcount [Blows/m]"]
def in_out(file_path,model,type_data,drop_list):
    dataset = pd.read_excel(file_path)
    X_pd = dataset.drop(columns=drop_list)
    X_pd.to_excel(my_path+"X_"+type_data+"_"+model+".xlsx",index=False)
    
    drop_columns_y = list(dataset.columns)
    drop_columns_y.remove('Blowcount [Blows/m]')
    Y_pd = dataset.drop(columns=drop_columns_y)
    Y_pd.to_excel(my_path+"Y_"+type_data+"_"+model+".xlsx",index=False)


# in_out(my_path+'0_Steven Model\\training_data_features_stev.xlsx',"stev","train",drop_list)
# in_out(my_path+'0_Steven Model\\test_data_features_stev.xlsx',"stev","test",drop_list)
# in_out(my_path+'1_Alm and Hamre Model\\training_data_features_alm.xlsx',"alm","train",drop_list)
# in_out(my_path+'1_Alm and Hamre Model\\test_data_features_alm.xlsx',"alm","test",drop_list)
# in_out(my_path+"2_Initial Data\\test_complete_data_cleaned.xlsx","no_feature","test",drop_list)

#########################################################################################################
###################################Graphic visualisation Corr Matrix#####################################

def correlation_ds(file_path,model_name):
    ds = pd.read_excel(file_path)
    ds = ds.drop(columns=["ID","Location ID","Diameter [m]"])
    new_order = ["z [m]","qc [MPa]","fs [MPa]","u2 [MPa]","Normalised ENTRHU [-]","Normalised hammer energy [-]","Bottom wall thickness [mm]","Pile penetration [m]","Qsi [kN]","Qb [kN]","Qt [kN]","Qt/ener [kN]","Blowcount [Blows/m]","Number of blows","Blow/m/ENTHRU"]
    ds = ds[new_order] #changing the order to separate input / output
    corrmat = ds.corr(min_periods=1) #calculation of the correlation matrix
    fig, ax = plt.subplots(figsize=(15,13)) #plotting the matrix as heatmap
    sns.set_context("notebook", font_scale=0.7, rc={"lines.linewidth": 1.5})
    sns.heatmap(corrmat, annot=True, square=True,cmap='coolwarm',vmin=-1, vmax=1)
    plt.title("Correlation Matrix of the features ("+model_name+")",fontsize=30)
    plt.savefig(saving_path+"Correlation matrix_"+model_name+".png",bbox_inches="tight")
    fig.tight_layout()
    fig.show()

# correlation_ds(my_path+'0_Steven Model\\training_data_features_stev.xlsx',"Stevens Model") 
# correlation_ds(my_path+'1_Alm and Hamre Model\\training_data_features_alm.xlsx',"Alm and Hamre Model")  
 

#########################################################################################################
################################# Importance of the Features ############################################

# X_pd_stev = pd.read_excel(my_path+"0_Steven Model\\X_train_stev.xlsx")
# Y_pd_stev = pd.read_excel(my_path+"0_Steven Model\\Y_train_stev.xlsx")

# X_pd_alm = pd.read_excel(my_path+"1_Alm and Hamre Model\\X_train_alm.xlsx")
# Y_pd_alm = pd.read_excel(my_path+"1_Alm and Hamre Model\\Y_train_alm.xlsx")

def rf_importance(X,y,model_name):
    #using random forest to assess the importance of the features
    model = RandomForestRegressor() #creating a regressor with default values
    model.fit(X, y) #fitting the model
    importances = model.feature_importances_ #saving the feature relative importance
      
    fig = plt.figure(figsize=(16,9)) #plotting the results
    columns = list(X.columns)

    indices = np.argsort(importances)[::-1] #sorting by ascending relative importance
    plt.bar(range(X.shape[1]), importances[indices], color="b", align="center")
    plt.xticks(range(len(columns)), [columns[i] for i in indices], rotation=45,fontsize=10)
    plt.title("Feature Importance for Blowcount/m with Random Forest ("+model_name+")",fontsize=20)
    plt.tight_layout()
    plt.grid()
    plt.savefig(saving_path+"Feature importance RF-"+model_name,bbox_inches="tight")
    plt.show()

    return importances

# importance = rf_importance(X_pd_stev, Y_pd_stev,"Stevens Model")
# importance = rf_importance(X_pd_alm, Y_pd_alm,"Alm and Hamre Model")



#########################################################################################################
###############################Graphic visualisation Input Feeatures#####################################

def graph_depth(input_path,output_path,loc):
    
    data = pd.read_excel(input_path)
    pile_data = data[data["Location ID"]==loc] #select the data for particular depth
    
    fig,((ax0, ax1)) = plt.subplots(1, 2, sharey=True, figsize=(16,9))
    
    ax0.scatter(data["Blowcount [Blows/m]"], data["z [m]"], s=3) #plotting the blowcount/m
    ax1.scatter(data["qc [MPa]"], data["z [m]"], s=3) #plotting tip resistance
    
    #plotting the particular location ID = loc
    ax0.plot(pile_data["Blowcount [Blows/m]"], pile_data["z [m]"],'r-',label="loc="+loc)
    ax1.plot(pile_data["qc [MPa]"], pile_data["z [m]"],'r-',label="loc="+loc)
    
    ax0.set_xlabel("Blowcount (Blows/m)",fontsize=15)
    ax1.set_xlabel("$ q_c $ (MPa)",fontsize=15)
    ax0.set_ylabel("Depth below mudline, $z$ (m)",fontsize=15)
    
    for ax in (ax0, ax1):
        ax.xaxis.tick_top()
        ax.set_ylim(35, 0)
        ax.grid()
        ax.legend()
        ax.tick_params(axis='both', labelsize=15)
        
       
    plt.suptitle("Evolution of the features with depth",fontsize=30)
    plt.savefig(output_path,bbox_inches="tight")

# graph_depth(my_path+"2_Initial Data\\"+"training_data_cleaned.xlsx",saving_path+"Figure 1.png","BG")

def graph_blowcount(input_path,output_path,loc):
    data = pd.read_excel(input_path)
    pile_data = data[data["Location ID"]==loc]
    
    fig,((ax0, ax1, ax2)) = plt.subplots(1, 3, sharey=True, figsize=(20,9))
    
    ax0.scatter(data["Normalised ENTRHU [-]"], data["Blowcount [Blows/m]"], s=10)
    ax1.scatter(data["Rs [kN]"], data["Blowcount [Blows/m]"], s=10)
    ax2.scatter(data["fs [MPa]"], data["Blowcount [Blows/m]"], s=10)
    
    #plotting the particular location ID = loc
    ax0.scatter(pile_data["Normalised ENTRHU [-]"], pile_data["Blowcount [Blows/m]"],color='r',label="loc="+loc)
    ax1.scatter(pile_data["Rs [kN]"], pile_data["Blowcount [Blows/m]"],color='r',label="loc="+loc)
    ax2.scatter(pile_data["fs [MPa]"], pile_data["Blowcount [Blows/m]"],color='r',label="loc="+loc)
    
    ax0.set_xlabel("Normalised ENTRHU",fontsize=15)
    ax1.set_xlabel("$ R_s $ (kN)",fontsize=15)
    ax2.set_xlabel("Sleeve friction",fontsize=15)
        
    for ax in (ax0, ax1, ax2):
        ax.xaxis.tick_top()
        ax.set_ylim(0,200)
        ax.grid()
        ax.legend()
        ax.tick_params(axis='both', labelsize=15)

    
    plt.suptitle("Evolution of the feature with blowcounts",fontsize=30)
    plt.savefig(output_path,bbox_inches="tight")

# graph_blowcount(my_path+"0_Steven Model\\"+"training_data_features_stev.xlsx",saving_path+"Figure 2.png","BG",)


#########################################################################################################
###############################Assessment of the Descriptive Stats ######################################


def descriptive_stat(input_path,output_path):
    data = pd.read_excel(input_path)
    stat = data.describe().transpose() #assess the descriptive statistics of the dataset
    stat.to_excel(output_path)
    return data,stat


# descriptive_stat(my_path+"0_Steven Model\\"+"training_data_features_stev.xlsx","G:\\Mon Drive\\Trinity\\0_Master Thesis\\5_Outputs\\Descriptive Statistics.xlsx")