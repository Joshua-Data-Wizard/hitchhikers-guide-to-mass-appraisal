####################################################################################################
##################################### NAL Navigator ################################################
####################################################################################################
# Authors = Luke Jorgensen and Joshua Jorgensen
# Version = 0.0.1
### Helpful Links
## DOR NAL Link: https://floridarevenue.com/property/dataportal/Pages/default.aspx?path=/property/dataportal/Documents/PTO%20Data%20Portal/Tax%20Roll%20Data%20Files/NAL/2024F
## DOR File Link: https://floridarevenue.com/property/dataportal/Pages/default.aspx?path=/property/dataportal/Documents/PTO%20Data%20Portal/Tax%20Roll%20Data%20Files
## Streamlit Emoji Short Codes : https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
### NOTES
## 1. Working On Building the Models for Condos and Commercial Properties


## Loading Needed Libareies

## Data Manipulation 
import numpy as np
import pandas as pd

## Graphical User Interface
import streamlit as st
import altair as alt

## image rendring 
import requests
from streamlit_lottie import st_lottie

## Geography 
import geopandas as gpd
from pyproj import CRS, Transformer

## System 
import os
#
# Read a .env file if one is present, so secrets and settings live in a
# gitignored file instead of in this source code. See .env.example.
# Nothing breaks if there is no .env - every lookup below has a default.
from dotenv import load_dotenv
load_dotenv()
import psutil
import time
## The `keyboard` package was removed from this import block and must not
## come back. The shutdown button near the bottom of this file explains why.

# Stats
import statistics
import scipy.stats as stats

# Pre Processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

## Importing Training and testing split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# Models 
import lightgbm as ltb

# Model optimization
import optuna

# Model Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Local Database 
import sqlite3

## Google GenAI 
from google import genai
from google.genai import types
from google.api_core import retry 

## Visualizing Data 
## Visualizing Data
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

#st.set_page_config(layout="wide")

## IF RUNNING LATEST WINDOWS VERSION OF STREAMLIT, UNCOMMENT THE CODE BELOW
#os.environ['LOKY_MAX_CPU_COUNT'] = '10'

## Application Variables 


if 'SFR_model_binary' not in st.session_state:
    st.session_state["SFR_model_binary"] = False

if 'Vac_Res_model_binary' not in st.session_state:
    st.session_state["Vac_Res_model_binary"] = False

if 'Condo_model_binary' not in st.session_state:
    st.session_state["Condo_model_binary"] = False

if 'Commercial_model_binary' not in st.session_state:
    st.session_state["Commercial_model_binary"] = False   

if 'preprocesing_ran' not in st.session_state:
    st.session_state["preprocesing_ran"] = False

if 'Model_ran' not in st.session_state:
    st.session_state["Model_ran"] = False

if 'NAL' not in st.session_state:
    st.session_state["NAL"] = None

if 'SDF' not in st.session_state:
    st.session_state["SDF"] = None

if 'GIS' not in st.session_state:
    st.session_state["GIS"] = None

if 'NAL_merged' not in st.session_state:
    st.session_state["NAL_merged"] = None

if 'NBHD_Encoder' not in st.session_state:
    st.session_state['NBHD_Encoder'] = None

if 'sfr_preformance_dict' not in st.session_state:
    st.session_state["sfr_preformance_dict"] = None

if "SFR_Model_Predicitons_All" not in st.session_state:
    st.session_state["SFR_Model_Predicitons_All"] = None

if "SFR_Model_Predicitons_For_Review" not in st.session_state:
    st.session_state["SFR_Model_Predicitons_For_Review"] = None

if 'Vac_Res_preformance_dict' not in st.session_state:
    st.session_state["Vac_Res_preformance_dict"] = None

if "Vac_Res_Model_Predicitons_All" not in st.session_state:
    st.session_state["Vac_Res_Model_Predicitons_All"] = None

if 'Condo_preformance_dict' not in st.session_state:
    st.session_state["Condo_preformance_dict"] = None

if "Condo_Model_Predicitons_All" not in st.session_state:
    st.session_state["Condo_Model_Predicitons_All"] = None

if 'Commercial_preformance_dict' not in st.session_state:
    st.session_state["Commercial_preformance_dict"] = None

if "Commercial_Model_Predicitons_All" not in st.session_state:
    st.session_state["Commercial_Model_Predicitons_All"] = None

if "DOR_Stats_Toggle" not in st.session_state:
    st.session_state['DOR_Stats_Toggle'] = False

if "Model_Performance_Toggle" not in st.session_state:
    st.session_state['Model_Performance_Toggle'] = False

if "Cost_Of_Sale_Adjustment_Toggle" not in st.session_state:
   st.session_state['Cost_Of_Sale_Adjustment_Toggle'] = False 

if "Stratum1_report" not in st.session_state:
    st.session_state['Stratum1_report'] = None

if "Stratum1GroupLevelStatistics_Report" not in st.session_state:
    st.session_state['Stratum1GroupLevelStatistics_Report'] = None      

if "Stratum1_report_Condos" not in st.session_state:
    st.session_state['Stratum1_report_Condos'] = None

if "Stratum1GroupLevelStatistics_Report_Condos" not in st.session_state:
    st.session_state['Stratum1GroupLevelStatistics_Report_Condos'] = None      

if "Stratum4_report" not in st.session_state:
    st.session_state['Stratum4_report'] = None

if "Stratum4GroupLevelStatistics_Report" not in st.session_state:
    st.session_state['Stratum4GroupLevelStatistics_Report'] = None      

if "Stratum6_report" not in st.session_state:
    st.session_state['Stratum6_report'] = None

if "Stratum6GroupLevelStatistics_Report" not in st.session_state:
    st.session_state['Stratum6GroupLevelStatistics_Report'] = None     


if "PSWRD_Entry" not in st.session_state:
    ## Starts empty. Whether a login is ever ASKED for depends on
    ## User_Password_List further down - see the note there.
    st.session_state['PSWRD_Entry'] = None

if "User_Name_Entry" not in st.session_state:
    st.session_state['User_Name_Entry'] = None  

if "Password_Count" not in st.session_state:
    st.session_state['Password_Count'] = 0

if "Valuation_Year" not in st.session_state:
    st.session_state['Valuation_Year'] = 2025

if "County_Selected" not in st.session_state:
    st.session_state['County_Selected'] = None

if "State_Coords_Selected" not in st.session_state:
    st.session_state['State_Coords_Selected'] = None

if "EPSG_List" not in st.session_state:
    st.session_state['EPSG_List'] = ["EPSG:2236" ,"EPSG:2237","EPSG:2238" ]

if "COS_ADJ_KEY" not in st.session_state:
    st.session_state["COS_ADJ_KEY"] = .85

if  "current_Adjust_dor" not in st.session_state:
    st.session_state['current_Adjust_dor'] = None

if "Hyperparameter_Tuning_Toggle" not in st.session_state:
    st.session_state['Hyperparameter_Tuning_Toggle'] = False 


if "Hyperparameter_Tuning_Trials" not in st.session_state:
    st.session_state['Hyperparameter_Tuning_Trials'] = 150

if "HyperParametet_Tuning_N_Estimators_Low" not in st.session_state:
    st.session_state['HyperParametet_Tuning_N_Estimators_Low'] = 250

if "HyperParametet_Tuning_N_Estimators_High" not in st.session_state:
    st.session_state['HyperParametet_Tuning_N_Estimators_High'] = 1500

if "HyperParametet_Tuning_num_leaves_Low" not in st.session_state:
    st.session_state['HyperParametet_Tuning_num_leaves_Low'] = 10

if "HyperParametet_Tuning_num_leaves_High" not in st.session_state:
    st.session_state['HyperParametet_Tuning_num_leaves_High'] = 100

if "HyperParametet_Tuning_learning_rate_Low" not in st.session_state:
    st.session_state['HyperParametet_Tuning_learning_rate_Low'] =.01

if "HyperParametet_Tuning_learning_rate_High" not in st.session_state:
    st.session_state['HyperParametet_Tuning_learning_rate_High'] = .3

if "HyperParametet_Tuning_subsample_Low" not in st.session_state:
    st.session_state['HyperParametet_Tuning_subsample_Low'] =.66

if "HyperParametet_Tuning_subsample_High" not in st.session_state:
    st.session_state['HyperParametet_Tuning_subsample_High'] = 1

if "HyperParametet_Tuning_feature_fraction_Low" not in st.session_state:
    st.session_state['HyperParametet_Tuning_feature_fraction_Low'] =.7

if "HyperParametet_Tuning_feature_fraction_High" not in st.session_state:
    st.session_state['HyperParametet_Tuning_feature_fraction_High'] = 1


if "HyperParametet_Tuning_max_depth_Low" not in st.session_state:
    st.session_state['HyperParametet_Tuning_max_depth_Low'] = 3

if "HyperParametet_Tuning_max_depth_High" not in st.session_state:
    st.session_state['HyperParametet_Tuning_max_depth_High'] = 20

if "HyperParametet_Tuning_min_data_in_leaf_Low" not in st.session_state:
    st.session_state['HyperParametet_Tuning_min_data_in_leaf_Low'] = 3

if "HyperParametet_Tuning_min_data_in_leaf_High" not in st.session_state:
    st.session_state['HyperParametet_Tuning_min_data_in_leaf_High'] = 30

if "HyperParametet_Tuning_min_sum_hessian_in_leaf_Low" not in st.session_state:
    st.session_state['HyperParametet_Tuning_min_sum_hessian_in_leaf_Low'] = .1

if "HyperParametet_Tuning_min_sum_hessian_in_leaf_High" not in st.session_state:
    st.session_state['HyperParametet_Tuning_min_sum_hessian_in_leaf_High'] = 20


if 'Model_Features' not in st.session_state:
    st.session_state['Model_Features'] =  ['MKT_AR','NBRHD_CD','IMP_QUAL','DOR_UC','Act_Age','LND_SQFOOT','TOT_LVG_AREA','Time_to_Sale', 'Age_Diff']


if 'Model_Features_Condos' not in st.session_state:
    st.session_state['Model_Features_Condos'] = ['MKT_AR','NBRHD_CD','IMP_QUAL','DOR_UC','Act_Age','TOT_LVG_AREA','Time_to_Sale', 'Age_Diff']

if 'Model_Features_Vac_Res' not in st.session_state:
    st.session_state['Model_Features_Vac_Res'] = ['MKT_AR','NBRHD_CD','LND_SQFOOT','DOR_UC','Time_to_Sale']

if 'Model_Features_Commercial' not in st.session_state:
    st.session_state['Model_Features_Commercial'] = ['MKT_AR','NBRHD_CD','IMP_QUAL','DOR_UC','Act_Age','LND_SQFOOT','TOT_LVG_AREA','Time_to_Sale', 'Age_Diff']


if  "Chat_Launched" not in st.session_state:
    st.session_state['Chat_Launched'] = None  

if  "Google_Key" not in st.session_state:
    st.session_state['Google_Key'] = None 

if  "Provided_Google_Key" not in st.session_state:
    st.session_state['Provided_Google_Key'] = None 

if "Chat_Model_Selected" not in st.session_state:
        st.session_state['Chat_Model_Selected'] = None

if "Chat_Bot_Data_Base" not in st.session_state:
        st.session_state['Chat_Bot_Data_Base'] = None

if "NAL_Chat_Bot" not in st.session_state:
        st.session_state['NAL_Chat_Bot'] = None

if "Chat_History" not in st.session_state:
    st.session_state["Chat_History"] = []

if "Display_Plot" not in st.session_state:
    st.session_state["Display_Plot"] = 0

if "Chat_Bot_Plot" not in st.session_state:
    st.session_state["Chat_Bot_Plot"] = 0

if 'County_List_Loc' not in st.session_state:
    st.session_state['County_List_Loc']  = {"Alachua (11)" : "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_11Alachua_F.csv"
                                            ,"Baker (12)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_12Baker_F.csv"
                                            , "Bay (13)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_13Bay_F.csv"
                                            , "Bradford (14)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_14Bradford_F.csv"
                                            , "Brevard (15)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_15Brevard_F.csv"
                                            , "Broward (16)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_16Broward_F.csv"
                                            , "Calhoun (17)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_17Calhoun_F.csv"
                                            , "Charlotte (18)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_18Charlotte_F.csv"
                                            , "Citrus (19)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_19Citrus_F.csv"
                                            , "Clay (20)"   : "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_20Clay_F.csv"
                                            , "Collier (21)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_21Collier_F.csv"
                                            , "Columbia (22)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_22Columbia_F.csv"
                                            , "Dade (23)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_23Dade_F.csv"
                                            , "DeSoto (24)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_24DeSoto_F.csv"
                                            , "Dixie (25)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_25Dixie_F.csv"
                                            , "Duval (26)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_26Duval_F.csv"
                                            , "Escambia (27)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_27Escambia_F.csv"
                                            , "Flagler (28)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_28Flagler_F.csv"
                                            , "Franklin (29)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_29Franklin_F.csv"
                                            , "Gadsden (30)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_30Gadsden_F.csv"
                                            , "Gilchrist (31)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_31Gilchrist_F.csv"
                                            , "Glades (32)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_32Glades_F.csv"
                                            , "Gulf (33)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_33Gulf_F.csv"
                                            , "Hamilton (34)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_34Hamilton_F.csv"
                                            , "Hardee (35)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_35Hardee_F.csv"
                                            , "Hendry (36)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_36Hendry_F.csv"
                                            , "Hernando (37)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_37Hernando_F.csv"
                                            ,"Hihglands (38)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_38Highlands_F.csv"
                                            , "Hillsborough (39)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_39Hillsborough_F.csv"
                                            , "Holmes (40)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_40Holmes_F.csv"
                                            , "Indian River (41)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_41Indian River_F.csv"
                                            , "Jackson (42)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_42Jackson_F.csv"
                                            , "Jefferson (43)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_43Jefferson_F.csv"
                                            , "Lafayette (44)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_44Lafayette_F.csv"
                                            , "Lake (45)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_45Lake_F.csv"
                                            , "Lee (46)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_46Lee_F.csv"
                                            , "Leon (47)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_47Leon_F.csv"
                                            , "Levy (48)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_48Levy_F.csv"
                                            , "Liberty (49)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_49Liberty_F.csv"
                                            , "Madison (50)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_50Madison_F.csv"
                                            , "Manatee (51)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_51Manatee_F.csv"
                                            , "Marion (52)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_52Marion_F.csv"
                                            , "Martin (53)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_53Martin_F.csv"
                                            , "Monroe (54)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_54Monroe_F.csv"
                                            , "Nassau (55)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_55Nassau_F.csv"
                                            , "Okaloosa (56)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_56Okaloosa_F.csv"
                                            , "Okeechobee (57)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_57Okeechobee_F.csv"
                                            , "Orange (58)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_58Orange_F.csv"
                                            , "Osceola (59)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_59Osceola_F.csv"
                                            , "Palm Beach (60)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_60Palm Beach_F.csv"
                                            , "Pasco (61)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_61Pasco_F.csv"
                                            , "Pinellas (62)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_62Pinellas_F.csv"
                                            , "Polk (63)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_63Polk_F.csv"
                                            , "Putnam (64)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_64Putnam_F.csv"
                                            , "St. Johns (65)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_65Saint Johns_F.csv"
                                            , "St. Lucie (66)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_66Saint Lucie_F.csv"
                                            , "Santa Rosa (67)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_67Santa Rosa_F.csv"
                                            , "Sarasota (68)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_68Sarasota_F.csv"
                                            , "Seminole (69)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_69Seminole_F.csv"
                                            , "Sumter (70)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_70Sumter_F.csv"
                                            , "Suwannee (71)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_71Suwannee_F.csv"
                                            , "Taylor (72)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_72Taylor_F.csv"
                                            , "Union (73)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_73Union_F.csv"
                                            , "Volusia (74)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_74Volusia_F.csv"
                                            , "Wakulla (75)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_75Wakulla_F.csv"
                                            , "Walton (76)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_76Walton_F.csv"
                                            , "Washington (77)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2024NAL\\2024F\\NAL_2024_77Washington_F.csv"}


if  'DOR_COS_ADJ_DICT' not in st.session_state:
    st.session_state["DOR_COS_ADJ_DICT"] = {'00':.85
                                            ,'01':.85
                                            ,'02':.85
                                            ,'03':.85
                                            ,'04':.85
                                            ,'05':.85
                                            ,'06':.85
                                            ,'07':.85
                                            ,'08':.85
                                            ,'09':.85
                                            ,'10':.85
                                            ,'11':.85
                                            ,'12':.85
                                            ,'13':.85
                                            ,'14':.85
                                            ,'15':.85
                                            ,'16':.85
                                            ,'17':.85
                                            ,'18':.85
                                            ,'19':.85
                                            ,'20':.85
                                            ,'21':.85
                                            ,'22':.85
                                            ,'23':.85
                                            ,'24':.85
                                            ,'25':.85
                                            ,'26':.85
                                            ,'27':.85
                                            ,'28':.85
                                            ,'29':.85
                                            ,'30':.85
                                            ,'31':.85
                                            ,'32':.85
                                            ,'33':.85
                                            ,'34':.85
                                            ,'35':.85
                                            ,'36':.85
                                            ,'37':.85
                                            ,'38':.85
                                            ,'39':.85
                                            ,'40':.85
                                            ,'41':.85
                                            ,'42':.85
                                            ,'43':.85
                                            ,'44':.85
                                            ,'45':.85
                                            ,'46':.85
                                            ,'47':.85
                                            ,'48':.85
                                            ,'49':.85
                                            ,'50':.85
                                            ,'51':.85
                                            ,'52':.85
                                            ,'53':.85
                                            ,'54':.85
                                            ,'55':.85
                                            ,'56':.85
                                            ,'57':.85
                                            ,'58':.85
                                            ,'59':.85
                                            ,'60':.85
                                            ,'61':.85
                                            ,'62':.85
                                            ,'63':.85
                                            ,'64':.85
                                            ,'65':.85
                                            ,'66':.85
                                            ,'67':.85
                                            ,'68':.85
                                            ,'69':.85
                                            ,'70':.85
                                            ,'71':.85
                                            ,'72':.85
                                            ,'73':.85
                                            ,'74':.85
                                            ,'75':.85
                                            ,'76':.85
                                            ,'77':.85
                                            ,'78':.85
                                            ,'79':.85
                                            ,'80':.85
                                            ,'81':.85
                                            ,'82':.85
                                            ,'83':.85
                                            ,'84':.85
                                            ,'85':.85
                                            ,'86':.85
                                            ,'87':.85
                                            ,'88':.85
                                            ,'89':.85
                                            ,'90':.85
                                            ,'91':.85
                                            ,'92':.85
                                            ,'93':.85
                                            ,'94':.85
                                            ,'95':.85
                                            ,'96':.85
                                            ,'97':.85
                                            ,'98':.85
                                            ,'99':.85
                                            
                                            } 

    

Result_Prescision = 4

## DOR_UC = DOR Unit Code
## QualCD = Qualification Sale Code



## ── The optional login gate ──────────────────────────────────────────────
##
## There is NO login screen unless you set BOTH of these in your .env:
##
##     HGMA_APP_USER=some_name
##     HGMA_APP_PASSWORD=some_password
##
## Leave them unset - the default - and the app opens straight to the data.
##
## 🔒 Credentials are never written in this file. This repository is public,
## and a password committed to a public repo is not a password, it is a
## published fact. An earlier version of this app shipped a real-looking
## username and password on this exact line.
##
## And to be clear about what this is: a courtesy screen for a shared demo
## laptop, NOT security. The app reads public taxroll data and serves it on
## localhost. Do not put anything behind it that would matter if seen.
_app_user = os.environ.get('HGMA_APP_USER')
_app_password = os.environ.get('HGMA_APP_PASSWORD')

User_Password_List = [[_app_user, _app_password]] if (_app_user and _app_password) else []

NAL_Features = ['MKT_AR','NBRHD_CD','CENSUS_BK','IMP_QUAL','DOR_UC','ACT_YR_BLT','EFF_YR_BLT','LND_SQFOOT','TOT_LVG_AREA' , 'Age_Diff' , 'Act_Age', 'Eff_Age','JV']

Sale_Features = ['SALE_MO','SALE_YR','SALE_PRC','VI_CD','SAL_CHG_CD','QUAL_CD','Time_to_Sale']  

Stratification_Variable = "SALE_YR"



#Model_Outcome = ['SALE_PRC']

Model_Outcome = ['Adjusted_Sale_Price']




def Update_County_List_Loc():
    if st.session_state['Valuation_Year'] != 2025: 
        st.session_state['County_List_Loc'] = {"Alachua (11)" : "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_11Alachua_F.csv"
                                            ,"Baker (12)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_12Baker_F.csv"
                                            , "Bay (13)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_13Bay_F.csv"
                                            , "Bradford (14)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_14Bradford_F.csv"
                                            , "Brevard (15)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_15Brevard_F.csv"
                                            , "Broward (16)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_16Broward_F.csv"
                                            , "Calhoun (17)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_17Calhoun_F.csv"
                                            , "Charlotte (18)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_18Charlotte_F.csv"
                                            , "Citrus (19)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_19Citrus_F.csv"
                                            , "Clay (20)"   : "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_20Clay_F.csv"
                                            , "Collier (21)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_21Collier_F.csv"
                                            , "Columbia (22)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_22Columbia_F.csv"
                                            , "Dade (23)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_23Dade_F.csv"
                                            , "DeSoto (24)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_24DeSoto_F.csv"
                                            , "Dixie (25)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_25Dixie_F.csv"
                                            , "Duval (26)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_26Duval_F.csv"
                                            , "Escambia (27)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_27Escambia_F.csv"
                                            , "Flagler (28)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_28Flagler_F.csv"
                                            , "Franklin (29)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_29Franklin_F.csv"
                                            , "Gadsden (30)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_30Gadsden_F.csv"
                                            , "Gilchrist (31)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_31Gilchrist_F.csv"
                                            , "Glades (32)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_32Glades_F.csv"
                                            , "Gulf (33)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_33Gulf_F.csv"
                                            , "Hamilton (34)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_34Hamilton_F.csv"
                                            , "Hardee (35)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_35Hardee_F.csv"
                                            , "Hendry (36)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_36Hendry_F.csv"
                                            , "Hernando (37)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_37Hernando_F.csv"
                                            ,"Hihglands (38)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_38Highlands_F.csv"
                                            , "Hillsborough (39)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_39Hillsborough_F.csv"
                                            , "Holmes (40)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_40Holmes_F.csv"
                                            , "Indian River (41)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_41Indian River_F.csv"
                                            , "Jackson (42)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_42Jackson_F.csv"
                                            , "Jefferson (43)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_43Jefferson_F.csv"
                                            , "Lafayette (44)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_44Lafayette_F.csv"
                                            , "Lake (45)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_45Lake_F.csv"
                                            , "Lee (46)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_46Lee_F.csv"
                                            , "Leon (47)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_47Leon_F.csv"
                                            , "Levy (48)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_48Levy_F.csv"
                                            , "Liberty (49)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_49Liberty_F.csv"
                                            , "Madison (50)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_50Madison_F.csv"
                                            , "Manatee (51)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_51Manatee_F.csv"
                                            , "Marion (52)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_52Marion_F.csv"
                                            , "Martin (53)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_53Martin_F.csv"
                                            , "Monroe (54)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_54Monroe_F.csv"
                                            , "Nassau (55)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_55Nassau_F.csv"
                                            , "Okaloosa (56)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_56Okaloosa_F.csv"
                                            , "Okeechobee (57)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_57Okeechobee_F.csv"
                                            , "Orange (58)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_58Orange_F.csv"
                                            , "Osceola (59)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_59Osceola_F.csv"
                                            , "Palm Beach (60)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_60Palm Beach_F.csv"
                                            , "Pasco (61)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_61Pasco_F.csv"
                                            , "Pinellas (62)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_62Pinellas_F.csv"
                                            , "Polk (63)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_63Polk_F.csv"
                                            , "Putnam (64)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_64Putnam_F.csv"
                                            , "St. Johns (65)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_65Saint Johns_F.csv"
                                            , "St. Lucie (66)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_66Saint Lucie_F.csv"
                                            , "Santa Rosa (67)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_67Santa Rosa_F.csv"
                                            , "Sarasota (68)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_68Sarasota_F.csv"
                                            , "Seminole (69)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_69Seminole_F.csv"
                                            , "Sumter (70)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_70Sumter_F.csv"
                                            , "Suwannee (71)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_71Suwannee_F.csv"
                                            , "Taylor (72)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_72Taylor_F.csv"
                                            , "Union (73)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_73Union_F.csv"
                                            , "Volusia (74)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_74Volusia_F.csv"
                                            , "Wakulla (75)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_75Wakulla_F.csv"
                                            , "Walton (76)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_76Walton_F.csv"
                                            , "Washington (77)": "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\2023NAL\\2023F\\NAL_2023_77Washington_F.csv"}








## Streamlit Functions 

def set_chatmodel():
    print(st.session_state["Chat_Model_Selected"])

def set_User_API_key():
    ## This exists because Streamlit wants an on_change callback for the
    ## API-key text box, and there is nothing safe to DO here.
    ##
    ## 🔒 It used to print the key. Never log a secret - not to a terminal,
    ## not "just while debugging". Console history outlives the debugging.
    pass


def update_COS_Input():
    st.session_state["COS_ADJ_KEY"] =  st.session_state["DOR_COS_ADJ_DICT"][st.session_state['current_Adjust_dor'] ]
    print('UPdated COS Key to ' + str(st.session_state["COS_ADJ_KEY"] ) + ' for DOR ' + st.session_state['current_Adjust_dor'] )          

def set_nal_data():
    print(st.session_state["County_Selected"])

### Sales Ratio Function  ###

def weighted_Mean_Sale_Ratio(sl: np.ndarray, ap: np.ndarray) -> float:
    return np.mean(ap) / np.mean(sl)


def COD(sl: np.ndarray, ap: np.ndarray) -> float:
    ratio = ap / sl
    med = np.median(ratio)
    dev = np.sum(np.abs(ratio - med))
    avgdev = dev / len(ratio)
    cod = 100 * (avgdev / med)
    return cod


def PRD(sl: np.ndarray, ap: np.ndarray) -> float:
    ratio = ap / sl
    mnratio = np.mean(ratio)
    mnap = np.mean(ap)
    mnsl = np.mean(sl)
    prd = mnratio / (mnap / mnsl)
    return prd



class DOR_Summary_Report_Generator ():

    Stratum_Number = None

    Filter_For_Qualified_Sales = True

    Qual_Col = None

    Qual_Code_List = None

    Parcel_Identifier_Column = None

    Sale_Value_Column = None

    NAL_Value_Column = None

    Stratum_Data_With_Predictions = None

    Sales_Data = None

    StratumSales = None

    sortedvalues = None

    StratumPop_unique = None

    Sorted_StratumPop = None

    trimmed = None

    trimmed_p = None

    Stratum = None

    Stratum_Text = None

    StratumSales_5 = None

    StratumSales_95 = None

    StratumSales_trimmed = None

    mean_s = None

    std_s = None

    GroupLevelStatistics = None

    GroupLevelStatistics_Text = None

    n = None

    Return_Type = 'Text'

    def __init__(self
                 , Stratum_Data
                 , Sales_Data
                 #, Parel_Identifier_Col
                # , Sale_Value_Col
                 #, NAL_Value_Col
                 ,  Stratum_Number
                 , Filter_For_Qualified_Sales = True
                 , Qualification_Col = 'QUAL_CD'
                 , Qual_Code_List = [1,2]
                 , Return_Type = 'Text'
                 , Apply_Strat_Filter = None
                 , Use_Only_Latest_Sale = False
                 , n = 5):

        self.Return_Type = Return_Type

        self.Stratum_Number = Stratum_Number

        self.Filter_For_Qualified_Sales = Filter_For_Qualified_Sales

        self.Qual_Code_List = Qual_Code_List

        self.Qual_Col = Qualification_Col

        self.Parcel_Identifier_Column = 'PARCEL_ID'

        self.Sale_Value_Column = Model_Outcome[0]

        self.NAL_Value_Column = 'Model_Prediction'

        self.Stratum_Data_With_Predictions = Stratum_Data.copy()
       
        if self.Filter_For_Qualified_Sales:
            self.Sales_Data = Sales_Data[Sales_Data[self.Qual_Col].isin(self.Qual_Code_List)].copy()
        else:
            self.Sales_Data = Sales_Data.copy()
       
        self.Apply_Strat_Filter = Apply_Strat_Filter

        self.Use_Only_Latest_Sale = Use_Only_Latest_Sale

        self.n = n

    def Format_My_Numbers(self, value , type = 'Money'):
        if type == 'Money':
            return str('{:,.2f}'.format(value))
        elif type == 'int':
            return str('{:,}'.format(int(round(value,0))))
        else :
            return str('{:,}'.format(value))

    def DOR_COD(self,sl: np.ndarray, ap: np.ndarray) -> float:
        if all(i == 0 for i in ap) or all(i == 0 for i in sl):
            return 0
        ratio = ap / sl
        med = np.median(ratio)
        dev = np.sum(np.abs(ratio - med))
        avgdev = dev / len(ratio)
        cod = 100 * (avgdev / med)
        return cod


    def DOR_PRD(self,sl: np.ndarray, ap: np.ndarray) -> float:
        if all(i == 0 for i in ap) or all(i == 0 for i in sl):
            return 0
        ratio = ap / sl
        mnratio = np.mean(ratio)
        mnap = np.mean(ap)
        mnsl = np.mean(sl)
        prd = mnratio / (mnap / mnsl)
        return prd

    def DOR_weighted_Mean(self,sl: np.ndarray, ap: np.ndarray) -> float:
        if all(i == 0 for i in ap) or all(i == 0 for i in sl):
            return 0
        return np.mean(ap) / np.mean(sl)
   
    def csuti(self, values, target):
            # cumulative_sum_until_target_index
            cum_sum = np.cumsum(values)
            return np.argmax(cum_sum > target) #- 1


    def Calculate_DOR_Summary_Data(self):

       

        self.StratumSales = pd.merge(self.Sales_Data, self.Stratum_Data_With_Predictions, on = self.Parcel_Identifier_Column , how= 'inner').copy()

        self.StratumSales = self.StratumSales.sort_values(by = self.NAL_Value_Column)

        self.StratumSales['Ratio'] =  self.StratumSales[self.NAL_Value_Column]  / self.StratumSales[self.Sale_Value_Column]
        self.StratumSales['Ratio'] =  self.StratumSales['Model_Prediction']  / self.StratumSales[Model_Outcome[0]]

        self.StratumPop_unique = self.Stratum_Data_With_Predictions.drop_duplicates(subset= self.Parcel_Identifier_Column
                                                                                     , keep = 'first')
        #self.StratumPop_unique = self.Stratum_Data_With_Predictions
        # Sort the DataFrame by the 'JustUncapped' column in ascending order
        self.Sorted_StratumPop = self.StratumPop_unique.sort_values(by=self.NAL_Value_Column)

        self.Sorted_StratumPop.reset_index(drop=True
                                            , inplace=True)

        self.trimmed = self.Sorted_StratumPop.loc[0:self.csuti(self.Sorted_StratumPop[self.NAL_Value_Column], (self.Sorted_StratumPop[self.NAL_Value_Column].sum())/20)].copy()

        self.trimmed_p = self.Sorted_StratumPop[self.Sorted_StratumPop[self.NAL_Value_Column] < self.trimmed[self.NAL_Value_Column].max()]

        #self.StratumPop_trimmed = self.Sorted_StratumPop[self.Sorted_StratumPop[self.NAL_Value_Column] >= self.trimmed[self.NAL_Value_Column].max()]
       
        self.StratumSales_5 = self.StratumSales[self.StratumSales[self.NAL_Value_Column] <= self.trimmed[self.NAL_Value_Column].max()]
       
        self.StratumSales_95 = self.StratumSales[self.StratumSales[self.NAL_Value_Column] > self.trimmed[self.NAL_Value_Column].max()]
       
        self.StratumSales_trimmed = self.StratumSales_95[(self.StratumSales_95['Ratio'] >= .3) & (self.StratumSales_95['Ratio'] <= 1.7)]
       
        self.StratumSales_5_trimmed = self.StratumSales_5[(self.StratumSales_5['Ratio'] >= .3) & (self.StratumSales_5['Ratio'] <= 1.7)]

        if self.Apply_Strat_Filter == '01':
            print('Apply Filter For Stratum 01')
            ## Check Improved
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['VacImp'] == 'I'].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['VacImp'] == 'I'].copy()
            # ## Check No Change Code
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['ChangeCode'].str.strip() == ''].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['ChangeCode'].str.strip() == ''].copy()
            ## Check Not New Construction
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['ActYearBuilt'] < self.StratumSales_trimmed['NAL_RollYear'] -1 ].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['ActYearBuilt'] < self.StratumSales_5_trimmed['NAL_RollYear'] -1  ].copy()
            ## Check that Most Recent Sale is not vacant
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['RecentImpCode'] != 'V'].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['RecentImpCode'] != 'V'].copy()
       
        elif self.Apply_Strat_Filter == '04':
            print('Apply Filter For Stratum 04')
            ## Check Vacant
            self.StratumSales_trimmed = self.StratumSales_trimmed[(self.StratumSales_trimmed['VacImp'] == 'V') | ((self.StratumSales_trimmed['VacImp'] == 'I') & (self.StratumSales_trimmed['DOR_Code'] == '07'))].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[(self.StratumSales_5_trimmed['VacImp'] == 'V') | ((self.StratumSales_5_trimmed['VacImp'] == 'I') & (self.StratumSales_5_trimmed['DOR_Code'] == '07'))].copy()
            # ## Check No Change Code
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['ChangeCode'].str.strip() == ''].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['ChangeCode'].str.strip() == ''].copy()
            ## Check Not New Construction
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['ActYearBuilt'] < self.StratumSales_trimmed['NAL_RollYear'] -1 ].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['ActYearBuilt'] < self.StratumSales_5_trimmed['NAL_RollYear'] -1  ].copy()
            ## Check that Most Recent Sale is not Imporved
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['RecentImpCode'] != 'I'].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['RecentImpCode'] != 'I'].copy()
       
        elif self.Apply_Strat_Filter == '05':
            print('Apply Filter For Stratum 05')
            ## Check Vacant
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['VacImp'] == 'V'].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['VacImp'] == 'V'].copy()
            # ## Check No Change Code
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['ChangeCode'].str.strip() == ''].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['ChangeCode'].str.strip() == ''].copy()
            ## Check Not New Construction
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['ActYearBuilt'] < self.StratumSales_trimmed['NAL_RollYear'] -1 ].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['ActYearBuilt'] < self.StratumSales_5_trimmed['NAL_RollYear'] -1  ].copy()
            ## Check that Most Recent Sale is not Imporved
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['RecentImpCode'] != 'I'].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['RecentImpCode'] != 'I'].copy()
       
        elif self.Apply_Strat_Filter == '06':
            print('Apply Filter For Stratum 06')
            ## Check Improved
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['VacImp'] == 'I'].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['VacImp'] == 'I'].copy()
            # ## Check No Change Code
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['ChangeCode'].str.strip() == ''].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['ChangeCode'].str.strip() == ''].copy()
            ## Check Not New Construction
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['ActYearBuilt'] < self.StratumSales_trimmed['NAL_RollYear'] -1 ].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['ActYearBuilt'] < self.StratumSales_5_trimmed['NAL_RollYear'] -1  ].copy()
            ## Check that Most Recent Sale is not vacant
            self.StratumSales_trimmed = self.StratumSales_trimmed[self.StratumSales_trimmed['RecentImpCode'] != 'V'].copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed[self.StratumSales_5_trimmed['RecentImpCode'] != 'V'].copy()


        if self.Use_Only_Latest_Sale:
            ## Removing Older Sales
            self.StratumSales_trimmed = self.StratumSales_trimmed.sort_values( by = 'Sale_Date' , ascending = False).drop_duplicates(subset = 'FolioID', keep = 'first').copy()
            self.StratumSales_5_trimmed = self.StratumSales_5_trimmed.sort_values( by = 'Sale_Date' , ascending = False).drop_duplicates(subset = 'FolioID', keep = 'first').copy()


       
        #G5 = self.trimmed_p
       
        #G5S = Stratum1Sales_5_trimmed[Stratum1Sales_5_trimmed[self.Parcel_Identifier_Column].isin(G5[self.Parcel_Identifier_Column])]

        self.sortedvalues_Untrimmed = self.Sorted_StratumPop.sort_values(by = self.NAL_Value_Column)
       
        self.sortedvalues_Untrimmed.reset_index(drop=True, inplace=True)

        self.G0 = self.Sorted_StratumPop.sort_values(by = self.NAL_Value_Column)
       
        self.G0.reset_index(drop=True, inplace=True)

        self.sortedvalues = self.G0[self.G0[self.NAL_Value_Column] > self.trimmed[self.NAL_Value_Column].max()]
        #self.sortedvalues = self.StratumPop_unique.sort_values(by = self.NAL_Value_Column)
        self.sortedvalues.reset_index(drop=True, inplace=True)

        #self.StratumPop_trimmed = self.sortedvalues.loc[(self.csuti(self.sortedvalues[self.NAL_Value_Column], (self.sortedvalues[self.NAL_Value_Column].sum())*.05)+1):self.sortedvalues[self.NAL_Value_Column].idxmax()]

        self.StratumPop_trimmed = self.sortedvalues[self.sortedvalues[self.NAL_Value_Column] > self.trimmed[self.NAL_Value_Column].max()]
        #self.Sorted_StratumPop.loc[1:self.csuti(self.Sorted_StratumPop[self.NAL_Value_Column], (self.Sorted_StratumPop[self.NAL_Value_Column].sum())/20)]
        #G5 = self.sortedvalues_Untrimmed.loc[0:self.csuti(self.sortedvalues_Untrimmed[self.NAL_Value_Column], (self.sortedvalues_Untrimmed[self.NAL_Value_Column].sum())*.05)]
        self.G5 = self.sortedvalues_Untrimmed[self.sortedvalues_Untrimmed[self.NAL_Value_Column] <= self.trimmed[self.NAL_Value_Column].max()]

        G5S = self.StratumSales_trimmed[self.StratumSales_trimmed[self.Parcel_Identifier_Column].isin(self.G5[self.Parcel_Identifier_Column])]

        # check if the number of rows is less than n for G5 and G5S

        #if len(G5S) < self.n:
        #    #set all rows to 0
        #    self.StratumSales_trimmed = self.StratumSales_trimmed[~self.StratumSales_trimmed[self.Parcel_Identifier_Column].isin(G5S[self.Parcel_Identifier_Column])]
        #    G5S = G5S.apply(lambda x: [0]*len(G5S))
       

        if len(self.G5) < self.n:
            #set all rows to 0  
            self.G5 = self.G5.apply(lambda x: [0]*len(self.G5))

        self.G1_trimmed = self.sortedvalues.loc[0:self.csuti(self.sortedvalues[self.NAL_Value_Column], (self.sortedvalues[ self.NAL_Value_Column].sum())*.25)]

        G1 = self.sortedvalues[(self.sortedvalues[self.NAL_Value_Column] >= self.G1_trimmed[self.NAL_Value_Column].min()) & (self.sortedvalues[self.NAL_Value_Column] <= self.G1_trimmed[self.NAL_Value_Column].max())]

        G1S = self.StratumSales_trimmed[self.StratumSales_trimmed[self.Parcel_Identifier_Column].isin(G1[self.Parcel_Identifier_Column])]

        if len(G1S) < self.n:
            #set all rows to 0
            self.StratumSales_trimmed = self.StratumSales_trimmed[~self.StratumSales_trimmed[self.Parcel_Identifier_Column].isin(G1S[self.Parcel_Identifier_Column])]
            G1S = G1S.apply(lambda x: [0]*len(G1S))
       

        if len(G1) < self.n:
            #set all rows to 0  
            G1 = G1.apply(lambda x: [0]*len(G1))

        self.G2_trimmed = self.sortedvalues.loc[(self.csuti(self.sortedvalues[self.NAL_Value_Column], (self.sortedvalues[self.NAL_Value_Column].sum())*.25)+1):self.csuti(self.sortedvalues[self.NAL_Value_Column], (self.sortedvalues[self.NAL_Value_Column].sum())*.5)]

        G2 = self.sortedvalues[(self.sortedvalues[self.NAL_Value_Column] >= self.G2_trimmed[self.NAL_Value_Column].min()) & (self.sortedvalues[self.NAL_Value_Column] <= self.G2_trimmed[self.NAL_Value_Column].max())]

        G2S = self.StratumSales_trimmed[self.StratumSales_trimmed[self.Parcel_Identifier_Column].isin(G2[self.Parcel_Identifier_Column])]

        if len(G2S) < self.n:
            #set all rows to 0
            self.StratumSales_trimmed = self.StratumSales_trimmed[~self.StratumSales_trimmed[self.Parcel_Identifier_Column].isin(G2S[self.Parcel_Identifier_Column])]
            G2S = G2S.apply(lambda x: [0]*len(G2S))
       

        if len(G2) < self.n:
            #set all rows to 0  
            G2 = G2.apply(lambda x: [0]*len(G2))

        self.G3_trimmed = self.sortedvalues.loc[(self.csuti(self.sortedvalues[self.NAL_Value_Column], (self.sortedvalues[self.NAL_Value_Column].sum())*.5)+1):self.csuti(self.sortedvalues[self.NAL_Value_Column], (self.sortedvalues[self.NAL_Value_Column].sum())*.75)]

        G3 = self.sortedvalues[(self.sortedvalues[self.NAL_Value_Column] >= self.G3_trimmed[self.NAL_Value_Column].min()) & (self.sortedvalues[self.NAL_Value_Column] <= self.G3_trimmed[self.NAL_Value_Column].max())]

        G3S = self.StratumSales_trimmed[self.StratumSales_trimmed[self.Parcel_Identifier_Column].isin(G3[self.Parcel_Identifier_Column])]

        if len(G3S) < self.n:
            #set all rows to 0
            self.StratumSales_trimmed = self.StratumSales_trimmed[~self.StratumSales_trimmed[self.Parcel_Identifier_Column].isin(G3S[self.Parcel_Identifier_Column])]
            G3S = G3S.apply(lambda x: [0]*len(G3S))
       

        if len(G3) < self.n:
            #set all rows to 0  
            G3 = G3.apply(lambda x: [0]*len(G3))

        self.G4_trimmed = self.sortedvalues.loc[(self.csuti(self.sortedvalues[self.NAL_Value_Column], (self.sortedvalues[self.NAL_Value_Column].sum())*.75)+1):self.sortedvalues[self.NAL_Value_Column].idxmax()]

        G4 = self.sortedvalues[(self.sortedvalues[self.NAL_Value_Column] >= self.G4_trimmed[self.NAL_Value_Column].min()) & (self.sortedvalues[self.NAL_Value_Column] <= self.G4_trimmed[self.NAL_Value_Column].max())]

        G4S = self.StratumSales_trimmed[self.StratumSales_trimmed[self.Parcel_Identifier_Column].isin(G4[self.Parcel_Identifier_Column])]

        if len(G4S) < self.n:
            #set all rows to 0
            self.StratumSales_trimmed = self.StratumSales_trimmed[~self.StratumSales_trimmed[self.Parcel_Identifier_Column].isin(G4S[self.Parcel_Identifier_Column])]
            G4S = G4S.apply(lambda x: [0]*len(G4S))
   

        if len(G4) < self.n:
            #set all rows to 0  
            G4 = G4.apply(lambda x: [0]*len(G4))

        # # Create an empty DataFrame

        self.Stratum = pd.DataFrame(columns=['STR', 'GRP', 'LOW', 'HIGH','#SAMP','COV','P.A SAMP VAL','TOT PAR','P.A JUST VAL'])
        self.Stratum_Text = pd.DataFrame(columns=['STR', 'GRP', 'LOW', 'HIGH','#SAMP','COV','P.A SAMP VAL','TOT PAR','P.A JUST VAL'])

        # # Fill out each row manually
        self.Stratum.loc[0] = [self.Stratum_Number + ' S'
                               , 1
                               , self.G1_trimmed[self.NAL_Value_Column].min()
                               , self.G1_trimmed[self.NAL_Value_Column].max()
                               , G1S.shape[0]
                               , ""
                               , G1S[self.NAL_Value_Column].sum()
                               , G1.shape[0]
                               , G1[self.NAL_Value_Column].sum()]
        self.Stratum.loc[1] = [self.Stratum_Number + ' S'
                               , 2
                               , self.G2_trimmed[self.NAL_Value_Column].min()
                               , self.G2_trimmed[self.NAL_Value_Column].max()
                               , G2S.shape[0]
                               , ""
                               , G2S[self.NAL_Value_Column].sum()
                               , G2.shape[0]
                               , G2[self.NAL_Value_Column].sum()]
        self.Stratum.loc[2] = [self.Stratum_Number + ' S'
                               , 3
                               , self.G3_trimmed[self.NAL_Value_Column].min()
                               , self.G3_trimmed[self.NAL_Value_Column].max()
                               , G3S.shape[0]
                               , ""
                               , G3S[self.NAL_Value_Column].sum()
                               , G3.shape[0]
                               , G3[self.NAL_Value_Column].sum()]
        self.Stratum.loc[3] = [self.Stratum_Number + ' S'
                               , 4
                               , self.G4_trimmed[self.NAL_Value_Column].min()
                               , self.G4_trimmed[self.NAL_Value_Column].max()
                               , G4S.shape[0]
                               , ""
                               , G4S[self.NAL_Value_Column].sum()
                               , G4.shape[0]
                               , G4[self.NAL_Value_Column].sum()]
        self.Stratum.loc[4] = [self.Stratum_Number + ' S'
                               , 5
                               , self.trimmed[self.NAL_Value_Column].min()
                               , self.trimmed[self.NAL_Value_Column].max()
                               , G5S.shape[0]
                               , ""
                               , G5S[self.NAL_Value_Column].sum()
                               , self.G5.shape[0]
                               , self.G5[self.NAL_Value_Column].sum()]

        self.Stratum_Text.loc[0] = [self.Stratum_Number + ' S'
                                    , 1
                                    , self.Format_My_Numbers(self.G1_trimmed[self.NAL_Value_Column].min() , type = 'int')
                                    , self.Format_My_Numbers(self.G1_trimmed[self.NAL_Value_Column].max(), type = 'int')
                                    , self.Format_My_Numbers(G1S.shape[0], type = 'int')
                                    , ""
                                    , self.Format_My_Numbers(G1S[self.NAL_Value_Column].sum(), type = 'int')
                                    , self.Format_My_Numbers(G1.shape[0], type = 'int')
                                    , self.Format_My_Numbers(G1[self.NAL_Value_Column].sum(), type = 'int')]
        self.Stratum_Text.loc[1] = [self.Stratum_Number + ' S'
                                    , 2
                                    , self.Format_My_Numbers(self.G2_trimmed[self.NAL_Value_Column].min(), type = 'int')
                                    , self.Format_My_Numbers(self.G2_trimmed[self.NAL_Value_Column].max(), type = 'int')
                                    , self.Format_My_Numbers(G2S.shape[0], type = 'int')
                                    , ""
                                    , self.Format_My_Numbers(G2S[self.NAL_Value_Column].sum(), type = 'int')
                                    , self.Format_My_Numbers(G2.shape[0], type = 'int')
                                    , self.Format_My_Numbers(G2[self.NAL_Value_Column].sum(), type = 'int')]
        self.Stratum_Text.loc[2] = [self.Stratum_Number + ' S'
                                    , 3
                                    , self.Format_My_Numbers(self.G3_trimmed[self.NAL_Value_Column].min(), type = 'int')
                                    , self.Format_My_Numbers(self.G3_trimmed[self.NAL_Value_Column].max(), type = 'int')
                                    , self.Format_My_Numbers(G3S.shape[0], type = 'int')
                                    , ""
                                    , self.Format_My_Numbers(G3S[self.NAL_Value_Column].sum(), type = 'int')
                                    , self.Format_My_Numbers(G3.shape[0], type = 'int')
                                    , self.Format_My_Numbers(G3[self.NAL_Value_Column].sum(), type = 'int')]
        self.Stratum_Text.loc[3] = [self.Stratum_Number + ' S', 4
                                    , self.Format_My_Numbers(self.G4_trimmed[self.NAL_Value_Column].min(), type = 'int')
                                    , self.Format_My_Numbers(self.G4_trimmed[self.NAL_Value_Column].max(), type = 'int')
                                    , self.Format_My_Numbers(G4S.shape[0], type = 'int')
                                    , ""
                                    , self.Format_My_Numbers(G4S[self.NAL_Value_Column].sum(), type = 'int')
                                    , self.Format_My_Numbers(G4.shape[0], type = 'int')
                                    , self.Format_My_Numbers(G4[self.NAL_Value_Column].sum(), type = 'int')]
        self.Stratum_Text.loc[4] = [self.Stratum_Number + ' S'
                                    , 5
                                    , self.Format_My_Numbers(self.trimmed[self.NAL_Value_Column].min(), type = 'int')
                                    , self.Format_My_Numbers(self.trimmed[self.NAL_Value_Column].max(), type = 'int')
                                    , self.Format_My_Numbers(G5S.shape[0], type = 'int')
                                    , ""
                                    , self.Format_My_Numbers(G5S[self.NAL_Value_Column].sum(), type = 'int')
                                    , self.Format_My_Numbers(self.G5.shape[0], type = 'int')
                                    , self.Format_My_Numbers(self.G5[self.NAL_Value_Column].sum(), type = 'int')]


        self.mean_s = self.StratumSales_trimmed['Ratio'].mean()

        self.std_s = self.StratumSales_trimmed['Ratio'].std()

        self.confidence_interval = stats.t.interval(0.95
                                                    , len(self.StratumSales_trimmed['Ratio'])-1
                                                    , loc = self.mean_s
                                                    , scale = self.std_s/np.sqrt(len(self.StratumSales_trimmed['Ratio'])))
       
       
        self.Stratum.loc[5] = [""
                                ,""
                                ,"Stratum"
                                ,"Total:"
                                ,self.StratumSales_trimmed.shape[0]
                                ,(self.std_s/self.mean_s)*100
                                ,self.StratumSales_trimmed[self.NAL_Value_Column].sum()
                                ,self.StratumPop_trimmed.shape[0]
                                ,self.StratumPop_trimmed[self.NAL_Value_Column].sum()]
       
        self.Stratum_Text.loc[5] = [""
                                ,""
                                ,"Stratum"
                                ,"Total:"
                                , self.Format_My_Numbers(self.StratumSales_trimmed.shape[0] , type = 'int')
                                , str(round((self.std_s/self.mean_s)*100,2) )
                                , self.Format_My_Numbers(self.StratumSales_trimmed[self.NAL_Value_Column].sum(), type = 'int' )
                                , self.Format_My_Numbers(self.StratumPop_trimmed.shape[0], type = 'int' )
                                , self.Format_My_Numbers(self.StratumPop_trimmed[self.NAL_Value_Column].sum(), type = 'int')]



        self.Stratum.loc[6] = [""
                               , "COD:"
                               , round(self.DOR_COD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]),2)
                               ,"PRD:"
                               , round(self.DOR_PRD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column])*100,2)
                               , "95 %"
                               , "Conf Intvl"
                               , round(self.confidence_interval[0],2)
                               , round(self.confidence_interval[1],2)]
       

        self.Stratum_Text.loc[6] = [""
                               , "COD:"
                               , str(round(self.DOR_COD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]),2))
                               , "PRD:"
                               , str(round(self.DOR_PRD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column])*100,2))
                               , "95 %"
                               , "Conf Intvl"
                               , str(round(self.confidence_interval[0],2))
                               , str(round(self.confidence_interval[1],2))]
       

        self.GroupLevelStatistics = pd.DataFrame(columns=['Stratum', 'Group', 'N', 'Median','Mean','COD','PRD','Weighted Mean'])
        self.GroupLevelStatistics_Text = pd.DataFrame(columns=['Stratum', 'Group', 'N', 'Median','Mean','COD','PRD','Weighted Mean'])

        Group_1_Mean_Sale_Ratio = 0
        Group_1_Median_Sale_Ratio = 0
        if G1S[self.NAL_Value_Column].size > self.n :
            Group_1_Mean_Sale_Ratio = round(((G1S[self.NAL_Value_Column]/G1S[self.Sale_Value_Column]).mean())*100,2)
            Group_1_Median_Sale_Ratio = round(((G1S[self.NAL_Value_Column]/G1S[self.Sale_Value_Column]).median())*100,2)

        Group_2_Mean_Sale_Ratio = 0
        Group_2_Median_Sale_Ratio = 0
        if G2S[self.NAL_Value_Column].size > self.n :
            Group_2_Mean_Sale_Ratio = round(((G2S[self.NAL_Value_Column]/G2S[self.Sale_Value_Column]).mean())*100,2)
            Group_2_Median_Sale_Ratio = round(((G2S[self.NAL_Value_Column]/G2S[self.Sale_Value_Column]).median())*100,2)

        Group_3_Mean_Sale_Ratio = 0
        Group_3_Median_Sale_Ratio = 0
        if G3S[self.NAL_Value_Column].size > self.n :
            Group_3_Mean_Sale_Ratio = round(((G3S[self.NAL_Value_Column]/G3S[self.Sale_Value_Column]).mean())*100,2)
            Group_3_Median_Sale_Ratio = round(((G3S[self.NAL_Value_Column]/G3S[self.Sale_Value_Column]).median())*100,2)

        Group_4_Mean_Sale_Ratio = 0
        Group_4_Median_Sale_Ratio = 0
        if G4S[self.NAL_Value_Column].size > self.n :
            Group_4_Mean_Sale_Ratio = round(((G4S[self.NAL_Value_Column]/G4S[self.Sale_Value_Column]).mean())*100,2)
            Group_4_Median_Sale_Ratio = round(((G4S[self.NAL_Value_Column]/G4S[self.Sale_Value_Column]).median())*100,2)

        ## Fill out each row manually
        self.GroupLevelStatistics.loc[0] = [self.Stratum_Number
                                        , 1
                                        , G1S.shape[0]
                                        , Group_1_Median_Sale_Ratio
                                        , Group_1_Mean_Sale_Ratio
                                        , round(self.DOR_COD(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column]),2)
                                        , round(self.DOR_PRD(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column])*100,2)
                                        , round(self.DOR_weighted_Mean(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column])*100,2)]
        self.GroupLevelStatistics.loc[1] = [self.Stratum_Number
                                        , 2
                                        , G2S.shape[0]
                                        , Group_2_Median_Sale_Ratio
                                        , Group_2_Mean_Sale_Ratio
                                        , round(self.DOR_COD(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column]),2)
                                        , round(self.DOR_PRD(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column])*100,2)
                                        , round(self.DOR_weighted_Mean(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column])*100,2)]
        self.GroupLevelStatistics.loc[2] = [self.Stratum_Number
                                        , 3
                                        , G3S.shape[0]
                                        , Group_3_Median_Sale_Ratio
                                        , Group_3_Mean_Sale_Ratio
                                        , round(self.DOR_COD(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column]),2)
                                        , round(self.DOR_PRD(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column])*100,2)
                                        , round(self.DOR_weighted_Mean(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column])*100,2)]
        self.GroupLevelStatistics.loc[3] = [self.Stratum_Number
                                        , 4
                                        , G4S.shape[0]
                                        , Group_4_Median_Sale_Ratio
                                        , Group_4_Mean_Sale_Ratio
                                        , round(self.DOR_COD(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column]),2)
                                        , round(self.DOR_PRD(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column])*100,2)
                                        , round(self.DOR_weighted_Mean(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column])*100,2)]
       
        self.GroupLevelStatistics.loc[4] = [self.Stratum_Number
                                        , 'Total'
                                        , self.StratumSales_trimmed.shape[0]
                                        , round(((self.StratumSales_trimmed[self.NAL_Value_Column]/self.StratumSales_trimmed[self.Sale_Value_Column]).median())*100,2)
                                        , round(((self.StratumSales_trimmed[self.NAL_Value_Column]/self.StratumSales_trimmed[self.Sale_Value_Column]).mean())*100,2)
                                        , round(self.DOR_COD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]),2)
                                        , round(self.DOR_PRD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column])*100,2)
                                        , round(self.DOR_weighted_Mean(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column])*100,2)]
       

        ## Fill out each row manually
        self.GroupLevelStatistics_Text.loc[0] = [self.Stratum_Number
                                                    , str(1)
                                                    , self.Format_My_Numbers(G1S.shape[0] , type = 'int')
                                                    , str(Group_1_Median_Sale_Ratio)
                                                    , str(Group_1_Mean_Sale_Ratio)
                                                    , str(round(self.DOR_COD(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column]),2))
                                                    , str(round(self.DOR_PRD(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column])*100,2))
                                                    , str(round(self.DOR_weighted_Mean(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column])*100,2))]
        self.GroupLevelStatistics_Text.loc[1] = [self.Stratum_Number
                                                    , str(2)
                                                    , self.Format_My_Numbers(G2S.shape[0], type = 'int')
                                                    , str(Group_2_Median_Sale_Ratio)
                                                    , str(Group_2_Mean_Sale_Ratio)
                                                    , str(round(self.DOR_COD(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column]),2))
                                                    , str(round(self.DOR_PRD(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column])*100,2))
                                                    , str(round(self.DOR_weighted_Mean(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column])*100,2))]
        self.GroupLevelStatistics_Text.loc[2] = [self.Stratum_Number
                                                    , 3
                                                    , self.Format_My_Numbers(G3S.shape[0], type = 'int')
                                                    , str(Group_3_Median_Sale_Ratio)
                                                    , str(Group_3_Mean_Sale_Ratio)
                                                    , str(round(self.DOR_COD(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column]),2))
                                                    , str(round(self.DOR_PRD(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column])*100,2))
                                                    , str(round(self.DOR_weighted_Mean(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column])*100,2))]
        self.GroupLevelStatistics_Text.loc[3] = [self.Stratum_Number
                                                    , 4
                                                    , self.Format_My_Numbers(G4S.shape[0], type = 'int')
                                                    , str(Group_4_Median_Sale_Ratio)
                                                    , str(Group_4_Mean_Sale_Ratio)
                                                    , str(round(self.DOR_COD(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column]),2))
                                                    , str(round(self.DOR_PRD(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column])*100,2))
                                                    , str(round(self.DOR_weighted_Mean(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column])*100,2))]
        self.GroupLevelStatistics_Text.loc[4] = [self.Stratum_Number
                                                    , 'Total'
                                                    , self.Format_My_Numbers(self.StratumSales_trimmed.shape[0], type = 'int')
                                                    , str(round(((self.StratumSales_trimmed[self.NAL_Value_Column]/self.StratumSales_trimmed[self.Sale_Value_Column]).median())*100,2))
                                                    , str(round(((self.StratumSales_trimmed[self.NAL_Value_Column]/self.StratumSales_trimmed[self.Sale_Value_Column]).mean())*100,2))
                                                    , str(round(self.DOR_COD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]),2))
                                                    , str(round(self.DOR_PRD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column])*100,2))
                                                    , str(round(self.DOR_weighted_Mean(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column])*100,2))]

        if self.Return_Type.lower() == 'text':
            return self.Stratum_Text , self.GroupLevelStatistics_Text
        else:
            return self.Stratum , self.GroupLevelStatistics


### GUI Functions ###
def render_animation():
    #animation_response = requests.get('https://assets1.lottiefiles.com/packages/lf20_vykpwvdsvsdvsdt8b.json')
    animation_response = requests.get('https://assets1.lottiefiles.com/packages/lf20_vykpwt8b.json')
    animation_json = dict()
    
    if animation_response.status_code == 200:
        animation_json = animation_response.json()
    else:
        print("Error in the URL")  
        1/0   
                           
    return st_lottie(animation_json,height=300,width=400)

def render_animation2():
    animation_response = requests.get('https://lottie.host/864a3252-e826-4a65-ab0d-7fe545b55100/GKjQEfQguG.json')
    animation_json = dict()
    
    if animation_response.status_code == 200:
        animation_json = animation_response.json()
    else:
        print("Error in the URL")   
        1/0  
                           
    return st_lottie(animation_json,height=300,width=400)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index = False).encode("utf-8")


def Password_Entry(u,p):
    global User_Password_List
    p_entry = [u,p]

    if p_entry in User_Password_List:
        st.session_state['User_Name_Entry'] = u
        st.session_state['PSWRD_Entry'] = p

    st.session_state['Password_Count'] = st.session_state['Password_Count'] + 1




### Data Manipulation Functions ###

def Merge_data(NAL_Data , SDF_Data , GIS_Data):
    using_last_NAL = False
  
    if NAL_Data is not None and SDF_Data is not None:
        SDF = SDF_Data.copy()
        NAL = NAL_Data.copy() 
    else : 
        SDF = SDF_Data.copy()
        NAL = pd.read_csv( st.session_state['County_List_Loc'][st.session_state['County_Selected']]
                            , low_memory = False
                            )
        st.session_state["NAL"] = NAL.copy() ### CHECK THIS CODE !!!!!! ### 
        using_last_NAL = True

    if len(SDF[~SDF['PARCEL_ID'].isin(NAL['PARCEL_ID'])]) > 0:
        SDF = SDF[SDF['PARCEL_ID'].isin(NAL['PARCEL_ID'])].copy()
        st.session_state['SDF'] = SDF.copy()

    ## Establishing Encoders if needed 
    if NAL['NBRHD_CD'].dtype == 'object': 
        #NAL_merged['NBRHD_CD'] = NAL_merged['NBRHD_CD'].astype('str')
        st.session_state['NBHD_Encoder'] = OrdinalEncoder( dtype= int 
                                      , encoded_missing_value = -1
                                        , handle_unknown = 'use_encoded_value'
                                        , unknown_value = -1
                                        )
        st.session_state['NBHD_Encoder'].fit(NAL[['NBRHD_CD']])

    ## Need a look up to get adjusted Sale price for SDF 
    NAL_SDF_Fix = NAL[['PARCEL_ID','DOR_UC']].copy()    

    ## Feature Engineering 
    SDF['Time_to_Sale'] = 12*(SDF['SALE_YR'] - min(SDF['SALE_YR'])) + SDF['SALE_MO']


    NAL['Age_Diff'] =  NAL['EFF_YR_BLT'] - NAL['ACT_YR_BLT'] 
    NAL['Act_Age'] = NAL['ASMNT_YR'] - NAL['ACT_YR_BLT']
    NAL['Eff_Age'] =  NAL['ASMNT_YR'] - NAL['EFF_YR_BLT']

    NAL['Age_Diff'] = NAL['Age_Diff'].astype('Int64')
    NAL['Act_Age'] = NAL['Act_Age'].astype('Int64')
    NAL['Eff_Age'] = NAL['Eff_Age'].astype('Int64')

    ## Merge NAL and SDF
    NAL_merged = pd.merge(NAL[['PARCEL_ID'] + NAL_Features]
                    , SDF[['PARCEL_ID'] + Sale_Features] 
                    , on='PARCEL_ID'
                    , how='inner')
    
    if GIS_Data is not None:

        GIS_Data['Northing'] = GIS_Data['geometry'].centroid.x
        GIS_Data['Easting'] = GIS_Data['geometry'].centroid.y


        GIS_Data_Clean = GIS_Data[['PARCEL_ID','Northing','Easting']].copy()

        if st.session_state['State_Coords_Selected'] is None :
            fl_state_plane_epsg = "EPSG:2237"
        else:
            fl_state_plane_epsg = st.session_state['State_Coords_Selected'] 
        # Define the target geographic coordinate system (WGS84 is common)
        wgs84_epsg = "EPSG:4326"  # WGS84
        # Create Transformer objects for the conversion
        transformer = Transformer.from_crs(fl_state_plane_epsg, wgs84_epsg)

        GIS_Data_Clean['Latitude']  , GIS_Data_Clean['Longitude'] = transformer.transform( GIS_Data_Clean['Northing'],GIS_Data_Clean['Easting'] )

        NAL_merged = pd.merge(NAL_merged
                                , GIS_Data_Clean[['PARCEL_ID','Latitude', 'Longitude']]
                                , on = 'PARCEL_ID'
                                , how = 'inner').copy()

        st.session_state['Model_Features'] = st.session_state['Model_Features'] + ['Latitude', 'Longitude']

        st.session_state['Model_Features'].remove('NBRHD_CD') 

        st.session_state['Model_Features_Condos'] = st.session_state['Model_Features_Condos'] + ['Latitude', 'Longitude']

        st.session_state['Model_Features_Condos'].remove('NBRHD_CD') 

        st.session_state['Model_Features_Vac_Res'] = st.session_state['Model_Features_Vac_Res'] + ['Latitude', 'Longitude']

        st.session_state['Model_Features_Vac_Res'].remove('NBRHD_CD') 

        st.session_state['Model_Features_Commercial'] = st.session_state['Model_Features_Commercial'] + ['Latitude', 'Longitude']

        st.session_state['Model_Features_Commercial'].remove('NBRHD_CD') 

        All_NAL_Data =st.session_state["NAL"].copy()

        All_NAL_Data_With_GIS = pd.merge(All_NAL_Data
                                        , GIS_Data_Clean[['PARCEL_ID','Latitude', 'Longitude']]
                                        , on = 'PARCEL_ID'
                                        , how = 'inner').copy()

        st.session_state["NAL"] = All_NAL_Data_With_GIS.copy()
    

    ## Trying to quickly clean if using last year NAL 
    if using_last_NAL == True:
        NAL_merged = NAL_merged[ ~( (NAL_merged['DOR_UC'].isin([0,10,40])) & (NAL_merged['VI_CD'].isin(['I']) ) )  ]
        NAL_merged = NAL_merged[ ~( ((~NAL_merged['DOR_UC'].isin([0,10,40]))) & (NAL_merged['VI_CD'].isin(['V'])) )  ]


    ## Fixing Data types 
    if NAL['NBRHD_CD'].dtype == 'object': 
        NAL_merged['NBRHD_CD'] = st.session_state['NBHD_Encoder'].transform(NAL_merged[['NBRHD_CD']])
    else:
        NAL_merged['NBRHD_CD'] = NAL_merged['NBRHD_CD'].astype('Int64').astype('str')

    NAL_merged['CENSUS_BK'] = NAL_merged['CENSUS_BK'].astype('str')
    NAL_merged['DOR_UC'] = NAL_merged['DOR_UC'].apply(lambda x : str(x).zfill(2))
    NAL_SDF_Fix['DOR_UC'] = NAL_SDF_Fix['DOR_UC'].apply(lambda x : str(x).zfill(2))

    ## Creating Adjusted Sale Amount for the Model 

    #NAL_merged['Adjusted_Sale_Price'] = round(NAL_merged['SALE_PRC'] * .85,0).astype('Int64')
    NAL_merged['Adjusted_Sale_Price']  = NAL_merged.apply(lambda row : round(row['SALE_PRC'] * st.session_state["DOR_COS_ADJ_DICT"][str(row['DOR_UC'])],0) , axis = 1)

    ## Adding the adjusted sale price to the SDF for use when creating summary stats
    NAL_SDF_Fix['Sale_Adjustment_factor'] = .85
    NAL_SDF_Fix['Sale_Adjustment_factor'] = NAL_SDF_Fix.apply(lambda row : st.session_state["DOR_COS_ADJ_DICT"][str(row['DOR_UC'])] , axis = 1)

    SDF_Fix_Dict = dict(zip(NAL_SDF_Fix['PARCEL_ID']  , NAL_SDF_Fix['Sale_Adjustment_factor']))

    st.session_state["SDF"]['Adjusted_Sale_Price']= st.session_state["SDF"].apply(lambda row : round(row['SALE_PRC'] * SDF_Fix_Dict[row['PARCEL_ID']],0) , axis = 1)

    
    return NAL_merged
        
    

def Model_Objective(trial, Model_Train_Data, Model_Test_Data,model_features, performance_metric = 'RMSE', model_early_stopping_Rounds = 150 ):


  
    Opt_model_params = {
                    'objective': 'regression'
                    ,'metric': 'rmse'
                    ,'boosting_type': 'gbdt'
                    ,'n_estimators': trial.suggest_int('n_estimators', low = st.session_state['HyperParametet_Tuning_N_Estimators_Low'], high = st.session_state['HyperParametet_Tuning_N_Estimators_High'])
                    ,'num_leaves': trial.suggest_int('num_leaves', low = st.session_state['HyperParametet_Tuning_num_leaves_Low'], high = st.session_state['HyperParametet_Tuning_num_leaves_High'])
                    ,'learning_rate': trial.suggest_float('learning_rate', low = st.session_state['HyperParametet_Tuning_learning_rate_Low'], high = st.session_state['HyperParametet_Tuning_learning_rate_High'])
              #      ,'subsample': trial.suggest_float('subsample', low = st.session_state['HyperParametet_Tuning_subsample_Low'], high = st.session_state['HyperParametet_Tuning_subsample_High'])
              #      ,'feature_fraction': trial.suggest_float('feature_fraction', low = st.session_state['HyperParametet_Tuning_feature_fraction_Low'], high = st.session_state['HyperParametet_Tuning_feature_fraction_High'] )
                    ,'max_depth': trial.suggest_int('max_depth', low = st.session_state['HyperParametet_Tuning_max_depth_Low'], high = st.session_state['HyperParametet_Tuning_max_depth_High'])
                    ,'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', low = st.session_state['HyperParametet_Tuning_min_data_in_leaf_Low'] , high = st.session_state['HyperParametet_Tuning_min_data_in_leaf_High'])
                    ,'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', low = st.session_state['HyperParametet_Tuning_min_sum_hessian_in_leaf_Low'], high = st.session_state['HyperParametet_Tuning_min_sum_hessian_in_leaf_High'])
                    ,'random_state': 42
                    ,'verbose': -1
                }



    lgb_model_opt = ltb.LGBMRegressor(**Opt_model_params)
    lgb_model_opt.fit(Model_Train_Data[model_features] 
                        , Model_Train_Data[Model_Outcome[0]]
                       # , eval_set=[(Model_Test_Data[model_features], Model_Test_Data[Model_Outcome[0]])]
                       # , eval_names=['train', 'test']
                       # , callbacks=[ltb.early_stopping(stopping_rounds = model_early_stopping_Rounds, verbose=False)]
                       )
    
    Opt_Model_Pred = lgb_model_opt.predict(Model_Test_Data[model_features])


        
    opt_metric_score = 0 
    if performance_metric.lower().strip() == 'rmse':
        opt_metric_score = np.sqrt(mean_squared_error(Model_Test_Data[Model_Outcome[0]], Opt_Model_Pred))
    elif performance_metric.lower().strip() == 'r2':
        opt_metric_score = r2_score(Model_Test_Data[Model_Outcome[0]], Opt_Model_Pred)
    elif performance_metric.lower().strip() == 'mae':
        opt_metric_score = mean_absolute_error(Model_Test_Data[Model_Outcome[0]], Opt_Model_Pred)
    elif performance_metric.lower().strip() == 'cod':
        opt_metric_score = COD(Model_Test_Data[Model_Outcome[0]], Opt_Model_Pred)

    return opt_metric_score


### ML Model Functions ###
def Build_ML_Regressors (pg_bar):

    percent_complete = 0 

    number_of_models = 0 

    pg_bar.progress(1, text='**Please hang on tight we are building something cool together!**')  
    ### Gathering Qualified Sales from Merged Data Frame ###

    Pop_NAL_Data = st.session_state["NAL"].copy()

    NAL_merged = st.session_state["NAL_merged"].copy()
    
    NAL_Model_Data = NAL_merged[(NAL_merged['QUAL_CD'].isin([1,2]))  ].copy()

    if 'NBRHD_CD' in NAL_Model_Data.columns:
        NAL_Model_Data[ 'NBRHD_CD'] = NAL_Model_Data['NBRHD_CD'].astype('Int64')

    NAL_Model_Data[ 'DOR_UC'] = NAL_Model_Data['DOR_UC'].astype('Int64')

    if st.session_state["SFR_model_binary"] == True:
        NAL_Model_Data_SFR = NAL_Model_Data[NAL_Model_Data['DOR_UC'].isin([1])]
        NAL_Pop_Data_SFR = Pop_NAL_Data[Pop_NAL_Data['DOR_UC'].isin([1])]

        if st.session_state['NBHD_Encoder'] is not None:
            NAL_Pop_Data_SFR['NBRHD_CD'] = st.session_state['NBHD_Encoder'].transform(NAL_Pop_Data_SFR[['NBRHD_CD']])

        number_of_models = number_of_models + 1

    if st.session_state["Vac_Res_model_binary"] == True:
        NAL_ModelData_Vac = NAL_Model_Data[NAL_Model_Data['DOR_UC'].isin([0])]
        NAL_Pop_Data_Vac = Pop_NAL_Data[Pop_NAL_Data['DOR_UC'].isin([0])]
        number_of_models = number_of_models + 1

    if st.session_state["Condo_model_binary"] == True:
        NAL_Model_Data_Condo = NAL_Model_Data[NAL_Model_Data['DOR_UC'].isin([4 ,5])]
        NAL_Pop_Data_Condo = Pop_NAL_Data[Pop_NAL_Data['DOR_UC'].isin([4 ,5])]
        number_of_models = number_of_models + 1

    if st.session_state["Commercial_model_binary"] == True:
        NAL_Model_Data_Com = NAL_Model_Data[NAL_Model_Data['DOR_UC'].isin([11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32
                                                                           ,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49])]
        NAL_Pop_Data_Com = Pop_NAL_Data[Pop_NAL_Data['DOR_UC'].isin([11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33
                                                                     ,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49])]
        number_of_models = number_of_models + 1



    current_Loading_progress = int(10) 

    pg_bar.progress(current_Loading_progress, text='**Please hang on tight we are building something cool together!**')  

    Model_percent_complete = round((100 - 10) /(number_of_models),0)

    

    if st.session_state["SFR_model_binary"] == True:
        NAL_Data_Train_SFR, NAL_Data_Test_SFR = train_test_split(NAL_Model_Data_SFR
                                                                    , test_size = 0.2
                                                                    , random_state=42
                                                                    , stratify = NAL_Model_Data_SFR['SALE_YR'])
        
        NAL_Data_Train_SFR_Features = NAL_Data_Train_SFR[st.session_state['Model_Features']]

        NAL_Data_Test_SFR_Features = NAL_Data_Test_SFR[st.session_state['Model_Features']]

        if st.session_state['Hyperparameter_Tuning_Toggle'] == False : 
            NAL_Model = ltb.LGBMRegressor(n_estimators = 500
                                        , min_child_samples = 3
                                        , max_depth = 10
                                        , random_state = 42)
            

            NAL_Model.fit(NAL_Data_Train_SFR_Features
                        , NAL_Data_Train_SFR[Model_Outcome])
            
        else: 
            print('---- PERFORMING SFR HYPERPARAMETER TUNING ----')
            LGBM_study = optuna.create_study(direction='minimize')
            LGBM_study_obj_func = lambda trial: Model_Objective(trial
                                                    , Model_Train_Data = NAL_Data_Train_SFR
                                                    , Model_Test_Data = NAL_Data_Test_SFR
                                                    , model_features = st.session_state['Model_Features']
                                                    , performance_metric = 'COD'
                                                    , model_early_stopping_Rounds = 150)    
            
            LGBM_study.optimize(LGBM_study_obj_func
                                , n_trials = st.session_state['Hyperparameter_Tuning_Trials'])
            
            LGBM_Best_Params = LGBM_study.best_params.copy()
            LGBM_Best_Params = {**LGBM_Best_Params, **{'random_state': 42}}
            #LGBM_Best_Params = {**LGBM_Best_Params, **{'n_estimators': 1000}}
            LGBM_Best_Params = {**LGBM_Best_Params, **{'verbose': -1}}

            NAL_Model = ltb.LGBMRegressor(**LGBM_Best_Params)

            NAL_Model.fit(NAL_Data_Train_SFR_Features
                            , NAL_Data_Train_SFR[Model_Outcome[0]]
                             ,eval_set=[(NAL_Data_Test_SFR[st.session_state['Model_Features']], NAL_Data_Test_SFR[Model_Outcome[0]])]
                            , eval_names=['train', 'test']
                            , callbacks=[ltb.early_stopping(stopping_rounds = 150, verbose=False)])
            

        NAL_Model_Train_SFR_results = NAL_Data_Train_SFR.copy()

        NAL_Model_Train_SFR_results['Model_Prediction'] = NAL_Model.predict(NAL_Data_Train_SFR_Features)

        NAL_Model_Train_SFR_results['sale_ratio'] = NAL_Model_Train_SFR_results['Model_Prediction'] / NAL_Model_Train_SFR_results[Model_Outcome[0]] 

        NAL_Model_Train_SFR_results_Trimmed = NAL_Model_Train_SFR_results[(NAL_Model_Train_SFR_results['sale_ratio'] > .3) & (NAL_Model_Train_SFR_results['sale_ratio'] < 1.7)]

        #print('Number of Records Before Trimming: ', NAL_Model_Train_SFR_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Train_SFR_results_Trimmed.shape[0])

        NAL_Model_Test_SFR_results = NAL_Data_Test_SFR.copy()

        NAL_Model_Test_SFR_results['Model_Prediction'] = NAL_Model.predict(NAL_Data_Test_SFR_Features)

        NAL_Model_Test_SFR_results['sale_ratio'] = NAL_Model_Test_SFR_results['Model_Prediction'] / NAL_Model_Test_SFR_results[Model_Outcome[0]] 

        NAL_Model_Test_SFR_results['sale_ratio_JV'] = (NAL_Model_Test_SFR_results['JV'] / 0.85) / NAL_Model_Test_SFR_results[Model_Outcome[0]] 

        NAL_Model_Test_results_Trimmed = NAL_Model_Test_SFR_results[(NAL_Model_Test_SFR_results['sale_ratio'] > .3) & (NAL_Model_Test_SFR_results['sale_ratio'] < 1.7)]

        NAL_Model_Test_results_Trimmed_JV = NAL_Model_Test_SFR_results[(NAL_Model_Test_SFR_results['sale_ratio_JV'] > .3) & (NAL_Model_Test_SFR_results['sale_ratio_JV'] < 1.7)]

        #print(NAL_Model_Test_results_Trimmed.info())

        #print('Number of Records Before Trimming: ', NAL_Model_Test_SFR_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Test_results_Trimmed.shape[0])

        NAL_Model_Data_SFR_For_Preds = st.session_state["NAL"].copy()

        if st.session_state['NBHD_Encoder'] != None:
            NAL_Model_Data_SFR_For_Preds['NBRHD_CD'] = st.session_state['NBHD_Encoder'].transform(NAL_Model_Data_SFR_For_Preds[['NBRHD_CD']])

        NAL_Model_Data_SFR_For_Preds = NAL_Model_Data_SFR_For_Preds[NAL_Model_Data_SFR_For_Preds['DOR_UC'].isin([1])]

        NAL_Model_Data_SFR_For_Preds['Time_to_Sale'] = 12*(st.session_state['Valuation_Year'] - min(st.session_state["SDF"]['SALE_YR'])) + 0

        NAL_Model_Data_SFR_For_Preds['Age_Diff'] =  NAL_Model_Data_SFR_For_Preds['EFF_YR_BLT'] - NAL_Model_Data_SFR_For_Preds['ACT_YR_BLT'] 
        NAL_Model_Data_SFR_For_Preds['Act_Age'] = NAL_Model_Data_SFR_For_Preds['ASMNT_YR'] - NAL_Model_Data_SFR_For_Preds['ACT_YR_BLT']
        NAL_Model_Data_SFR_For_Preds['Eff_Age'] =  NAL_Model_Data_SFR_For_Preds['ASMNT_YR'] - NAL_Model_Data_SFR_For_Preds['EFF_YR_BLT']

        NAL_Model_Data_SFR_For_Preds['Age_Diff'] = NAL_Model_Data_SFR_For_Preds['Age_Diff'].astype('Int64')
        NAL_Model_Data_SFR_For_Preds['Act_Age'] = NAL_Model_Data_SFR_For_Preds['Act_Age'].astype('Int64')
        NAL_Model_Data_SFR_For_Preds['Eff_Age'] = NAL_Model_Data_SFR_For_Preds['Eff_Age'].astype('Int64')

        NAL_Model_Data_SFR_For_Preds['Model_Prediction'] = NAL_Model.predict(NAL_Model_Data_SFR_For_Preds[st.session_state['Model_Features']])

        st.session_state["SFR_Model_Predicitons_All"] = NAL_Model_Data_SFR_For_Preds[['PARCEL_ID','Model_Prediction']].copy()

        Pop_Counts_SFR = NAL_Pop_Data_SFR.groupby(['NBRHD_CD']).size().to_frame('Nbhd_Counts')

        Sale_Counts_SFR = NAL_Model_Data_SFR.groupby(['NBRHD_CD']).size().to_frame('Sale_Counts')

        Pop_Sale_Review_SFR  = pd.merge(Pop_Counts_SFR
                                        , Sale_Counts_SFR
                                        , on='NBRHD_CD'
                                        , how='left')


        Pop_Sale_Review_SFR['Sale_Counts'] = Pop_Sale_Review_SFR['Sale_Counts'].fillna(0)

        Pop_Sale_Review_SFR['Sales_Coverage_Percent'] = round(Pop_Sale_Review_SFR['Sale_Counts'] / Pop_Sale_Review_SFR['Nbhd_Counts'],3)
        Pop_Sale_Review_SFR['Sales_Coverage_Flag'] = Pop_Sale_Review_SFR['Sales_Coverage_Percent'].apply(lambda x: 'Ample Sale Coverage' if x >= .1 else 'Low Sale Coverage' if x> 0 else 'No Sale Coverage')

        Pop_Data_SFR_For_Review = NAL_Model_Data_SFR_For_Preds[['PARCEL_ID','DOR_UC','MKT_AR','NBRHD_CD','JV','Model_Prediction']].copy()

        Pop_Data_SFR_For_Review['Model_Prediction'] = round(Pop_Data_SFR_For_Review['Model_Prediction'],0 ) 

        Pop_Data_SFR_For_Review['Model_Value_Pct_Diff_Just_Value'] = round((Pop_Data_SFR_For_Review['Model_Prediction'] - (Pop_Data_SFR_For_Review['JV'] / .85)) / (Pop_Data_SFR_For_Review['JV'] / .85) ,3)


        Pop_Data_SFR_For_Review_Complete = pd.merge(Pop_Data_SFR_For_Review
                                                        ,Pop_Sale_Review_SFR
                                                        ,how= 'left'
                                                        ,on='NBRHD_CD' )
        
        if st.session_state['NBHD_Encoder'] is not None:
            Pop_Data_SFR_For_Review_Complete['NBRHD_CD'] = st.session_state['NBHD_Encoder'].inverse_transform(Pop_Data_SFR_For_Review_Complete[['NBRHD_CD']]).flatten()

        st.session_state["SFR_Model_Predicitons_For_Review"] = Pop_Data_SFR_For_Review_Complete.copy()



        print('SIZE OF ALL SFR PREDICTIONS IS:', st.session_state["SFR_Model_Predicitons_All"].shape[0]) 

        DOR_Summary_Stratum1 =  DOR_Summary_Report_Generator( Stratum_Data = st.session_state["SFR_Model_Predicitons_All"].copy()  
                                                             , Sales_Data = st.session_state['SDF'].copy() 
                                                             , Stratum_Number = '01')

        st.session_state["Stratum1_report"] , st.session_state["Stratum1GroupLevelStatistics_Report"] = DOR_Summary_Stratum1.Calculate_DOR_Summary_Data()

        st.session_state["sfr_preformance_dict"] = {'Train R Squared' : round(r2_score(  NAL_Data_Train_SFR[Model_Outcome], NAL_Model.predict(NAL_Data_Train_SFR_Features).ravel() ),Result_Prescision)
                                
                                                    , 'Train Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Data_Train_SFR[Model_Outcome] , NAL_Model.predict(NAL_Data_Train_SFR_Features).ravel() ),Result_Prescision)
                                                    
                                                    , 'Train COD' : round(COD(NAL_Data_Train_SFR[Model_Outcome].values.ravel() , NAL_Model.predict(NAL_Data_Train_SFR_Features).ravel() ),Result_Prescision)  

                                                    , 'Train PRD' : round(PRD(NAL_Data_Train_SFR[Model_Outcome].values.ravel() , NAL_Model.predict(NAL_Data_Train_SFR_Features).ravel()),Result_Prescision)

                                                    , 'Train MAE' : round(mean_absolute_error(NAL_Data_Train_SFR[Model_Outcome] , NAL_Model.predict(NAL_Data_Train_SFR_Features).ravel() ),Result_Prescision)

                                                    , 'Test R Squared' : round(r2_score(  NAL_Data_Test_SFR[Model_Outcome], NAL_Model.predict(NAL_Data_Test_SFR_Features).ravel() ),Result_Prescision)

                                                    , 'Test Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Data_Test_SFR[Model_Outcome] , NAL_Model.predict(NAL_Data_Test_SFR_Features).ravel() ),Result_Prescision)

                                                    , 'Test COD' : round(COD(NAL_Data_Test_SFR[Model_Outcome].values.ravel() , NAL_Model.predict(NAL_Data_Test_SFR_Features).ravel() ),Result_Prescision)   

                                                    , 'Test PRD' : round(PRD(NAL_Data_Test_SFR[Model_Outcome].values.ravel() , NAL_Model.predict(NAL_Data_Test_SFR_Features).ravel()),Result_Prescision)

                                                    , 'Test MAE' : round(mean_absolute_error(NAL_Data_Test_SFR[Model_Outcome] , NAL_Model.predict(NAL_Data_Test_SFR_Features).ravel() ),Result_Prescision)

                                                    , 'Number of Records Before Trimming Train' : NAL_Model_Train_SFR_results.shape[0]
                                                    , 'Number of Records After Trimming Train' : NAL_Model_Train_SFR_results_Trimmed.shape[0]

                                                    , 'Number of Records Before Trimming Test' : NAL_Model_Test_SFR_results.shape[0]
                                                    , 'Number of Records After Trimming Test' : NAL_Model_Test_results_Trimmed.shape[0]
                                                    
                                                    ,'Trimmed Train R Squared' : round(r2_score(  NAL_Model_Train_SFR_results_Trimmed[Model_Outcome], NAL_Model_Train_SFR_results_Trimmed['Model_Prediction']  ),Result_Prescision)

                                                    , 'Trimmed Train Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Train_SFR_results_Trimmed[Model_Outcome] , NAL_Model_Train_SFR_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Train COD' : round(COD(NAL_Model_Train_SFR_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Train_SFR_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Train PRD' : round(PRD(NAL_Model_Train_SFR_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Train_SFR_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    , 'Trimmed Train MAE' : round(mean_absolute_error(NAL_Model_Train_SFR_results_Trimmed[Model_Outcome] , NAL_Model_Train_SFR_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    ,'Trimmed Test R Squared' : round(r2_score(  NAL_Model_Test_results_Trimmed[Model_Outcome], NAL_Model_Test_results_Trimmed['Model_Prediction']  ),Result_Prescision)

                                                    , 'Trimmed Test Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_results_Trimmed[Model_Outcome] , NAL_Model_Test_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Test COD' : round(COD(NAL_Model_Test_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Test PRD' : round(PRD(NAL_Model_Test_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    , 'Trimmed Test MAE' : round(mean_absolute_error(NAL_Model_Test_results_Trimmed[Model_Outcome] , NAL_Model_Test_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    # ,'Trimmed Test NAL R Squared' : round(r2_score(  NAL_Model_Test_results_Trimmed[Model_Outcome], NAL_Model_Test_results_Trimmed['JV']  /.85 ),Result_Prescision)

                                                    # , 'Trimmed Test NAL Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_results_Trimmed[Model_Outcome] , NAL_Model_Test_results_Trimmed['JV']  /.85 ),Result_Prescision)

                                                    # , 'Trimmed Test NAL COD' : round(COD(NAL_Model_Test_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_results_Trimmed['JV']/.85  ),Result_Prescision)

                                                    # , 'Trimmed Test NAL PRD' : round(PRD(NAL_Model_Test_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_results_Trimmed['JV']/.85 ),Result_Prescision)

                                                    # , 'Trimmed Test NAL MAE' : round(mean_absolute_error(NAL_Model_Test_results_Trimmed[Model_Outcome] , NAL_Model_Test_results_Trimmed['JV']/.85 ),Result_Prescision)

                                                    ,'Trimmed Test NAL R Squared' : round(r2_score(  NAL_Model_Test_results_Trimmed_JV[Model_Outcome], NAL_Model_Test_results_Trimmed_JV['JV']  /.85 ),Result_Prescision)

                                                    , 'Trimmed Test NAL Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_results_Trimmed_JV[Model_Outcome] , NAL_Model_Test_results_Trimmed_JV['JV']  /.85 ),Result_Prescision)

                                                    , 'Trimmed Test NAL COD' : round(COD(NAL_Model_Test_results_Trimmed_JV[Model_Outcome].values.ravel() , NAL_Model_Test_results_Trimmed_JV['JV']/.85  ),Result_Prescision)

                                                    , 'Trimmed Test NAL PRD' : round(PRD(NAL_Model_Test_results_Trimmed_JV[Model_Outcome].values.ravel() , NAL_Model_Test_results_Trimmed_JV['JV']/.85 ),Result_Prescision)

                                                    , 'Trimmed Test NAL MAE' : round(mean_absolute_error(NAL_Model_Test_results_Trimmed_JV[Model_Outcome] , NAL_Model_Test_results_Trimmed_JV['JV']/.85 ),Result_Prescision)

                                                    
                                                    }


        current_Loading_progress = int(current_Loading_progress + Model_percent_complete)

        pg_bar.progress(current_Loading_progress, text = '**Please hang on tight we are building something cool together!**')  



    if st.session_state["Vac_Res_model_binary"] == True:
        NAL_Data_Train_Vac_Res, NAL_Data_Test_Vac_Res = train_test_split(NAL_ModelData_Vac
                                                                            , test_size=0.2
                                                                            , random_state=42
                                                                            , stratify = NAL_ModelData_Vac['SALE_YR'])
        
        NAL_Data_Train_Vac_Res_Features = NAL_Data_Train_Vac_Res[st.session_state['Model_Features_Vac_Res']]

        NAL_Data_Test_Vac_Res_Features = NAL_Data_Test_Vac_Res[st.session_state['Model_Features_Vac_Res']]


        if st.session_state['Hyperparameter_Tuning_Toggle'] == False : 
            NAL_Model_Vac_Res = ltb.LGBMRegressor(n_estimators = 500
                                                    , min_child_samples = 3
                                                    , max_depth = 10
                                                    , random_state = 42)
            

            NAL_Model_Vac_Res.fit(NAL_Data_Train_Vac_Res_Features
                                    , NAL_Data_Train_Vac_Res[Model_Outcome])
            
        else:
            print('---- PERFORMING VAC RES HYPERPARAMETER TUNING ----')
            LGBM_study = optuna.create_study(direction='minimize')
            LGBM_study_obj_func = lambda trial: Model_Objective(trial
                                                    , Model_Train_Data = NAL_Data_Train_Vac_Res
                                                    , Model_Test_Data = NAL_ModelData_Vac
                                                    , model_features = st.session_state['Model_Features_Vac_Res']
                                                    , performance_metric = 'COD'
                                                    , model_early_stopping_Rounds = 150)    
            
            LGBM_study.optimize(LGBM_study_obj_func
                                , n_trials = st.session_state['Hyperparameter_Tuning_Trials'])
            
            LGBM_Best_Params = LGBM_study.best_params.copy()
            LGBM_Best_Params = {**LGBM_Best_Params, **{'random_state': 42}}
            #LGBM_Best_Params = {**LGBM_Best_Params, **{'n_estimators': 1000}}
            LGBM_Best_Params = {**LGBM_Best_Params, **{'verbose': -1}}

            NAL_Model_Vac_Res = ltb.LGBMRegressor(**LGBM_Best_Params)
            NAL_Model_Vac_Res.fit(NAL_Data_Train_Vac_Res_Features
                                    , NAL_Data_Train_Vac_Res[Model_Outcome[0]]
                                #    ,eval_set=[(NAL_Data_Test_Vac_Res[st.session_state['Model_Features_Vac_Res']], NAL_Data_Test_Vac_Res[Model_Outcome[0]])]
                                #    , eval_names=['train', 'test']
                                #    , callbacks=[ltb.early_stopping(stopping_rounds = 150, verbose=False)]
                                    )
            

        

        NAL_Model_Train_Vac_Res_results = NAL_Data_Train_Vac_Res.copy()

        NAL_Model_Train_Vac_Res_results['Model_Prediction'] = NAL_Model_Vac_Res.predict(NAL_Data_Train_Vac_Res_Features)

        NAL_Model_Train_Vac_Res_results['sale_ratio'] = NAL_Model_Train_Vac_Res_results['Model_Prediction'] / NAL_Model_Train_Vac_Res_results[Model_Outcome[0]] 

        NAL_Model_Train_Vac_Res_results_Trimmed = NAL_Model_Train_Vac_Res_results[(NAL_Model_Train_Vac_Res_results['sale_ratio'] > .3) & (NAL_Model_Train_Vac_Res_results['sale_ratio'] < 1.7)]

        #print('Number of Records Before Trimming: ', NAL_Model_Train_Vac_Res_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Train_Vac_Res_results_Trimmed.shape[0])

        NAL_Model_Test_Vac_Res_results = NAL_Data_Test_Vac_Res.copy()

        NAL_Model_Test_Vac_Res_results['Model_Prediction'] = NAL_Model_Vac_Res.predict(NAL_Data_Test_Vac_Res_Features)

        NAL_Model_Test_Vac_Res_results['sale_ratio'] = NAL_Model_Test_Vac_Res_results['Model_Prediction'] / NAL_Model_Test_Vac_Res_results[Model_Outcome[0]] 

        NAL_Model_Test_Vac_Res_results['sale_ratio_JV'] = (NAL_Model_Test_Vac_Res_results['JV'] / .85) / NAL_Model_Test_Vac_Res_results[Model_Outcome[0]] 

        NAL_Model_Test_Vac_Res_results_Trimmed = NAL_Model_Test_Vac_Res_results[(NAL_Model_Test_Vac_Res_results['sale_ratio'] > .3) & (NAL_Model_Test_Vac_Res_results['sale_ratio'] < 1.7)]

        NAL_Model_Test_Vac_Res_results_Trimmed_JV = NAL_Model_Test_Vac_Res_results[(NAL_Model_Test_Vac_Res_results['sale_ratio_JV'] > .3) & (NAL_Model_Test_Vac_Res_results['sale_ratio_JV'] < 1.7)]

        #print(NAL_Model_Test_results_Trimmed.info())

        #print('Number of Records Before Trimming: ', NAL_Model_Test_Vac_Res_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Test_Vac_Res_results_Trimmed.shape[0])

        NAL_Model_Data_Vac_Res_For_Preds = st.session_state["NAL"].copy()

        if st.session_state['NBHD_Encoder'] != None:
            NAL_Model_Data_Vac_Res_For_Preds['NBRHD_CD'] = st.session_state['NBHD_Encoder'].transform(NAL_Model_Data_Vac_Res_For_Preds[['NBRHD_CD']])

        NAL_Model_Data_Vac_Res_For_Preds = NAL_Model_Data_Vac_Res_For_Preds[NAL_Model_Data_Vac_Res_For_Preds['DOR_UC'].isin([0])]

        NAL_Model_Data_Vac_Res_For_Preds['Time_to_Sale'] = 12*(st.session_state['Valuation_Year'] - min(st.session_state["SDF"]['SALE_YR'])) + 0

        NAL_Model_Data_Vac_Res_For_Preds['Age_Diff'] =  NAL_Model_Data_Vac_Res_For_Preds['EFF_YR_BLT'] - NAL_Model_Data_Vac_Res_For_Preds['ACT_YR_BLT'] 
        NAL_Model_Data_Vac_Res_For_Preds['Act_Age'] = NAL_Model_Data_Vac_Res_For_Preds['ASMNT_YR'] - NAL_Model_Data_Vac_Res_For_Preds['ACT_YR_BLT']
        NAL_Model_Data_Vac_Res_For_Preds['Eff_Age'] =  NAL_Model_Data_Vac_Res_For_Preds['ASMNT_YR'] - NAL_Model_Data_Vac_Res_For_Preds['EFF_YR_BLT']

        NAL_Model_Data_Vac_Res_For_Preds['Age_Diff'] = NAL_Model_Data_Vac_Res_For_Preds['Age_Diff'].astype('Int64')
        NAL_Model_Data_Vac_Res_For_Preds['Act_Age'] = NAL_Model_Data_Vac_Res_For_Preds['Act_Age'].astype('Int64')
        NAL_Model_Data_Vac_Res_For_Preds['Eff_Age'] = NAL_Model_Data_Vac_Res_For_Preds['Eff_Age'].astype('Int64')

        NAL_Model_Data_Vac_Res_For_Preds['Model_Prediction'] = NAL_Model_Vac_Res.predict(NAL_Model_Data_Vac_Res_For_Preds[st.session_state['Model_Features_Vac_Res']])

        st.session_state["Vac_Res_Model_Predicitons_All"] = NAL_Model_Data_Vac_Res_For_Preds[['PARCEL_ID','Model_Prediction']].copy()

        print('SIZE OF ALL Vacant Res PREDICTIONS IS:', st.session_state["Vac_Res_Model_Predicitons_All"].shape[0]) 

        DOR_Summary_Stratum4 =  DOR_Summary_Report_Generator( Stratum_Data = st.session_state["Vac_Res_Model_Predicitons_All"].copy()  
                                                             ,Sales_Data = st.session_state['SDF'].copy()
                                                            , Stratum_Number = '04' )

        st.session_state["Stratum4_report"] , st.session_state["Stratum4GroupLevelStatistics_Report"] = DOR_Summary_Stratum4.Calculate_DOR_Summary_Data()

        st.session_state["Vac_Res_preformance_dict"] = {'Train R Squared' : round(r2_score(  NAL_Data_Train_Vac_Res[Model_Outcome], NAL_Model_Vac_Res.predict(NAL_Data_Train_Vac_Res_Features).ravel() ),Result_Prescision)
                                
                                                    , 'Train Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Data_Train_Vac_Res[Model_Outcome] , NAL_Model_Vac_Res.predict(NAL_Data_Train_Vac_Res_Features).ravel() ),Result_Prescision)
                                                    
                                                    , 'Train COD' : round(COD(NAL_Data_Train_Vac_Res[Model_Outcome].values.ravel() , NAL_Model_Vac_Res.predict(NAL_Data_Train_Vac_Res_Features).ravel() ),Result_Prescision)  

                                                    , 'Train PRD' : round(PRD(NAL_Data_Train_Vac_Res[Model_Outcome].values.ravel() , NAL_Model_Vac_Res.predict(NAL_Data_Train_Vac_Res_Features).ravel()),Result_Prescision)

                                                    , 'Train MAE' : round(mean_absolute_error(NAL_Data_Train_Vac_Res[Model_Outcome] , NAL_Model_Vac_Res.predict(NAL_Data_Train_Vac_Res_Features).ravel() ),Result_Prescision)

                                                    , 'Test R Squared' : round(r2_score(  NAL_Data_Test_Vac_Res[Model_Outcome], NAL_Model_Vac_Res.predict(NAL_Data_Test_Vac_Res_Features).ravel() ),Result_Prescision)

                                                    , 'Test Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Data_Test_Vac_Res[Model_Outcome] , NAL_Model_Vac_Res.predict(NAL_Data_Test_Vac_Res_Features).ravel() ),Result_Prescision)

                                                    , 'Test COD' : round(COD(NAL_Data_Test_Vac_Res[Model_Outcome].values.ravel() , NAL_Model_Vac_Res.predict(NAL_Data_Test_Vac_Res_Features).ravel() ),Result_Prescision)   

                                                    , 'Test PRD' : round(PRD(NAL_Data_Test_Vac_Res[Model_Outcome].values.ravel() , NAL_Model_Vac_Res.predict(NAL_Data_Test_Vac_Res_Features).ravel()),Result_Prescision)

                                                    , 'Test MAE' : round(mean_absolute_error(NAL_Data_Test_Vac_Res[Model_Outcome] , NAL_Model_Vac_Res.predict(NAL_Data_Test_Vac_Res_Features).ravel() ),Result_Prescision)

                                                    , 'Number of Records Before Trimming Train' : NAL_Model_Train_Vac_Res_results.shape[0]
                                                    , 'Number of Records After Trimming Train' : NAL_Model_Train_Vac_Res_results_Trimmed.shape[0]

                                                    , 'Number of Records Before Trimming Test' : NAL_Model_Test_Vac_Res_results.shape[0]
                                                    , 'Number of Records After Trimming Test' : NAL_Model_Test_Vac_Res_results_Trimmed.shape[0]
                                                    
                                                    ,'Trimmed Train R Squared' : round(r2_score(  NAL_Model_Train_Vac_Res_results_Trimmed[Model_Outcome], NAL_Model_Train_Vac_Res_results_Trimmed['Model_Prediction']  ),Result_Prescision)

                                                    , 'Trimmed Train Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Train_Vac_Res_results_Trimmed[Model_Outcome] , NAL_Model_Train_Vac_Res_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Train COD' : round(COD(NAL_Model_Train_Vac_Res_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Train_Vac_Res_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Train PRD' : round(PRD(NAL_Model_Train_Vac_Res_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Train_Vac_Res_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    , 'Trimmed Train MAE' : round(mean_absolute_error(NAL_Model_Train_Vac_Res_results_Trimmed[Model_Outcome] , NAL_Model_Train_Vac_Res_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    ,'Trimmed Test R Squared' : round(r2_score(  NAL_Model_Test_Vac_Res_results_Trimmed[Model_Outcome], NAL_Model_Test_Vac_Res_results_Trimmed['Model_Prediction']  ),Result_Prescision)

                                                    , 'Trimmed Test Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_Vac_Res_results_Trimmed[Model_Outcome] , NAL_Model_Test_Vac_Res_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Test COD' : round(COD(NAL_Model_Test_Vac_Res_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Vac_Res_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Test PRD' : round(PRD(NAL_Model_Test_Vac_Res_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Vac_Res_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    , 'Trimmed Test MAE' : round(mean_absolute_error(NAL_Model_Test_Vac_Res_results_Trimmed[Model_Outcome] , NAL_Model_Test_Vac_Res_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    # ,'Trimmed Test NAL R Squared' : round(r2_score(  NAL_Model_Test_Vac_Res_results_Trimmed[Model_Outcome], NAL_Model_Test_Vac_Res_results_Trimmed['JV'] /.85  ),Result_Prescision)

                                                    # , 'Trimmed Test NAL Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_Vac_Res_results_Trimmed[Model_Outcome] , NAL_Model_Test_Vac_Res_results_Trimmed['JV'] /.85 ),Result_Prescision)

                                                    # , 'Trimmed Test NAL COD' : round(COD(NAL_Model_Test_Vac_Res_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Vac_Res_results_Trimmed['JV'] /.85  ),Result_Prescision)

                                                    # , 'Trimmed Test NAL PRD' : round(PRD(NAL_Model_Test_Vac_Res_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Vac_Res_results_Trimmed['JV'] /.85 ),Result_Prescision)

                                                    # , 'Trimmed Test NAL MAE' : round(mean_absolute_error(NAL_Model_Test_Vac_Res_results_Trimmed[Model_Outcome] , NAL_Model_Test_Vac_Res_results_Trimmed['JV'] /.85 ),Result_Prescision)

                                                    ,'Trimmed Test NAL R Squared' : round(r2_score(  NAL_Model_Test_Vac_Res_results_Trimmed_JV[Model_Outcome], NAL_Model_Test_Vac_Res_results_Trimmed_JV['JV'] /.85  ),Result_Prescision)

                                                    , 'Trimmed Test NAL Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_Vac_Res_results_Trimmed_JV[Model_Outcome] , NAL_Model_Test_Vac_Res_results_Trimmed_JV['JV'] /.85 ),Result_Prescision)

                                                    , 'Trimmed Test NAL COD' : round(COD(NAL_Model_Test_Vac_Res_results_Trimmed_JV[Model_Outcome].values.ravel() , NAL_Model_Test_Vac_Res_results_Trimmed_JV['JV'] /.85  ),Result_Prescision)

                                                    , 'Trimmed Test NAL PRD' : round(PRD(NAL_Model_Test_Vac_Res_results_Trimmed_JV[Model_Outcome].values.ravel() , NAL_Model_Test_Vac_Res_results_Trimmed_JV['JV'] /.85 ),Result_Prescision)

                                                    , 'Trimmed Test NAL MAE' : round(mean_absolute_error(NAL_Model_Test_Vac_Res_results_Trimmed_JV[Model_Outcome] , NAL_Model_Test_Vac_Res_results_Trimmed_JV['JV'] /.85 ),Result_Prescision)

                                                    
                                                    }


        current_Loading_progress = int(current_Loading_progress + Model_percent_complete)

        pg_bar.progress(current_Loading_progress, text = '**Please hang on tight we are building something cool together!**')    

 

    if st.session_state["Condo_model_binary"] == True:
        NAL_Data_Train_Condo, NAL_Data_Test_Condo = train_test_split(NAL_Model_Data_Condo
                                                                            , test_size=0.2
                                                                            , random_state=42
                                                                            , stratify = NAL_Model_Data_Condo['SALE_YR'])
        
        NAL_Data_Train_Condo_Features = NAL_Data_Train_Condo[st.session_state['Model_Features_Condos']]

        NAL_Data_Test_Condo_Features = NAL_Data_Test_Condo[st.session_state['Model_Features_Condos']]


        if st.session_state['Hyperparameter_Tuning_Toggle'] == False : 
            NAL_Model_Condo = ltb.LGBMRegressor(n_estimators = 500
                                                    , min_child_samples = 3
                                                    , max_depth = 10
                                                    , random_state = 42)
            

            NAL_Model_Condo.fit(NAL_Data_Train_Condo_Features
                                    , NAL_Data_Train_Condo[Model_Outcome])
            
        else:
            print('---- PERFORMING CONDO HYPERPARAMETER TUNING ----')
            LGBM_study = optuna.create_study(direction='minimize')
            LGBM_study_obj_func = lambda trial: Model_Objective(trial
                                                    , Model_Train_Data = NAL_Data_Train_Condo
                                                    , Model_Test_Data = NAL_Data_Test_Condo
                                                    , model_features = st.session_state['Model_Features_Condos']
                                                    , performance_metric = 'COD'
                                                    , model_early_stopping_Rounds = 150)    
            
            LGBM_study.optimize(LGBM_study_obj_func
                                , n_trials = st.session_state['Hyperparameter_Tuning_Trials'])
            
            LGBM_Best_Params = LGBM_study.best_params.copy()
            LGBM_Best_Params = {**LGBM_Best_Params, **{'random_state': 42}}
            #LGBM_Best_Params = {**LGBM_Best_Params, **{'n_estimators': 1000}}
            LGBM_Best_Params = {**LGBM_Best_Params, **{'verbose': -1}}

            NAL_Model_Condo = ltb.LGBMRegressor(**LGBM_Best_Params)
            

            NAL_Model_Condo.fit(NAL_Data_Train_Condo_Features
                                    , NAL_Data_Train_Condo[Model_Outcome[0]]
                                    ,eval_set=[(NAL_Data_Test_Condo[st.session_state['Model_Features_Condos']], NAL_Data_Test_Condo[Model_Outcome[0]])]
                                    , eval_names = ['train', 'test']
                                    , callbacks=[ltb.early_stopping(stopping_rounds = 150, verbose=False)])
        

        NAL_Model_Train_Condo_results = NAL_Data_Train_Condo.copy()

        NAL_Model_Train_Condo_results['Model_Prediction'] = NAL_Model_Condo.predict(NAL_Data_Train_Condo_Features)

        NAL_Model_Train_Condo_results['sale_ratio'] = NAL_Model_Train_Condo_results['Model_Prediction'] / NAL_Model_Train_Condo_results[Model_Outcome[0]] 

        NAL_Model_Train_Condo_results['sale_ratio_JV'] = (NAL_Model_Train_Condo_results['JV'] / 0.85) / NAL_Model_Train_Condo_results[Model_Outcome[0]] 

        NAL_Model_Train_Condo_results_Trimmed = NAL_Model_Train_Condo_results[(NAL_Model_Train_Condo_results['sale_ratio'] > .3) & (NAL_Model_Train_Condo_results['sale_ratio'] < 1.7)]

        NAL_Model_Train_Condo_results_Trimmed_JV = NAL_Model_Train_Condo_results[(NAL_Model_Train_Condo_results['sale_ratio_JV'] > .3) & (NAL_Model_Train_Condo_results['sale_ratio_JV'] < 1.7)]

        #print('Number of Records Before Trimming: ', NAL_Model_Train_Condo_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Train_Condo_results_Trimmed.shape[0])

        NAL_Model_Test_Condo_results = NAL_Data_Test_Condo.copy()

        NAL_Model_Test_Condo_results['Model_Prediction'] = NAL_Model_Condo.predict(NAL_Data_Test_Condo_Features)

        NAL_Model_Test_Condo_results['sale_ratio'] = NAL_Model_Test_Condo_results['Model_Prediction'] / NAL_Model_Test_Condo_results[Model_Outcome[0]] 

        NAL_Model_Test_Condo_results_Trimmed = NAL_Model_Test_Condo_results[(NAL_Model_Test_Condo_results['sale_ratio'] > .3) & (NAL_Model_Test_Condo_results['sale_ratio'] < 1.7)]

        #print(NAL_Model_Test_results_Trimmed.info())

        #print('Number of Records Before Trimming: ', NAL_Model_Test_Condo_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Test_Condo_results_Trimmed.shape[0])

        NAL_Model_Data_Condo_For_Preds = st.session_state["NAL"].copy()

        if st.session_state['NBHD_Encoder'] != None:
            NAL_Model_Data_Condo_For_Preds['NBRHD_CD'] = st.session_state['NBHD_Encoder'].transform(NAL_Model_Data_Condo_For_Preds[['NBRHD_CD']])

        NAL_Model_Data_Condo_For_Preds = NAL_Model_Data_Condo_For_Preds[NAL_Model_Data_Condo_For_Preds['DOR_UC'].isin([4,5])]

        NAL_Model_Data_Condo_For_Preds['Time_to_Sale'] = 12*(st.session_state['Valuation_Year'] - min(st.session_state["SDF"]['SALE_YR'])) + 0

        NAL_Model_Data_Condo_For_Preds['Age_Diff'] =  NAL_Model_Data_Condo_For_Preds['EFF_YR_BLT'] - NAL_Model_Data_Condo_For_Preds['ACT_YR_BLT'] 
        NAL_Model_Data_Condo_For_Preds['Act_Age'] = NAL_Model_Data_Condo_For_Preds['ASMNT_YR'] - NAL_Model_Data_Condo_For_Preds['ACT_YR_BLT']
        NAL_Model_Data_Condo_For_Preds['Eff_Age'] =  NAL_Model_Data_Condo_For_Preds['ASMNT_YR'] - NAL_Model_Data_Condo_For_Preds['EFF_YR_BLT']

        NAL_Model_Data_Condo_For_Preds['Age_Diff'] = NAL_Model_Data_Condo_For_Preds['Age_Diff'].astype('Int64')
        NAL_Model_Data_Condo_For_Preds['Act_Age'] = NAL_Model_Data_Condo_For_Preds['Act_Age'].astype('Int64')
        NAL_Model_Data_Condo_For_Preds['Eff_Age'] = NAL_Model_Data_Condo_For_Preds['Eff_Age'].astype('Int64')

        NAL_Model_Data_Condo_For_Preds['Model_Prediction'] = NAL_Model_Condo.predict(NAL_Model_Data_Condo_For_Preds[st.session_state['Model_Features_Condos']])

        st.session_state["Condo_Model_Predicitons_All"] = NAL_Model_Data_Condo_For_Preds[['PARCEL_ID','Model_Prediction']].copy()

        print('SIZE OF ALL CONDO PREDICTIONS IS:', st.session_state["Condo_Model_Predicitons_All"].shape[0]) 

        DOR_Summary_Stratum1_Condos =  DOR_Summary_Report_Generator( Stratum_Data = st.session_state["Condo_Model_Predicitons_All"].copy()  
                                                                    , Sales_Data = st.session_state['SDF'].copy()
                                                                     , Stratum_Number = '01' )

        st.session_state["Stratum1_report_Condos"] , st.session_state["Stratum1GroupLevelStatistics_Report_Condos"] = DOR_Summary_Stratum1_Condos.Calculate_DOR_Summary_Data()

        st.session_state["Condo_preformance_dict"] = {'Train R Squared' : round(r2_score(  NAL_Data_Train_Condo[Model_Outcome], NAL_Model_Condo.predict(NAL_Data_Train_Condo_Features).ravel() ),Result_Prescision)
                                
                                                    , 'Train Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Data_Train_Condo[Model_Outcome] , NAL_Model_Condo.predict(NAL_Data_Train_Condo_Features).ravel() ),Result_Prescision)
                                                    
                                                    , 'Train COD' : round(COD(NAL_Data_Train_Condo[Model_Outcome].values.ravel() , NAL_Model_Condo.predict(NAL_Data_Train_Condo_Features).ravel() ),Result_Prescision)  

                                                    , 'Train PRD' : round(PRD(NAL_Data_Train_Condo[Model_Outcome].values.ravel() , NAL_Model_Condo.predict(NAL_Data_Train_Condo_Features).ravel()),Result_Prescision)

                                                    , 'Train MAE' : round(mean_absolute_error(NAL_Data_Train_Condo[Model_Outcome] , NAL_Model_Condo.predict(NAL_Data_Train_Condo_Features).ravel() ),Result_Prescision)

                                                    , 'Test R Squared' : round(r2_score(  NAL_Data_Test_Condo[Model_Outcome], NAL_Model_Condo.predict(NAL_Data_Test_Condo_Features).ravel() ),Result_Prescision)

                                                    , 'Test Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Data_Test_Condo[Model_Outcome] , NAL_Model_Condo.predict(NAL_Data_Test_Condo_Features).ravel() ),Result_Prescision)

                                                    , 'Test COD' : round(COD(NAL_Data_Test_Condo[Model_Outcome].values.ravel() , NAL_Model_Condo.predict(NAL_Data_Test_Condo_Features).ravel() ),Result_Prescision)   

                                                    , 'Test PRD' : round(PRD(NAL_Data_Test_Condo[Model_Outcome].values.ravel() , NAL_Model_Condo.predict(NAL_Data_Test_Condo_Features).ravel()),Result_Prescision)

                                                    , 'Test MAE' : round(mean_absolute_error(NAL_Data_Test_Condo[Model_Outcome] , NAL_Model_Condo.predict(NAL_Data_Test_Condo_Features).ravel() ),Result_Prescision)

                                                    , 'Number of Records Before Trimming Train' : NAL_Model_Train_Condo_results.shape[0]
                                                    , 'Number of Records After Trimming Train' : NAL_Model_Train_Condo_results_Trimmed.shape[0]

                                                    , 'Number of Records Before Trimming Test' : NAL_Model_Test_Condo_results.shape[0]
                                                    , 'Number of Records After Trimming Test' : NAL_Model_Test_Condo_results_Trimmed.shape[0]
                                                    
                                                    ,'Trimmed Train R Squared' : round(r2_score(  NAL_Model_Train_Condo_results_Trimmed[Model_Outcome], NAL_Model_Train_Condo_results_Trimmed['Model_Prediction']  ),Result_Prescision)

                                                    , 'Trimmed Train Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Train_Condo_results_Trimmed[Model_Outcome] , NAL_Model_Train_Condo_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Train COD' : round(COD(NAL_Model_Train_Condo_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Train_Condo_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Train PRD' : round(PRD(NAL_Model_Train_Condo_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Train_Condo_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    , 'Trimmed Train MAE' : round(mean_absolute_error(NAL_Model_Train_Condo_results_Trimmed[Model_Outcome] , NAL_Model_Train_Condo_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    ,'Trimmed Test R Squared' : round(r2_score(  NAL_Model_Test_Condo_results_Trimmed[Model_Outcome], NAL_Model_Test_Condo_results_Trimmed['Model_Prediction']  ),Result_Prescision)

                                                    , 'Trimmed Test Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_Condo_results_Trimmed[Model_Outcome] , NAL_Model_Test_Condo_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Test COD' : round(COD(NAL_Model_Test_Condo_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Condo_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Test PRD' : round(PRD(NAL_Model_Test_Condo_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Condo_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    , 'Trimmed Test MAE' : round(mean_absolute_error(NAL_Model_Test_Condo_results_Trimmed[Model_Outcome] , NAL_Model_Test_Condo_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    # ,'Trimmed Test NAL R Squared' : round(r2_score(  NAL_Model_Test_Condo_results_Trimmed[Model_Outcome], NAL_Model_Test_Condo_results_Trimmed['JV'] /.85  ),Result_Prescision)

                                                    # , 'Trimmed Test NAL Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_Condo_results_Trimmed[Model_Outcome] , NAL_Model_Test_Condo_results_Trimmed['JV'] /.85 ),Result_Prescision)

                                                    # , 'Trimmed Test NAL COD' : round(COD(NAL_Model_Test_Condo_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Condo_results_Trimmed['JV']/.85  ),Result_Prescision)

                                                    # , 'Trimmed Test NAL PRD' : round(PRD(NAL_Model_Test_Condo_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Condo_results_Trimmed['JV']/.85 ),Result_Prescision)

                                                    # , 'Trimmed Test NAL MAE' : round(mean_absolute_error(NAL_Model_Test_Condo_results_Trimmed[Model_Outcome] , NAL_Model_Test_Condo_results_Trimmed['JV']/.85 ),Result_Prescision)
                                                    
                                                    ,'Trimmed Test NAL R Squared' : round(r2_score(  NAL_Model_Train_Condo_results_Trimmed_JV[Model_Outcome], NAL_Model_Train_Condo_results_Trimmed_JV['JV'] /.85  ),Result_Prescision)

                                                    , 'Trimmed Test NAL Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Train_Condo_results_Trimmed_JV[Model_Outcome] , NAL_Model_Train_Condo_results_Trimmed_JV['JV'] /.85 ),Result_Prescision)

                                                    , 'Trimmed Test NAL COD' : round(COD(NAL_Model_Train_Condo_results_Trimmed_JV[Model_Outcome].values.ravel() , NAL_Model_Train_Condo_results_Trimmed_JV['JV']/.85  ),Result_Prescision)

                                                    , 'Trimmed Test NAL PRD' : round(PRD(NAL_Model_Train_Condo_results_Trimmed_JV[Model_Outcome].values.ravel() , NAL_Model_Train_Condo_results_Trimmed_JV['JV']/.85 ),Result_Prescision)

                                                    , 'Trimmed Test NAL MAE' : round(mean_absolute_error(NAL_Model_Train_Condo_results_Trimmed_JV[Model_Outcome] , NAL_Model_Train_Condo_results_Trimmed_JV['JV']/.85 ),Result_Prescision)
                                                    
                                                    }


        current_Loading_progress = int(current_Loading_progress + Model_percent_complete)

        pg_bar.progress(current_Loading_progress, text = '**Please hang on tight we are building something cool together!**') 





    if st.session_state["Commercial_model_binary"] == True:
        NAL_Data_Train_Commercial, NAL_Data_Test_Commercial = train_test_split( NAL_Model_Data_Com
                                                                        , test_size=0.2
                                                                        , random_state=42
                                                                        , stratify = NAL_Model_Data_Com['SALE_YR'])
        
        NAL_Data_Train_Commercial_Features = NAL_Data_Train_Commercial[st.session_state['Model_Features_Commercial']]

        NAL_Data_Test_Commercial_Features = NAL_Data_Test_Commercial[st.session_state['Model_Features_Commercial']]

        if st.session_state['Hyperparameter_Tuning_Toggle'] == False : 
            NAL_Model_Commercial = ltb.LGBMRegressor(n_estimators = 500
                                                    , min_child_samples = 5
                                                # , max_depth = 10
                                                , max_depth = 4
                                                    , random_state = 42)
            

            NAL_Model_Commercial.fit(NAL_Data_Train_Commercial_Features
                                    , NAL_Data_Train_Commercial[Model_Outcome[0]])
        else:
            print('---- PERFORMING COMMERCIALHYPERPARAMETER TUNING ----')
            LGBM_study = optuna.create_study(direction='minimize')
            LGBM_study_obj_func = lambda trial: Model_Objective(trial
                                                    , Model_Train_Data = NAL_Data_Train_Commercial
                                                    , Model_Test_Data = NAL_Data_Test_Commercial
                                                    , model_features = st.session_state['Model_Features_Commercial']
                                                    , performance_metric = 'COD'
                                                    , model_early_stopping_Rounds = 150)    
            
            LGBM_study.optimize(LGBM_study_obj_func
                                , n_trials = st.session_state['Hyperparameter_Tuning_Trials'])
            
            LGBM_Best_Params = LGBM_study.best_params.copy()
            LGBM_Best_Params = {**LGBM_Best_Params, **{'random_state': 42}}
            #LGBM_Best_Params = {**LGBM_Best_Params, **{'n_estimators': 1000}}
            LGBM_Best_Params = {**LGBM_Best_Params, **{'verbose': -1}}

            NAL_Model_Commercial = ltb.LGBMRegressor(**LGBM_Best_Params)
            

            NAL_Model_Commercial.fit(NAL_Data_Train_Commercial_Features
                                    , NAL_Data_Train_Commercial[Model_Outcome[0]]
                                    , eval_set=[(NAL_Data_Test_Commercial[st.session_state['Model_Features_Commercial']], NAL_Data_Test_Commercial[Model_Outcome[0]])]
                                    , eval_names=['train', 'test']
                                    , callbacks=[ltb.early_stopping(stopping_rounds = 150, verbose=False)])


        NAL_Model_Train_Commercial_results = NAL_Data_Train_Commercial.copy()

        NAL_Model_Train_Commercial_results['Model_Prediction'] = NAL_Model_Commercial.predict(NAL_Data_Train_Commercial_Features)

        NAL_Model_Train_Commercial_results['sale_ratio'] = NAL_Model_Train_Commercial_results['Model_Prediction'] / NAL_Model_Train_Commercial_results[Model_Outcome[0]] 

        NAL_Model_Train_Commercial_results_Trimmed = NAL_Model_Train_Commercial_results[(NAL_Model_Train_Commercial_results['sale_ratio'] > .3) & (NAL_Model_Train_Commercial_results['sale_ratio'] < 1.7)]

        #print('Number of Records Before Trimming: ', NAL_Model_Train_Commercial_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Train_Commercial_results_Trimmed.shape[0])

        NAL_Model_Test_Commercial_results = NAL_Data_Test_Commercial.copy()

        NAL_Model_Test_Commercial_results['Model_Prediction'] = NAL_Model_Commercial.predict(NAL_Data_Test_Commercial_Features)

        NAL_Model_Test_Commercial_results['sale_ratio'] = NAL_Model_Test_Commercial_results['Model_Prediction'] / NAL_Model_Test_Commercial_results[Model_Outcome[0]] 

        NAL_Model_Test_Commercial_results['sale_ratio_JV'] = (NAL_Model_Test_Commercial_results['JV'] / 0.85) / NAL_Model_Test_Commercial_results[Model_Outcome[0]] 

        NAL_Model_Test_Commercial_results_Trimmed = NAL_Model_Test_Commercial_results[(NAL_Model_Test_Commercial_results['sale_ratio'] > .3) & (NAL_Model_Test_Commercial_results['sale_ratio'] < 1.7)]

        NAL_Model_Test_Commercial_results_Trimmed_JV = NAL_Model_Test_Commercial_results[(NAL_Model_Test_Commercial_results['sale_ratio_JV'] > .3) & (NAL_Model_Test_Commercial_results['sale_ratio_JV'] < 1.7)]

        #print(NAL_Model_Test_results_Trimmed.info())

        #print('Number of Records Before Trimming: ', NAL_Model_Test_Commercial_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Test_Commercial_results_Trimmed.shape[0])

        NAL_Model_Data_Commercial_For_Preds = st.session_state["NAL"].copy()

        if st.session_state['NBHD_Encoder'] != None:
            NAL_Model_Data_Commercial_For_Preds['NBRHD_CD'] = st.session_state['NBHD_Encoder'].transform(NAL_Model_Data_Commercial_For_Preds[['NBRHD_CD']])

        NAL_Model_Data_Commercial_For_Preds = NAL_Model_Data_Commercial_For_Preds[NAL_Model_Data_Commercial_For_Preds['DOR_UC'].isin([11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49])]

        NAL_Model_Data_Commercial_For_Preds['Time_to_Sale'] = 12*( st.session_state['Valuation_Year'] - min(st.session_state["SDF"]['SALE_YR'])) + 0

        NAL_Model_Data_Commercial_For_Preds['Age_Diff'] =  NAL_Model_Data_Commercial_For_Preds['EFF_YR_BLT'] - NAL_Model_Data_Commercial_For_Preds['ACT_YR_BLT'] 
        NAL_Model_Data_Commercial_For_Preds['Act_Age'] = NAL_Model_Data_Commercial_For_Preds['ASMNT_YR'] - NAL_Model_Data_Commercial_For_Preds['ACT_YR_BLT']
        NAL_Model_Data_Commercial_For_Preds['Eff_Age'] =  NAL_Model_Data_Commercial_For_Preds['ASMNT_YR'] - NAL_Model_Data_Commercial_For_Preds['EFF_YR_BLT']

        NAL_Model_Data_Commercial_For_Preds['Age_Diff'] = NAL_Model_Data_Commercial_For_Preds['Age_Diff'].astype('Int64')
        NAL_Model_Data_Commercial_For_Preds['Act_Age'] = NAL_Model_Data_Commercial_For_Preds['Act_Age'].astype('Int64')
        NAL_Model_Data_Commercial_For_Preds['Eff_Age'] = NAL_Model_Data_Commercial_For_Preds['Eff_Age'].astype('Int64')

        NAL_Model_Data_Commercial_For_Preds['Model_Prediction'] = NAL_Model_Commercial.predict(NAL_Model_Data_Commercial_For_Preds[st.session_state['Model_Features_Commercial']])

        st.session_state["Commercial_Model_Predicitons_All"] = NAL_Model_Data_Commercial_For_Preds[['PARCEL_ID','Model_Prediction']].copy()

        print('SIZE OF ALL COMMERCIAL PREDICTIONS IS:', st.session_state["Commercial_Model_Predicitons_All"].shape[0]) 

        DOR_Summary_Stratum6 =  DOR_Summary_Report_Generator( Stratum_Data = st.session_state["Commercial_Model_Predicitons_All"].copy()  
                                                             , Sales_Data = st.session_state['SDF'].copy()
                                                              , Stratum_Number = '06' )

        st.session_state["Stratum6_report"] , st.session_state["Stratum6GroupLevelStatistics_Report"] = DOR_Summary_Stratum6.Calculate_DOR_Summary_Data()

        st.session_state["Commercial_preformance_dict"] = {'Train R Squared' : round(r2_score(  NAL_Data_Train_Commercial[Model_Outcome], NAL_Model_Commercial.predict(NAL_Data_Train_Commercial_Features).ravel() ),Result_Prescision)
                                
                                                    , 'Train Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Data_Train_Commercial[Model_Outcome] , NAL_Model_Commercial.predict(NAL_Data_Train_Commercial_Features).ravel() ),Result_Prescision)
                                                    
                                                    , 'Train COD' : round(COD(NAL_Data_Train_Commercial[Model_Outcome].values.ravel() , NAL_Model_Commercial.predict(NAL_Data_Train_Commercial_Features).ravel() ),Result_Prescision)  

                                                    , 'Train PRD' : round(PRD(NAL_Data_Train_Commercial[Model_Outcome].values.ravel() , NAL_Model_Commercial.predict(NAL_Data_Train_Commercial_Features).ravel()),Result_Prescision)

                                                    , 'Train MAE' : round(mean_absolute_error(NAL_Data_Train_Commercial[Model_Outcome] , NAL_Model_Commercial.predict(NAL_Data_Train_Commercial_Features).ravel() ),Result_Prescision)

                                                    , 'Test R Squared' : round(r2_score(  NAL_Data_Test_Commercial[Model_Outcome], NAL_Model_Commercial.predict(NAL_Data_Test_Commercial_Features).ravel() ),Result_Prescision)

                                                    , 'Test Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Data_Test_Commercial[Model_Outcome] , NAL_Model_Commercial.predict(NAL_Data_Test_Commercial_Features).ravel() ),Result_Prescision)

                                                    , 'Test COD' : round(COD(NAL_Data_Test_Commercial[Model_Outcome].values.ravel() , NAL_Model_Commercial.predict(NAL_Data_Test_Commercial_Features).ravel() ),Result_Prescision)   

                                                    , 'Test PRD' : round(PRD(NAL_Data_Test_Commercial[Model_Outcome].values.ravel() , NAL_Model_Commercial.predict(NAL_Data_Test_Commercial_Features).ravel()),Result_Prescision)

                                                    , 'Test MAE' : round(mean_absolute_error(NAL_Data_Test_Commercial[Model_Outcome] , NAL_Model_Commercial.predict(NAL_Data_Test_Commercial_Features).ravel() ),Result_Prescision)

                                                    , 'Number of Records Before Trimming Train' : NAL_Model_Train_Commercial_results.shape[0]
                                                    , 'Number of Records After Trimming Train' : NAL_Model_Train_Commercial_results_Trimmed.shape[0]

                                                    , 'Number of Records Before Trimming Test' : NAL_Model_Test_Commercial_results.shape[0]
                                                    , 'Number of Records After Trimming Test' : NAL_Model_Test_Commercial_results_Trimmed.shape[0]
                                                    
                                                    ,'Trimmed Train R Squared' : round(r2_score(  NAL_Model_Train_Commercial_results_Trimmed[Model_Outcome], NAL_Model_Train_Commercial_results_Trimmed['Model_Prediction']  ),Result_Prescision)

                                                    , 'Trimmed Train Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Train_Commercial_results_Trimmed[Model_Outcome] , NAL_Model_Train_Commercial_results_Trimmed['Model_Prediction'] ),Result_Prescision)
                                                 #   , 'Trimmed Train Weighted Mean' : 5
                                                    , 'Trimmed Train COD' : round(COD(NAL_Model_Train_Commercial_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Train_Commercial_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Train PRD' : round(PRD(NAL_Model_Train_Commercial_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Train_Commercial_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    , 'Trimmed Train MAE' : round(mean_absolute_error(NAL_Model_Train_Commercial_results_Trimmed[Model_Outcome] , NAL_Model_Train_Commercial_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    ,'Trimmed Test R Squared' : round(r2_score(  NAL_Model_Test_Commercial_results_Trimmed[Model_Outcome], NAL_Model_Test_Commercial_results_Trimmed['Model_Prediction']  ),Result_Prescision)


                                                    , 'Trimmed Test Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_Commercial_results_Trimmed[Model_Outcome] , NAL_Model_Test_Commercial_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Test COD' : round(COD(NAL_Model_Test_Commercial_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Commercial_results_Trimmed['Model_Prediction'] ),Result_Prescision)

                                                    , 'Trimmed Test PRD' : round(PRD(NAL_Model_Test_Commercial_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Commercial_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    , 'Trimmed Test MAE' : round(mean_absolute_error(NAL_Model_Test_Commercial_results_Trimmed[Model_Outcome] , NAL_Model_Test_Commercial_results_Trimmed['Model_Prediction']),Result_Prescision)

                                                    # ,'Trimmed Test NAL R Squared' : round(r2_score(  NAL_Model_Test_Commercial_results_Trimmed[Model_Outcome], NAL_Model_Test_Commercial_results_Trimmed['JV']  /.85 ),Result_Prescision)

                                                    # , 'Trimmed Test NAL Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_Commercial_results_Trimmed[Model_Outcome] , NAL_Model_Test_Commercial_results_Trimmed['JV'] /.85 ),Result_Prescision)

                                                    # , 'Trimmed Test NAL COD' : round(COD(NAL_Model_Test_Commercial_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Commercial_results_Trimmed['JV'] /.85 ),Result_Prescision)

                                                    # , 'Trimmed Test NAL PRD' : round(PRD(NAL_Model_Test_Commercial_results_Trimmed[Model_Outcome].values.ravel() , NAL_Model_Test_Commercial_results_Trimmed['JV']/.85  ),Result_Prescision)

                                                    # , 'Trimmed Test NAL MAE' : round(mean_absolute_error(NAL_Model_Test_Commercial_results_Trimmed[Model_Outcome] , NAL_Model_Test_Commercial_results_Trimmed['JV']/.85 ),Result_Prescision)
                                                    
                                                    ,'Trimmed Test NAL R Squared' : round(r2_score(  NAL_Model_Test_Commercial_results_Trimmed_JV[Model_Outcome], NAL_Model_Test_Commercial_results_Trimmed_JV['JV']  /.85 ),Result_Prescision)

                                                    , 'Trimmed Test NAL Weighted Mean' : round(weighted_Mean_Sale_Ratio(NAL_Model_Test_Commercial_results_Trimmed_JV[Model_Outcome] , NAL_Model_Test_Commercial_results_Trimmed_JV['JV'] /.85 ),Result_Prescision)

                                                    , 'Trimmed Test NAL COD' : round(COD(NAL_Model_Test_Commercial_results_Trimmed_JV[Model_Outcome].values.ravel() , NAL_Model_Test_Commercial_results_Trimmed_JV['JV'] /.85 ),Result_Prescision)

                                                    , 'Trimmed Test NAL PRD' : round(PRD(NAL_Model_Test_Commercial_results_Trimmed_JV[Model_Outcome].values.ravel() , NAL_Model_Test_Commercial_results_Trimmed_JV['JV']/.85  ),Result_Prescision)

                                                    , 'Trimmed Test NAL MAE' : round(mean_absolute_error(NAL_Model_Test_Commercial_results_Trimmed_JV[Model_Outcome] , NAL_Model_Test_Commercial_results_Trimmed_JV['JV']/.85 ),Result_Prescision)
                                                    
                                                    }


        current_Loading_progress = int(current_Loading_progress + Model_percent_complete)

        pg_bar.progress(current_Loading_progress, text = '**Please hang on tight we are building something cool together!**') 



    ### Models Complete and final progress bar update 


    pg_bar.progress(100, text='**Please hang on tight we are building something cool together!**')  

    time.sleep(1)

    # for percent_complete in range(100):
    #     time.sleep(0.05)
    #     pg_bar.progress(percent_complete + 1, text='**Please hang on tight we are building something cool together!**')  
        


####### Creating Chat Agent to Review the Data #######

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)
  
def list_tables() -> list[str]:
    """Retrieve the names of all tables in the database."""
    # Include print logging statements so you can see when functions are being called.
    print(' - DB CALL: list_tables()')

    cursor = st.session_state["Chat_Bot_Data_Base"].cursor()

    # Fetch the table names.
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    tables = cursor.fetchall()
    return [t[0] for t in tables]

  
def describe_table(table_name: str) -> list[tuple[str, str]]:
    """Look up the table schema.

    Returns:
      List of columns, where each entry is a tuple of (column, type).
    """
    print(f' - DB CALL: describe_table({table_name})')

    cursor = st.session_state["Chat_Bot_Data_Base"].cursor()

    cursor.execute(f"PRAGMA table_info({table_name});")

    schema = cursor.fetchall()
    # [column index, column name, column type, ...]
    return [(col[1], col[2]) for col in schema]

def execute_query(sql: str) -> list[list[str]]:
    """Execute an SQL statement, returning the results."""
    print(f' - DB CALL: execute_query({sql})')

    cursor = st.session_state["Chat_Bot_Data_Base"].cursor()

    cursor.execute(sql)
    return cursor.fetchall()



def describe_Data(sql_query: str):
    """   Given a SQL query that has a single column of a continuous variable, This will return the descriptive statistics for that variable"""
    
    df = pd.read_sql_query(sql_query, st.session_state["Chat_Bot_Data_Base"])
 
    #descriptive statistics on 'CenaUgoverEvro' and convert to dataframe
    descriptive_stats = df.iloc[:,0].describe().to_string()
    
    return descriptive_stats

def query_plotter(sql_query: str ): 
    """Given a SQL query that has a single column of a continuous variable, This function will create a plot"""

    df = pd.read_sql_query(sql_query, st.session_state["Chat_Bot_Data_Base"])
    st_fig, st_ax = plt.subplots(figsize=(8, 6))

    sns.histplot(df.iloc[:,0], kde=True , ax=st_ax)
    ##plt.show()
    st.session_state["Display_Plot"] = 1
    st.session_state["Chat_Bot_Plot"] = st_fig


def Build_Chat_Bot_And_Data():

    #### Building Data Set For the Model ####

    all_data_Chat_Bot  = None

    if st.session_state["SFR_Model_Predicitons_All"] is not None :
        if all_data_Chat_Bot is None: 
            all_data_Chat_Bot = st.session_state["SFR_Model_Predicitons_All"].copy() 
        else:    
            all_data_Chat_Bot = pd.concat([all_data_Chat_Bot, st.session_state["SFR_Model_Predicitons_All"]], ignore_index = True)    

    if st.session_state["Condo_Model_Predicitons_All"] is not None :
        if all_data_Chat_Bot is None: 
            all_data_Chat_Bot = st.session_state["Condo_Model_Predicitons_All"].copy() 
        else:
            all_data_Chat_Bot = pd.concat([all_data_Chat_Bot, st.session_state["Condo_Model_Predicitons_All"]], ignore_index = True)

    if st.session_state["Vac_Res_Model_Predicitons_All"] is not None :
        if all_data_Chat_Bot is None:
            all_data_Chat_Bot = st.session_state["Vac_Res_Model_Predicitons_All"].copy() 
        else:
            all_data_Chat_Bot = pd.concat([all_data_Chat_Bot, st.session_state["Vac_Res_Model_Predicitons_All"]], ignore_index = True)
    
    if st.session_state["Commercial_Model_Predicitons_All"] is not None :
        if all_data_Chat_Bot is None:
            all_data_Chat_Bot = st.session_state["Commercial_Model_Predicitons_All"].copy()
        else:
            all_data_Chat_Bot = pd.concat([all_data_Chat_Bot, st.session_state["Commercial_Model_Predicitons_All"]], ignore_index = True)

    Parcel_Data = all_data_Chat_Bot.copy()

    Parcel_Data = Parcel_Data.rename(columns = {'Model_Prediction' : 'Model_Value'})

    NAL_For_DB =  st.session_state["NAL"][['PARCEL_ID', 'DOR_UC', 'ACT_YR_BLT', 'EFF_YR_BLT', 'LND_SQFOOT', 'TOT_LVG_AREA', 'JV','NBRHD_CD', 'PHY_ADDR1','PHY_CITY','PHY_ZIPCD']].copy()


    NAL_For_DB = NAL_For_DB.rename(columns={'ACT_YR_BLT': 'Actual_Year_Built', 'EFF_YR_BLT': 'Effective_Year_Built', 'LND_SQFOOT': 'Land_Square_Footage', 'TOT_LVG_AREA': 'Total_Living_Area', 'JV': 'Just_Value' , 'DOR_UC': 'DOR_Code'
                                            , 'NBRHD_CD': 'Neighborhood_Code', 'PHY_ADDR1': 'Address','PHY_CITY': 'City','PHY_ZIPCD': 'ZIP_Code'} )
    

    Parcel_Data = pd.merge(NAL_For_DB
                            , Parcel_Data
                            , on='PARCEL_ID'
                            , how='inner')
    
    Parcel_Data['Model_Value'] = round(Parcel_Data['Model_Value'],0).astype('Int64')

    Parcel_Data.to_csv("C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\Chat_Bot_Data.csv"
                
                       , index = False)


    ## Creating Local Database from the data Created by the model 
    local_db_file = "C:\\Users\\tajor\\OneDrive\\Documents\\Python Applications\\Contracting\\NAL Project\\Data\\NAL_Navigator.db"

    # if os.path.exists(local_db_file):
    #     os.remove(local_db_file)
    #     print("Old Database Removed")
    # else: 
    #     print("No Old Database Found")

    st.session_state["Chat_Bot_Data_Base"] = sqlite3.connect(local_db_file
                                                            , check_same_thread = False)

    Parcel_Data.to_sql('Parcel_Data', st.session_state["Chat_Bot_Data_Base"], if_exists='replace')

    ## Setting up google API key 

    GOOGLE_API_KEY = st.session_state["Google_Key"]

    if  (st.session_state["Provided_Google_Key"] is not None) & (len(st.session_state['Provided_Google_Key'])  > 12  ) :
        GOOGLE_API_KEY = st.session_state["Provided_Google_Key"]

    ## 🔒 NEVER print the key itself. An earlier version did, which wrote a
    ## live credential to the terminal every time the chatbot started.
    ## To check a key arrived, report only its SHAPE:
    if GOOGLE_API_KEY:
        print(f'API key loaded: yes ({len(str(GOOGLE_API_KEY))} characters)')
    else:
        print('API key loaded: NO - none provided')

    client = genai.Client(api_key=GOOGLE_API_KEY)

    Model_Name = "gemini-2.0-flash"

    db_tools = [list_tables, describe_table, execute_query,describe_Data,query_plotter ]

    ## snarky
    instruction2 = """You are a helpful chatbot that can interact with a SQL database containing property data. You will take the users questions and turn them into SQL
    queries using the tools available. Once you have the information you need, you will
    answer the user's question using the data returned.


    The database contains the following tables:
        - Parcel_Data
            - PARCEL_ID
            - DOR_Code
            - Actual_Year_Built
            - Effective_Year_Built
            - Land_Square_Footage
            - Total_Living_Area
            - Just_Value
            - Model_Value

    The Parcel_ID is a unique identifier for each property.


    You can use the following tools:
        - list_tables
        - describe_table
        - execute_query
        - describe_Data
        - query_plotter
    Use list_tables to see what tables are present, describe_table to understand the
    schema, and execute_query to issue an SQL SELECT query.

    If you do not know the answer, you can say "I don't know"
    """

    instruction_Original = """You are a helpful chatbot that can interact with a SQL database containing property data. You will take the users questions and turn them into SQL
    queries using the tools available. Once you have the information you need, you will
    answer the user's question using the data returned.


    The database contains only one table named Parcel_Data with the following columns:
        
        - PARCEL_ID
        - DOR_Code
        - Actual_Year_Built
        - Effective_Year_Built
        - Land_Square_Footage
        - Total_Living_Area
        - Just_Value
        - Model_Value

    The Parcel_ID is a unique identifier for each property.


    You can use the following tools:
        - list_tables
        - describe_table
        - execute_query

    Use list_tables to see what tables are present, describe_table to understand the
    schema, and execute_query to issue an SQL SELECT query.

    If you do not know the answer, you can say "I don't know"
    """

    instruction = """You are a helpful chatbot that can interact with a SQL database containing property data. You will take the users questions and turn them into SQL
    queries using the tools available. Once you have the information you need, you will
    answer the user's question using the data returned.


    The database contains only one table named Parcel_Data with the following columns:
        
        - PARCEL_ID
        - DOR_Code
        - Actual_Year_Built
        - Effective_Year_Built
        - Land_Square_Footage
        - Total_Living_Area
        - Just_Value
        - Model_Value

    The Parcel_ID is a unique identifier for each property.


    You can use the following tools:
        - list_tables
        - describe_table
        - execute_query
        - describe_Data
        - query_plotter

    Use list_tables to see what tables are present, describe_table to understand the
    schema, and execute_query to issue an SQL SELECT query. The tool query_plotter can be used to plot a histogram of a single continuous variable when it is given a valid SQL query with one continuous variable. 
    The describe_Data tool will return the descriptive statistics as a string for a single continuous variable when it is given a valid SQL query contianing one continuous variable.

    When creating descriptive statistics you should format the output for the user in a way that is easy to read and provide a summary. If the user asks you to create a plot and to describe the plot 
    use the describe_Data function to create the descriptive statistics for the plot.

    If you do not know the answer, you can say "I don't know"
    """

    client = genai.Client(api_key = GOOGLE_API_KEY)

    # Start a chat with automatic function calling enabled.
    chat = client.chats.create(
                                    model = Model_Name,
                                    config = types.GenerateContentConfig(
                                                                            system_instruction = instruction,
                                                                            tools = db_tools,
                                                                        ),
                                )
    greeting_text = "Hi there! I am a custom LLM that is here to better help you understand your NAL data. Please feel free to ask me any appraisal questions, a great first question would be 'What columns are in the database?' this way I can let you know exactly what data I have access too!"
    with st.chat_message("assistant"):
        st.write(greeting_text)
    
    st.session_state["Chat_History"] = [{"role": "assistant", "content": greeting_text}]

    st.session_state["NAL_Chat_Bot"] = chat

    st.session_state['Chat_Launched'] = 'ready'

def Clear_Chat_History():
    st.session_state["Chat_History"] = []    

def response_generator( question ):
    st.session_state["Display_Plot"] = 0
    st.session_state["Display_Chart"] = 0
    response = st.session_state["NAL_Chat_Bot"].send_message(question)
    #response =  "Hi, human! Is there anything I can help you with?"

    print('The submitted question is:', question)


    
    for word in response.text.split(" "):
        yield word + " "
        time.sleep(0.05)

def Query_Check (): 
    print(str(execute_query('SELECT count(*) FROM Parcel_Data')))
    

######## APP Layout And Buttons ########


### Setting Title 
st.title("NAL :red[Navigator]" , anchor= "right")


## Show the login screen only if a credential was actually configured.
## With HGMA_APP_USER / HGMA_APP_PASSWORD unset, User_Password_List is empty
## and this is False, so the app opens straight to the data.
if User_Password_List and (st.session_state["PSWRD_Entry"] is None) and (st.session_state["User_Name_Entry"] is None):
    st.markdown('##')
    st.markdown('## Password :red[Entry]')
    st.write("---")
    usrname= st.text_input("User Name")
    pswrd =st.text_input("Password"
                        , type="password")

    st.button("Submit", on_click = Password_Entry , args=(usrname,pswrd))

    if st.session_state['Password_Count'] > 0:
        st.markdown('####')
        st.markdown(":warning:   :red[Password Entry Failed]  	:warning:")


else:

   
    #st.markdown('##')


    ## Building Sidebar
    exit_app = st.sidebar.button("Shut Down"
                                , type="primary")
    
    st.sidebar.markdown('##')
    st.sidebar.markdown('##')
    st.sidebar.markdown('##')


    st.sidebar.number_input("Valuation Year"
                            , value = 2025
                            ,key = 'Valuation_Year'
                            , on_change= Update_County_List_Loc
                            , step = 1)
    st.session_state['County_Selected'] = st.sidebar.selectbox("Select Your County"
                                                            , list(st.session_state['County_List_Loc'] .keys())
                                                            , index = None
                                                            , placeholder = "Select County"
                                                        ##    , on_change = set_nal_data
                                                            )
    

    st.session_state['State_Coords_Selected'] = st.sidebar.selectbox("Select Your Coordinate System"
                                                                    , st.session_state['EPSG_List']
                                                                    , index = None
                                                                    , placeholder = "Select EPSG"
                                                                ##    , on_change = set_nal_data
                                                                    )
                                                            

    st.sidebar.markdown('##')
    st.sidebar.markdown('##')

    st.session_state['Cost_Of_Sale_Adjustment_Toggle'] = st.sidebar.toggle("Update Cost of Sale Adjustments")

    st.sidebar.markdown('##')
    st.sidebar.markdown('##')

    st.session_state['Model_Performance_Toggle'] = st.sidebar.toggle("Display Model Performance Graphs")
    st.sidebar.markdown('##')
    st.sidebar.markdown('##')
    st.session_state['DOR_Stats_Toggle'] = st.sidebar.toggle("Display DOR Summary Statistics")





    st.markdown('##')
    st.markdown('## Data Uploading')
    st.write("---")


    ##Loading NAL File

    NAL_Files = st.file_uploader("## Upload NAL file"
                                , type=["csv"]) 
    
    
    if NAL_Files is not None:
    # st.markdown(":green[NAL File Loaded!]")
        if st.session_state["NAL"] is None:
            with st.spinner("One second just a little tidying up to do!"):
                st.session_state["NAL"]  = pd.read_csv(NAL_Files
                                                    , low_memory= False)
        #print(type(NAL))
        st.markdown(""":green[ Your NAL Loaded Succesfully! There are """ + str(st.session_state["NAL"].shape[0]) + """ rows of data in your NAL File ]""")

    if st.session_state['County_Selected'] is not None and st.session_state["NAL"] is None:
        st.markdown( f""":orange[ By Selecting your county we will use the last year NAL file for {st.session_state['County_Selected']} if you do not upload a new NAL file, for the most accurate results please upload a new NAL file ]""")

        

    st.markdown('#####')
    SDF_Files = st.file_uploader("## Upload SDF file"
                                , type=["csv"]) 

    if SDF_Files is not None:
        if st.session_state["SDF"] is None:
            with st.spinner("One second just a little tidying up to do!"):
                st.session_state["SDF"]  = pd.read_csv(SDF_Files
                                                        , low_memory= False)
        
        st.markdown(""":green[ Your SDF Loaded Succesfully! There are """ + str(st.session_state["SDF"].shape[0]) + """ rows of data in your SDF File ]""")


    st.markdown('#####')    
    # GIS_Files = st.file_uploader("## Upload GIS file", type=["shp"]) 

    GIS_File_Path = st.text_input("## Enter Path to GIS file"
                                , placeholder = "Enter Path to GIS file")
    
    if st.button("Load GIS File"):
        with st.status("Loading your GIS File... This may take a few seconds", expanded=True) as status_message:
            GIS_File_Path_clean = GIS_File_Path.replace("/", '\\')
            st.session_state["GIS"] = gpd.read_file(GIS_File_Path_clean)
            status_message.update(label="Wow that was tougher than I thought but I am done!", state="complete", expanded=False)

    if st.session_state["GIS"] is not None:
        st.markdown(""":green[GIS File Loaded! There are """ + str(st.session_state["GIS"].shape[0]) + """ rows of data in your GIS File ]""")

    # if GIS_Files is not None:
    #     if st.session_state["GIS"] is None:
    #         with st.spinner("One second just a little tidying up to do!"):
    #             st.session_state["GIS"] = gpd.read_file(GIS_Files)

    #    st.markdown(":green[GIS File Loaded!]")
    st.markdown('#####')

    if st.session_state['Cost_Of_Sale_Adjustment_Toggle'] == True:
        st.markdown('##')
        st.markdown('## Set DR 493 Adjustments')
        st.write("---")

        st.selectbox("Select DOR Code to Adjust"
                        , list(st.session_state["DOR_COS_ADJ_DICT"].keys())
                        , index = 1
                        , key = 'current_Adjust_dor'
                        , placeholder = "Select DOR Code"
                        , on_change = update_COS_Input
                        # , args = ('01',)
                        
                        )

        st.markdown('##')

        new_COS_ADJ =    st.number_input("Set Adjustments"
                            , key = "COS_ADJ_KEY"
                            , min_value = 0.0
                            , max_value = 2.0
                            , step = .01)                                                    

        st.markdown('##')

        if(st.button("Update the DR 493 Adjustemt")):
            with st.spinner("Applying the Update!"):
                print('The New COS adjustment is ' + str(new_COS_ADJ) + ' for ' + st.session_state['current_Adjust_dor'] )
                st.session_state["DOR_COS_ADJ_DICT"][st.session_state['current_Adjust_dor'] ] = new_COS_ADJ













        st.markdown('#####')


    if (st.session_state["SDF"]  is not None) and ((st.session_state["NAL"] is not None or st.session_state['County_Selected'] is not None   ) ):
        st.markdown('##')
        st.markdown('## Data Pre-Processing')
        st.write("---")
        #st.markdown('##')

        left, middle = st.columns([.3 , .7])

        if (left.button("Merge and Preprocess My Files", use_container_width=True)) :
            with st.spinner("Hang on tight as I glue all your data together!"):
                st.session_state["NAL_merged"] = Merge_data(st.session_state["NAL"] , st.session_state["SDF"].copy() , st.session_state["GIS"] ).copy()

            st.session_state["preprocesing_ran"]  = True

        if (st.session_state["preprocesing_ran"] == True):
            middle.markdown(""":green[ Your Data has been merged and Pre Processed View the results Below!]""")

        if st.session_state["NAL_merged"] is not None:
            st.markdown('##')
            with st.expander("View Merged Data"):
                st.dataframe(st.session_state["NAL_merged"].head(20))


        if st.session_state["preprocesing_ran"] == True:
            with st.form(" Build Model Form" , border= False,  clear_on_submit= False):
                st.markdown('##')
                st.markdown('## Build Machine Learning Model')
                st.write("---")

                left2, middle2, right2 = st.columns(3)
                left2.markdown('What Type of Model would you like to build?')

                if middle2.checkbox("Single Family Residential Improved Model"):

                    st.session_state["SFR_model_binary"] = True
                    #SFR_model_binary = True

                if middle2.checkbox("Vacant Residential Model"):
                    #Vac_Res_model_binary = True
                    st.session_state["Vac_Res_model_binary"] = True    

                if right2.checkbox("Condominium Model"):
                    #Condo_model_binary = True
                    st.session_state["Condo_model_binary"] = True  

                right2.markdown('')     

                if right2.checkbox("Commercial Model"):
                    #Commercial_model_binary = True
                    st.session_state["Commercial_model_binary"] = True  



                st.markdown('##')

                with st.expander("Advanced Model Settings"):
                    st.markdown('##')
                    st.checkbox("Run Hyperparameter Tuning"
                                , key = "Hyperparameter_Tuning_Toggle")
                    st.number_input("Number of Trials"
                                    , min_value = 10
                                    , max_value = 400
                                    , value = 150
                                    , step = 1
                                    , key = "Hyperparameter_Tuning_Trials")
                    Model_Expand_col1, Model_Expand_col2 = st.columns(2)
                    
                    with Model_Expand_col1:
                        st.number_input("Minimum Number of Estiamtors"
                                        ,value = 250
                                        , key = "HyperParametet_Tuning_N_Estimators_Low")
                        st.number_input("Minimum Number of Leaves"
                                        ,value = 10
                                        , key = "HyperParametet_Tuning_num_leaves_Low")
                        st.number_input("Minimum Learning Rate"
                                        ,value = .01
                                        , key = "HyperParametet_Tuning_learning_rate_Low")
                        st.number_input("Minimum Subsample"
                                        ,value = .66
                                        , key = "HyperParametet_Tuning_subsample_Low")
                        st.number_input("Minimum Feature Fraction"
                                        ,value = .7
                                        , key = "HyperParametet_Tuning_feature_fraction_Low")
                        st.number_input("Minimum Max Depth"
                                        ,value = 3
                                        , key = "HyperParametet_Tuning_max_depth_Low")
                        st.number_input("Minimum Min Data in Leaf"
                                        ,value = 3
                                        , key = "HyperParametet_Tuning_min_data_in_leaf_Low")
                        st.number_input("Minimum Min Sum Hessian in Leaf"
                                        ,value = .1
                                        , key = "HyperParametet_Tuning_min_sum_hessian_in_leaf_Low")
                        

                    with Model_Expand_col2:
                        st.number_input("Maximum Number of Estiamtors"
                                        ,value = 1500
                                        , key = "HyperParametet_Tuning_N_Estimators_High")
                        st.number_input("Maximum Number of Leaves"
                                        ,value = 100
                                        , key = "HyperParametet_Tuning_num_leaves_High")
                        st.number_input("Maximum Learning Rate"
                                        ,value = .3
                                        , key = "HyperParametet_Tuning_learning_rate_High")
                        st.number_input("Maximum Subsample"
                                        ,value = 1
                                        , key = "HyperParametet_Tuning_subsample_High")
                        st.number_input("Maximum Feature Fraction"
                                        ,value = 1
                                        , key = "HyperParametet_Tuning_feature_fraction_High")
                        st.number_input("Maximum Max Depth"
                                        ,value = 20
                                        , key = "HyperParametet_Tuning_max_depth_High")
                        st.number_input("Maximum Min Data in Leaf"
                                        ,value = 30
                                        , key = "HyperParametet_Tuning_min_data_in_leaf_High")
                        st.number_input("Maximum Min Sum Hessian in Leaf"
                                        ,value = 3
                                        , key = "HyperParametet_Tuning_min_sum_hessian_in_leaf_High")
                        

                    st.markdown('##')

                st.markdown('##')

                build_model_button = st.form_submit_button("Build Model")

                if build_model_button:

                    response_container = st.empty()                                    
                    with response_container.container():
                        progress_bar_cols = st.columns([2,5,2])                                
                        with progress_bar_cols[1]:                                                                            
                            my_bar = st.progress(0, text='')
                            try:
                                render_animation() 
                            except:
                                st.spinner()     
                            time.sleep(1)
                            Build_ML_Regressors(my_bar)
                            


                        # for percent_complete in range(100):
                        #     time.sleep(0.05)
                            # my_bar.progress(percent_complete + 1, text='**Please hang on tight we are building something cool together!**')



                            
                    with response_container.container():
                        st.success(f'✅  We have successfully built some really cool models!  :tada:')  
                        st.session_state["Model_ran"] = True

            if st.session_state["Model_ran"] == True:
                # with st.spinner('Wait for it....'):
                #     time.sleep(5)
                st.markdown('##')
                st.markdown('## Model(s) Performance')
                st.write("---")

                if st.session_state["sfr_preformance_dict"] is not None: 

                    st.markdown('###')
                    st.markdown("### :house: SFR Model Performance:")

                    sfr_left, sfr_middle = st.columns([.6 , .4])

                    sfr_left.markdown("**Test Data Size Before Trimming**:")
                    sfr_middle.markdown(str(st.session_state["sfr_preformance_dict"]["Number of Records Before Trimming Test"]))
                    sfr_left.markdown("**Test Data Size After Trimming**:")
                    sfr_middle.markdown(str(st.session_state["sfr_preformance_dict"]["Number of Records After Trimming Test"]))

                    st.markdown('####')

                    sfr_radio_left,sfr_radio_right = st.columns([.5 , .5])

                    train_stats_type = sfr_radio_left.radio('**Training Model Results are Calculated on :**'
                                                            , ["Un-Trimmed Data" ,'Trimmed Data']
                                                            , index  = 0 )
                    
                    
                    test_stats_type = sfr_radio_right.radio('**Test Model Results are Calculated on :**'
                                        , ["Un-Trimmed Data" ,'Trimmed Data']
                                        , index = 1)            


                    st.markdown('#####')

                    sfr_m_left , sfr_m_middle, sfr_m_right = st.columns([.5 , .25 , .25])

                    sfr_m_left.markdown("**Model Metric**")
                    sfr_m_middle.markdown("**Training Pefrormance**")
                    sfr_m_right.markdown("**Test Pefrormance**")
                    sfr_m_left.markdown('')
                    sfr_m_middle.markdown('')
                    sfr_m_right.markdown('')

                    train_stat_r_square = st.session_state["sfr_preformance_dict"]["Train R Squared"]
                    train_stat_weighted_mean_sale_ratio = st.session_state["sfr_preformance_dict"]["Train Weighted Mean"]
                    train_stat_COD = st.session_state["sfr_preformance_dict"]["Train COD"]
                    train_stat_PRD = st.session_state["sfr_preformance_dict"]["Train PRD"]
                    train_stat_MAE = st.session_state["sfr_preformance_dict"]["Train MAE"]

                    if train_stats_type == "Trimmed Data":
                        train_stat_r_square = st.session_state["sfr_preformance_dict"]["Trimmed Train R Squared"]
                        train_stat_weighted_mean_sale_ratio = st.session_state["sfr_preformance_dict"]["Trimmed Train Weighted Mean"]
                        train_stat_COD = st.session_state["sfr_preformance_dict"]["Trimmed Train COD"]
                        train_stat_PRD = st.session_state["sfr_preformance_dict"]["Trimmed Train PRD"]
                        train_stat_MAE = st.session_state["sfr_preformance_dict"]["Trimmed Train MAE"]


                    test_stat_r_square = st.session_state["sfr_preformance_dict"]["Test R Squared"]
                    test_stat_weighted_mean_sale_ratio = st.session_state["sfr_preformance_dict"]["Test Weighted Mean"]
                    test_stat_COD = st.session_state["sfr_preformance_dict"]["Test COD"]                
                    test_stat_PRD = st.session_state["sfr_preformance_dict"]["Test PRD"]
                    test_stat_MAE = st.session_state["sfr_preformance_dict"]["Test MAE"]

                    if test_stats_type == "Trimmed Data":
                        test_stat_r_square = st.session_state["sfr_preformance_dict"]["Trimmed Test R Squared"]
                        test_stat_weighted_mean_sale_ratio = st.session_state["sfr_preformance_dict"]["Trimmed Test Weighted Mean"] 
                        test_stat_COD = st.session_state["sfr_preformance_dict"]["Trimmed Test COD"]
                        test_stat_PRD = st.session_state["sfr_preformance_dict"]["Trimmed Test PRD"]
                        test_stat_MAE = st.session_state["sfr_preformance_dict"]["Trimmed Test MAE"]


                    #test_stat_r_square = st.session_state["sfr_preformance_dict"]["Trimmed Test R Squared"]



                    sfr_m_left.markdown("**R Squared**")
                    sfr_perf_color = 'green' if train_stat_r_square > 0.8 else 'orange'
                    if train_stat_r_square < .5 :
                        sfr_perf_color = 'red'
                    sfr_m_middle.markdown( ":" + sfr_perf_color + "["  + str(train_stat_r_square ) + "]"   )
                    sfr_perf_color = 'green' if test_stat_r_square> 0.8 else 'orange'
                    if test_stat_r_square < .5 :
                        sfr_perf_color = 'red'
                    sfr_m_right.markdown( ":" + sfr_perf_color + "["  + str(test_stat_r_square) + "]"   )

                    sfr_m_left.markdown("**Weighted Mean Sale Ratio**")
                    sfr_perf_color = 'green' if abs(train_stat_weighted_mean_sale_ratio - 1 ) <= 0.03 else 'orange'
                    if abs(train_stat_weighted_mean_sale_ratio - 1 ) > 0.05 :
                        sfr_perf_color = 'red'
                    sfr_m_middle.markdown( ":" + sfr_perf_color + "["  + str(train_stat_weighted_mean_sale_ratio) + "]"   )
                    sfr_perf_color = 'green' if abs(test_stat_weighted_mean_sale_ratio - 1 ) <= 0.03 else 'orange'
                    if abs(test_stat_weighted_mean_sale_ratio - 1 ) > 0.05 :
                        sfr_perf_color = 'red'
                    sfr_m_right.markdown( ":" + sfr_perf_color + "["  + str(test_stat_weighted_mean_sale_ratio) + "]"   )

                    sfr_m_left.markdown("**COD**")
                    sfr_perf_color = 'green' if train_stat_COD < 10 else 'orange'
                    if train_stat_COD > 15 :
                        sfr_perf_color = 'red'
                    sfr_m_middle.markdown( ":" + sfr_perf_color + "["  + str(train_stat_COD) + "]"   )
                    sfr_perf_color = 'green' if test_stat_COD < 10 else 'orange'
                    if test_stat_COD > 15 :
                        sfr_perf_color = 'red'
                    sfr_m_right.markdown( ":" + sfr_perf_color + "["  + str(test_stat_COD) + "]"   )

                    sfr_m_left.markdown("**PRD**")
                    sfr_perf_color = 'green' if (train_stat_PRD   > .97 ) and (train_stat_PRD   > 1.02 ) else 'orange'
                    if abs(train_stat_PRD  - 1 ) > .04 :
                        sfr_perf_color = 'red'
                    sfr_m_middle.markdown( ":" + sfr_perf_color + "["  + str(train_stat_PRD ) + "]"   )
                    sfr_perf_color = 'green' if (test_stat_PRD < .97) and (test_stat_PRD > 1.02)  else 'orange'
                    if abs(test_stat_PRD - 1 )  > 0.04 :
                        sfr_perf_color = 'red'
                    sfr_m_right.markdown( ":" + sfr_perf_color + "["  + str(test_stat_PRD) + "]"   )

                    sfr_m_left.markdown("**Mean Absolute Error**")
            
                    sfr_m_middle.markdown(  str(round(train_stat_MAE,0))  )
        
                    sfr_m_right.markdown( str(round(test_stat_MAE,0)) )


                    if  st.session_state['Model_Performance_Toggle']:
                        st.markdown('#####')
                        st.markdown('#####')

                        st.markdown('#### Graphs to compare Model Performance to NAL Values On the Trimmed Test Set (SFR Model)')

                        graph_metric = st.radio(
                                                "## Model Metric:"
                                                ,['R squared' , 'Weighted Mean Sale Ratio' , 'COD' , 'PRD' , 'Mean Absolute Error']
                                                , horizontal = True
                                                , index = 0
                                            )                    


                        if graph_metric == 'R squared':  
                            
                            graph_data = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["sfr_preformance_dict"]["Trimmed Test R Squared"] , st.session_state["sfr_preformance_dict"]["Trimmed Test NAL R Squared"]] }
                            graph_lab = 'R Squared'

                        elif graph_metric == 'Weighted Mean Sale Ratio':
                            graph_data = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["sfr_preformance_dict"]["Trimmed Test Weighted Mean"] , st.session_state["sfr_preformance_dict"]["Trimmed Test NAL Weighted Mean"]] }
                            graph_lab = 'Weighted Mean Sale Ratio'

                        elif graph_metric == 'COD':
                            graph_data = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["sfr_preformance_dict"]["Trimmed Test COD"] , st.session_state["sfr_preformance_dict"]["Trimmed Test NAL COD"]] }
                            graph_lab = 'COD'

                        elif graph_metric == 'PRD':
                            graph_data = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["sfr_preformance_dict"]["Trimmed Test PRD"] , st.session_state["sfr_preformance_dict"]["Trimmed Test NAL PRD"]] }
                            graph_lab = 'PRD'    

                        elif graph_metric == 'Mean Absolute Error':     
                            graph_data = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["sfr_preformance_dict"]["Trimmed Test MAE"] , st.session_state["sfr_preformance_dict"]["Trimmed Test NAL MAE"]] }
                            graph_lab = 'Mean Absolute Error'


                        st.markdown('#####')

                        
                        st.bar_chart( pd.DataFrame(graph_data)
                                    , x = 'Value Source'
                                    
                                    , y = 'Metric'
                                    
                                    , y_label= graph_lab
                                    
                                    , horizontal = True

                    
                                    , height = 200
                                    , width= 150
                                    )

                    

                if st.session_state["Condo_preformance_dict"] is not None: 

                    st.markdown('###')
                    st.markdown("### :office: Condo Model Performance:")

                    Condo_left, Condo_middle = st.columns([.6 , .4])

                    Condo_left.markdown("**Test Data Size Before Trimming**:")
                    Condo_middle.markdown(str(st.session_state["Condo_preformance_dict"]["Number of Records Before Trimming Test"]))
                    Condo_left.markdown("**Test Data Size After Trimming**:")
                    Condo_middle.markdown(str(st.session_state["Condo_preformance_dict"]["Number of Records After Trimming Test"]))

                    st.markdown('####')

                    Condo_radio_left, Condo_radio_right = st.columns([.5 , .5])

                    train_stats_type_Condo = Condo_radio_left.radio('**Training Model Results are Calculated on :**  '
                                                            , ["Un-Trimmed Data" ,'Trimmed Data']
                                                            , index  = 0 )
                    
                    
                    test_stats_type_Condo = Condo_radio_right.radio('**Test Model Results are Calculated on :**  '
                                                                    , ["Un-Trimmed Data" ,'Trimmed Data']
                                                                    , index = 1)            


                    st.markdown('#####')


                    Condo_m_left , Condo_m_middle, Condo_m_right = st.columns([.5 , .25 , .25])

                    Condo_m_left.markdown("**Model Metric**")
                    Condo_m_middle.markdown("**Training Pefrormance**")
                    Condo_m_right.markdown("**Test Pefrormance**")
                    Condo_m_left.markdown('')
                    Condo_m_middle.markdown('')
                    Condo_m_right.markdown('')

                    Condo_train_stat_r_square = st.session_state["Condo_preformance_dict"]["Train R Squared"]
                    Condo_train_stat_weighted_mean_sale_ratio = st.session_state["Condo_preformance_dict"]["Train Weighted Mean"]
                    Condo_train_stat_COD = st.session_state["Condo_preformance_dict"]["Train COD"]
                    Condo_train_stat_PRD = st.session_state["Condo_preformance_dict"]["Train PRD"]
                    Condo_train_stat_MAE = st.session_state["Condo_preformance_dict"]["Train MAE"]

                    if train_stats_type_Condo == "Trimmed Data":
                        Condo_train_stat_r_square = st.session_state["Condo_preformance_dict"]["Trimmed Train R Squared"]
                        Condo_train_stat_weighted_mean_sale_ratio = st.session_state["Condo_preformance_dict"]["Trimmed Train Weighted Mean"]
                        Condo_train_stat_COD = st.session_state["Condo_preformance_dict"]["Trimmed Train COD"]
                        Condo_train_stat_PRD = st.session_state["Condo_preformance_dict"]["Trimmed Train PRD"]
                        Condo_train_stat_MAE = st.session_state["Condo_preformance_dict"]["Trimmed Train MAE"]


                    Condo_test_stat_r_square = st.session_state["Condo_preformance_dict"]["Test R Squared"]
                    Condo_test_stat_weighted_mean_sale_ratio = st.session_state["Condo_preformance_dict"]["Test Weighted Mean"]
                    Condo_test_stat_COD = st.session_state["Condo_preformance_dict"]["Test COD"]                
                    Condo_test_stat_PRD = st.session_state["Condo_preformance_dict"]["Test PRD"]
                    Condo_test_stat_MAE = st.session_state["Condo_preformance_dict"]["Test MAE"]

                    if test_stats_type_Condo == "Trimmed Data":
                        Condo_test_stat_r_square = st.session_state["Condo_preformance_dict"]["Trimmed Test R Squared"]
                        Condo_test_stat_weighted_mean_sale_ratio = st.session_state["Condo_preformance_dict"]["Trimmed Test Weighted Mean"] 
                        Condo_test_stat_COD = st.session_state["Condo_preformance_dict"]["Trimmed Test COD"]
                        Condo_test_stat_PRD = st.session_state["Condo_preformance_dict"]["Trimmed Test PRD"]
                        Condo_test_stat_MAE = st.session_state["Condo_preformance_dict"]["Trimmed Test MAE"]
                    

                    Condo_m_left.markdown("**R Squared**")
                    Condo_perf_color = 'green' if Condo_train_stat_r_square > 0.8 else 'orange'
                    if Condo_train_stat_r_square < .5 :
                        Condo_perf_color = 'red'
                    Condo_m_middle.markdown( ":" + Condo_perf_color + "["  + str(Condo_train_stat_r_square ) + "]"   )
                    Condo_perf_color = 'green' if Condo_test_stat_r_square> 0.8 else 'orange'
                    if Condo_test_stat_r_square < .5 :
                        Condo_perf_color = 'red'
                    Condo_m_right.markdown( ":" + Condo_perf_color + "["  + str(Condo_test_stat_r_square) + "]"   )

                    Condo_m_left.markdown("**Weighted Mean Sale Ratio**")
                    Condo_perf_color = 'green' if abs(Condo_train_stat_weighted_mean_sale_ratio - 1 ) <= 0.03 else 'orange'
                    if abs(Condo_train_stat_weighted_mean_sale_ratio - 1 ) > 0.05 :
                        Condo_perf_color = 'red'
                    Condo_m_middle.markdown( ":" + Condo_perf_color + "["  + str(Condo_train_stat_weighted_mean_sale_ratio) + "]"   )
                    Condo_perf_color = 'green' if abs(Condo_test_stat_weighted_mean_sale_ratio - 1 ) <= 0.03 else 'orange'
                    if abs(Condo_test_stat_weighted_mean_sale_ratio - 1 ) > 0.05 :
                        Condo_perf_color = 'red'
                    Condo_m_right.markdown( ":" + Condo_perf_color + "["  + str(Condo_test_stat_weighted_mean_sale_ratio) + "]"   )

                    Condo_m_left.markdown("**COD**")
                    Condo_perf_color = 'green' if Condo_train_stat_COD < 10 else 'orange'
                    if Condo_train_stat_COD > 15 :
                        Condo_perf_color = 'red'
                    Condo_m_middle.markdown( ":" + Condo_perf_color + "["  + str(Condo_train_stat_COD) + "]"   )
                    Condo_perf_color = 'green' if Condo_test_stat_COD < 10 else 'orange'
                    if Condo_test_stat_COD > 15 :
                        Condo_perf_color = 'red'
                    Condo_m_right.markdown( ":" + Condo_perf_color + "["  + str(Condo_test_stat_COD) + "]"   )

                    Condo_m_left.markdown("**PRD**")
                    Condo_perf_color = 'green' if (Condo_train_stat_PRD   > .97 ) and (Condo_train_stat_PRD   > 1.02 ) else 'orange'
                    if abs(Condo_train_stat_PRD  - 1 ) > .04 :
                        Condo_perf_color = 'red'
                    Condo_m_middle.markdown( ":" + Condo_perf_color + "["  + str(Condo_train_stat_PRD ) + "]"   )
                    Condo_perf_color = 'green' if (Condo_test_stat_PRD < .97) and (Condo_test_stat_PRD > 1.02)  else 'orange'
                    if abs(Condo_test_stat_PRD - 1 )  > 0.04 :
                        Condo_perf_color = 'red'
                    Condo_m_right.markdown( ":" + Condo_perf_color + "["  + str(Condo_test_stat_PRD) + "]"   )

                    Condo_m_left.markdown("**Mean Absolute Error**")
            
                    Condo_m_middle.markdown(  str(round(Condo_train_stat_MAE,0))  )
        
                    Condo_m_right.markdown( str(round(Condo_test_stat_MAE,0)) )


                    if  st.session_state['Model_Performance_Toggle']:
                        st.markdown('#####')
                        st.markdown('#####')

                        st.markdown('#### Graphs to compare Model Performance to NAL Values On the Trimmed Test Set (Condo Model)')

                        graph_metric_con = st.radio(
                                                "## Model Metric:"
                                                ,['R squared ' , 'Weighted Mean Sale Ratio ' , 'COD ' , 'PRD ' , 'Mean Absolute Error ']
                                                , horizontal = True
                                                , index = 0
                                            )                    


                        if graph_metric_con == 'R squared ':  
                            
                            graph_data_con = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Condo_preformance_dict"]["Trimmed Test R Squared"] , st.session_state["Condo_preformance_dict"]["Trimmed Test NAL R Squared"]] }
                            graph_lab_con = 'R Squared'

                        elif graph_metric_con == 'Weighted Mean Sale Ratio ':
                            graph_data_con = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Condo_preformance_dict"]["Trimmed Test Weighted Mean"] , st.session_state["Condo_preformance_dict"]["Trimmed Test NAL Weighted Mean"]] }
                            graph_lab_con = 'Weighted Mean Sale Ratio'

                        elif graph_metric_con == 'COD ':
                            graph_data_con = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Condo_preformance_dict"]["Trimmed Test COD"] , st.session_state["Condo_preformance_dict"]["Trimmed Test NAL COD"]] }
                            graph_lab_con = 'COD'

                        elif graph_metric_con == 'PRD ':
                            graph_data_con = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Condo_preformance_dict"]["Trimmed Test PRD"] , st.session_state["Condo_preformance_dict"]["Trimmed Test NAL PRD"]] }
                            graph_lab_con = 'PRD'    

                        elif graph_metric_con == 'Mean Absolute Error ':     
                            graph_data_con = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Condo_preformance_dict"]["Trimmed Test MAE"] , st.session_state["Condo_preformance_dict"]["Trimmed Test NAL MAE"]] }
                            graph_lab_con = 'Mean Absolute Error'


                        st.markdown('#####')

                        
                        st.bar_chart( pd.DataFrame(graph_data_con)
                                    , x = 'Value Source'
                                    
                                    , y = 'Metric'
                                    
                                    , y_label= graph_lab_con
                                    
                                    , horizontal = True

                    
                                    , height = 200
                                    , width= 150
                                    )




                if st.session_state["Vac_Res_preformance_dict"] is not None: 

                    st.markdown('###')
                    st.markdown("### :desert_island: Vacant Residential Model Performance:")

                    VR_left, VR_middle = st.columns([.6 , .4])

                    VR_left.markdown("**Test Data Size Before Trimming**:")
                    VR_middle.markdown(str(st.session_state["Vac_Res_preformance_dict"]["Number of Records Before Trimming Test"]))
                    VR_left.markdown("**Test Data Size After Trimming**:")
                    VR_middle.markdown(str(st.session_state["Vac_Res_preformance_dict"]["Number of Records After Trimming Test"]))

                    st.markdown('####')

                    VR_radio_left, VR_radio_right = st.columns([.5 , .5])

                    train_stats_type_VR = VR_radio_left.radio('**Training Model Results are Calculated on :** '
                                                            , ["Un-Trimmed Data" ,'Trimmed Data']
                                                            , index  = 0 )
                    
                    
                    test_stats_type_VR = VR_radio_right.radio('**Test Model Results are Calculated on :** '
                                        , ["Un-Trimmed Data" ,'Trimmed Data']
                                        , index = 1)            


                    st.markdown('#####')


                    VR_m_left , VR_m_middle, VR_m_right = st.columns([.5 , .25 , .25])

                    VR_m_left.markdown("**Model Metric**")
                    VR_m_middle.markdown("**Training Pefrormance**")
                    VR_m_right.markdown("**Test Pefrormance**")
                    VR_m_left.markdown('')
                    VR_m_middle.markdown('')
                    VR_m_right.markdown('')

                    VR_train_stat_r_square = st.session_state["Vac_Res_preformance_dict"]["Train R Squared"]
                    VR_train_stat_weighted_mean_sale_ratio = st.session_state["Vac_Res_preformance_dict"]["Train Weighted Mean"]
                    VR_train_stat_COD = st.session_state["Vac_Res_preformance_dict"]["Train COD"]
                    VR_train_stat_PRD = st.session_state["Vac_Res_preformance_dict"]["Train PRD"]
                    VR_train_stat_MAE = st.session_state["Vac_Res_preformance_dict"]["Train MAE"]

                    if train_stats_type_VR == "Trimmed Data":
                        VR_train_stat_r_square = st.session_state["Vac_Res_preformance_dict"]["Trimmed Train R Squared"]
                        VR_train_stat_weighted_mean_sale_ratio = st.session_state["Vac_Res_preformance_dict"]["Trimmed Train Weighted Mean"]
                        VR_train_stat_COD = st.session_state["Vac_Res_preformance_dict"]["Trimmed Train COD"]
                        VR_train_stat_PRD = st.session_state["Vac_Res_preformance_dict"]["Trimmed Train PRD"]
                        VR_train_stat_MAE = st.session_state["Vac_Res_preformance_dict"]["Trimmed Train MAE"]


                    VR_test_stat_r_square = st.session_state["Vac_Res_preformance_dict"]["Test R Squared"]
                    VR_test_stat_weighted_mean_sale_ratio = st.session_state["Vac_Res_preformance_dict"]["Test Weighted Mean"]
                    VR_test_stat_COD = st.session_state["Vac_Res_preformance_dict"]["Test COD"]                
                    VR_test_stat_PRD = st.session_state["Vac_Res_preformance_dict"]["Test PRD"]
                    VR_test_stat_MAE = st.session_state["Vac_Res_preformance_dict"]["Test MAE"]

                    if test_stats_type_VR == "Trimmed Data":
                        VR_test_stat_r_square = st.session_state["Vac_Res_preformance_dict"]["Trimmed Test R Squared"]
                        VR_test_stat_weighted_mean_sale_ratio = st.session_state["Vac_Res_preformance_dict"]["Trimmed Test Weighted Mean"] 
                        VR_test_stat_COD = st.session_state["Vac_Res_preformance_dict"]["Trimmed Test COD"]
                        VR_test_stat_PRD = st.session_state["Vac_Res_preformance_dict"]["Trimmed Test PRD"]
                        VR_test_stat_MAE = st.session_state["Vac_Res_preformance_dict"]["Trimmed Test MAE"]
                    

                    VR_m_left.markdown("**R Squared**")
                    VR_perf_color = 'green' if VR_train_stat_r_square > 0.8 else 'orange'
                    if VR_train_stat_r_square < .5 :
                        VR_perf_color = 'red'
                    VR_m_middle.markdown( ":" + VR_perf_color + "["  + str(VR_train_stat_r_square ) + "]"   )
                    VR_perf_color = 'green' if VR_test_stat_r_square> 0.8 else 'orange'
                    if VR_test_stat_r_square < .5 :
                        VR_perf_color = 'red'
                    VR_m_right.markdown( ":" + VR_perf_color + "["  + str(VR_test_stat_r_square) + "]"   )

                    VR_m_left.markdown("**Weighted Mean Sale Ratio**")
                    VR_perf_color = 'green' if abs(VR_train_stat_weighted_mean_sale_ratio - 1 ) <= 0.03 else 'orange'
                    if abs(VR_train_stat_weighted_mean_sale_ratio - 1 ) > 0.05 :
                        VR_perf_color = 'red'
                    VR_m_middle.markdown( ":" + VR_perf_color + "["  + str(VR_train_stat_weighted_mean_sale_ratio) + "]"   )
                    VR_perf_color = 'green' if abs(VR_test_stat_weighted_mean_sale_ratio - 1 ) <= 0.03 else 'orange'
                    if abs(VR_test_stat_weighted_mean_sale_ratio - 1 ) > 0.05 :
                        VR_perf_color = 'red'
                    VR_m_right.markdown( ":" + VR_perf_color + "["  + str(VR_test_stat_weighted_mean_sale_ratio) + "]"   )

                    VR_m_left.markdown("**COD**")
                    VR_perf_color = 'green' if VR_train_stat_COD < 10 else 'orange'
                    if VR_train_stat_COD > 15 :
                        VR_perf_color = 'red'
                    VR_m_middle.markdown( ":" + VR_perf_color + "["  + str(VR_train_stat_COD) + "]"   )
                    VR_perf_color = 'green' if VR_test_stat_COD < 10 else 'orange'
                    if VR_test_stat_COD > 15 :
                        VR_perf_color = 'red'
                    VR_m_right.markdown( ":" + VR_perf_color + "["  + str(VR_test_stat_COD) + "]"   )

                    VR_m_left.markdown("**PRD**")
                    VR_perf_color = 'green' if (VR_train_stat_PRD   > .97 ) and (VR_train_stat_PRD   > 1.02 ) else 'orange'
                    if abs(VR_train_stat_PRD  - 1 ) > .04 :
                        VR_perf_color = 'red'
                    VR_m_middle.markdown( ":" + VR_perf_color + "["  + str(VR_train_stat_PRD ) + "]"   )
                    VR_perf_color = 'green' if (VR_test_stat_PRD < .97) and (VR_test_stat_PRD > 1.02)  else 'orange'
                    if abs(VR_test_stat_PRD - 1 )  > 0.04 :
                        VR_perf_color = 'red'
                    VR_m_right.markdown( ":" + VR_perf_color + "["  + str(VR_test_stat_PRD) + "]"   )

                    VR_m_left.markdown("**Mean Absolute Error**")
            
                    VR_m_middle.markdown(  str(round(VR_train_stat_MAE,0))  )
        
                    VR_m_right.markdown( str(round(VR_test_stat_MAE,0)) )

                    if  st.session_state['Model_Performance_Toggle']:
                        st.markdown('#####')
                        st.markdown('#####')

                        st.markdown('#### Graphs to compare Model Performance to NAL Values On the Trimmed Test Set (Vacant Residential Model)')

                        graph_metric_VR = st.radio(
                                                "## Model Metric:"
                                                ,['R squared  ' , 'Weighted Mean Sale Ratio  ' , 'COD  ' , 'PRD  ' , 'Mean Absolute Error  ']
                                                , horizontal = True
                                                , index = 0
                                            )                    


                        if graph_metric_VR == 'R squared  ':  
                            
                            graph_data_VR = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Vac_Res_preformance_dict"]["Trimmed Test R Squared"] , st.session_state["Vac_Res_preformance_dict"]["Trimmed Test NAL R Squared"]] }
                            graph_lab_VR = 'R Squared'

                        elif graph_metric_VR == 'Weighted Mean Sale Ratio  ':
                            graph_data_VR = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Vac_Res_preformance_dict"]["Trimmed Test Weighted Mean"] , st.session_state["Vac_Res_preformance_dict"]["Trimmed Test NAL Weighted Mean"]] }
                            graph_lab_VR = 'Weighted Mean Sale Ratio'

                        elif graph_metric_VR== 'COD  ':
                            graph_data_VR = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Vac_Res_preformance_dict"]["Trimmed Test COD"] , st.session_state["Vac_Res_preformance_dict"]["Trimmed Test NAL COD"]] }
                            graph_lab_VR = 'COD'

                        elif graph_metric_VR == 'PRD  ':
                            graph_data_VR = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Vac_Res_preformance_dict"]["Trimmed Test PRD"] , st.session_state["Vac_Res_preformance_dict"]["Trimmed Test NAL PRD"]] }
                            graph_lab_VR = 'PRD'    

                        elif graph_metric_VR == 'Mean Absolute Error  ':     
                            graph_data_VR = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Vac_Res_preformance_dict"]["Trimmed Test MAE"] , st.session_state["Vac_Res_preformance_dict"]["Trimmed Test NAL MAE"]] }
                            graph_lab_VR = 'Mean Absolute Error'


                        st.markdown('#####')

                        
                        st.bar_chart( pd.DataFrame(graph_data_VR)
                                    , x = 'Value Source'
                                    
                                    , y = 'Metric'
                                    
                                    , y_label= graph_lab_VR
                                    
                                    , horizontal = True

                    
                                    , height = 200
                                    , width= 150
                                    )


                if st.session_state["Commercial_preformance_dict"] is not None: 

                    st.markdown('###')
                    st.markdown("### :department_store: :factory: Commercial Model Performance:")

                    Com_left, Com_middle = st.columns([.6 , .4])

                    Com_left.markdown("**Test Data Size Before Trimming**:")
                    Com_middle.markdown(str(st.session_state["Commercial_preformance_dict"]["Number of Records Before Trimming Test"]))
                    Com_left.markdown("**Test Data Size After Trimming**:")
                    Com_middle.markdown(str(st.session_state["Commercial_preformance_dict"]["Number of Records After Trimming Test"]))

                    st.markdown('####')

                    Com_radio_left, Com_radio_right = st.columns([.5 , .5])

                    train_stats_type_Com = Com_radio_left.radio('**Training Model Results are Calculated on :**   '
                                                            , ["Un-Trimmed Data" ,'Trimmed Data']
                                                            , index  = 0 )
                    
                    
                    test_stats_type_Com = Com_radio_right.radio('**Test Model Results are Calculated on :**   '
                                                                    , ["Un-Trimmed Data" ,'Trimmed Data']
                                                                    , index = 1)            


                    st.markdown('#####')



                    Com_m_left , Com_m_middle, Com_m_right = st.columns([.5 , .25 , .25])

                    Com_m_left.markdown("**Model Metric**")
                    Com_m_middle.markdown("**Training Pefrormance**")
                    Com_m_right.markdown("**Test Pefrormance**")
                    Com_m_left.markdown('')
                    Com_m_middle.markdown('')
                    Com_m_right.markdown('')

                    Com_train_stat_r_square = st.session_state["Commercial_preformance_dict"]["Train R Squared"]
                    Com_train_stat_weighted_mean_sale_ratio = st.session_state["Commercial_preformance_dict"]["Train Weighted Mean"]
                    Com_train_stat_COD = st.session_state["Commercial_preformance_dict"]["Train COD"]
                    Com_train_stat_PRD = st.session_state["Commercial_preformance_dict"]["Train PRD"]
                    Com_train_stat_MAE = st.session_state["Commercial_preformance_dict"]["Train MAE"]

                    if train_stats_type_Com == "Trimmed Data":
                        Com_train_stat_r_square = st.session_state["Commercial_preformance_dict"]["Trimmed Train R Squared"]
                        Com_train_stat_weighted_mean_sale_ratio = st.session_state["Commercial_preformance_dict"]["Trimmed Train Weighted Mean"]
                        Com_train_stat_COD = st.session_state["Commercial_preformance_dict"]["Trimmed Train COD"]
                        Com_train_stat_PRD = st.session_state["Commercial_preformance_dict"]["Trimmed Train PRD"]
                        Com_train_stat_MAE = st.session_state["Commercial_preformance_dict"]["Trimmed Train MAE"]


                    Com_test_stat_r_square = st.session_state["Commercial_preformance_dict"]["Test R Squared"]
                    Com_test_stat_weighted_mean_sale_ratio = st.session_state["Commercial_preformance_dict"]["Test Weighted Mean"]
                    Com_test_stat_COD = st.session_state["Commercial_preformance_dict"]["Test COD"]                
                    Com_test_stat_PRD = st.session_state["Commercial_preformance_dict"]["Test PRD"]
                    Com_test_stat_MAE = st.session_state["Commercial_preformance_dict"]["Test MAE"]

                    if test_stats_type_Com == "Trimmed Data":
                        Com_test_stat_r_square = st.session_state["Commercial_preformance_dict"]["Trimmed Test R Squared"]
                        Com_test_stat_weighted_mean_sale_ratio = st.session_state["Commercial_preformance_dict"]["Trimmed Test Weighted Mean"] 
                        Com_test_stat_COD = st.session_state["Commercial_preformance_dict"]["Trimmed Test COD"]
                        Com_test_stat_PRD = st.session_state["Commercial_preformance_dict"]["Trimmed Test PRD"]
                        Com_test_stat_MAE = st.session_state["Commercial_preformance_dict"]["Trimmed Test MAE"]
                    

                    Com_m_left.markdown("**R Squared**")
                    Com_perf_color = 'green' if Com_train_stat_r_square > 0.8 else 'orange'
                    if Com_train_stat_r_square < .5 :
                        Com_perf_color = 'red'
                    Com_m_middle.markdown( ":" + Com_perf_color + "["  + str(Com_train_stat_r_square ) + "]"   )
                    Com_perf_color = 'green' if Com_test_stat_r_square> 0.8 else 'orange'
                    if Com_test_stat_r_square < .5 :
                        Com_perf_color = 'red'
                    Com_m_right.markdown( ":" + Com_perf_color + "["  + str(Com_test_stat_r_square) + "]"   )

                    Com_m_left.markdown("**Weighted Mean Sale Ratio**")
                    Com_perf_color = 'green' if abs(Com_train_stat_weighted_mean_sale_ratio - 1 ) <= 0.03 else 'orange'
                    if abs(Com_train_stat_weighted_mean_sale_ratio - 1 ) > 0.05 :
                        Com_perf_color = 'red'
                    Com_m_middle.markdown( ":" + Com_perf_color + "["  + str(Com_train_stat_weighted_mean_sale_ratio) + "]"   )
                    Com_perf_color = 'green' if abs(Com_test_stat_weighted_mean_sale_ratio - 1 ) <= 0.03 else 'orange'
                    if abs(Com_test_stat_weighted_mean_sale_ratio - 1 ) > 0.05 :
                        Com_perf_color = 'red'
                    Com_m_right.markdown( ":" + Com_perf_color + "["  + str(Com_test_stat_weighted_mean_sale_ratio) + "]"   )

                    Com_m_left.markdown("**COD**")
                    Com_perf_color = 'green' if Com_train_stat_COD < 10 else 'orange'
                    if Com_train_stat_COD > 15 :
                        Com_perf_color = 'red'
                    Com_m_middle.markdown( ":" + Com_perf_color + "["  + str(Com_train_stat_COD) + "]"   )
                    Com_perf_color = 'green' if Com_test_stat_COD < 10 else 'orange'
                    if Com_test_stat_COD > 15 :
                        Com_perf_color = 'red'
                    Com_m_right.markdown( ":" + Com_perf_color + "["  + str(Com_test_stat_COD) + "]"   )

                    Com_m_left.markdown("**PRD**")
                    Com_perf_color = 'green' if (Com_train_stat_PRD   > .97 ) and (Com_train_stat_PRD   > 1.02 ) else 'orange'
                    if abs(Com_train_stat_PRD  - 1 ) > .04 :
                        Com_perf_color = 'red'
                    Com_m_middle.markdown( ":" + Com_perf_color + "["  + str(Com_train_stat_PRD ) + "]"   )
                    Com_perf_color = 'green' if (Com_test_stat_PRD < .97) and (Com_test_stat_PRD > 1.02)  else 'orange'
                    if abs(Com_test_stat_PRD - 1 )  > 0.04 :
                        Com_perf_color = 'red'
                    Com_m_right.markdown( ":" + Com_perf_color + "["  + str(Com_test_stat_PRD) + "]"   )

                    Com_m_left.markdown("**Mean Absolute Error**")
            
                    Com_m_middle.markdown(  str(round(Com_train_stat_MAE,0))  )
        
                    Com_m_right.markdown( str(round(Com_test_stat_MAE,0)) )


                    if  st.session_state['Model_Performance_Toggle']:
                        st.markdown('#####')
                        st.markdown('#####')

                        st.markdown('#### Graphs to compare Model Performance to NAL Values On the Trimmed Test Set (Commercial Model)')

                        graph_metric_COM = st.radio(
                                                "## Model Metric:"
                                                ,['R squared   ' , 'Weighted Mean Sale Ratio   ' , 'COD   ' , 'PRD   ' , 'Mean Absolute Error   ']
                                                , horizontal = True
                                                , index = 0
                                            )                    


                        if graph_metric_COM == 'R squared   ':  
                            
                            graph_data_COM = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Commercial_preformance_dict"]["Trimmed Test R Squared"] , st.session_state["Commercial_preformance_dict"]["Trimmed Test NAL R Squared"]] }
                            graph_lab_COM = 'R Squared'

                        elif graph_metric_COM == 'Weighted Mean Sale Ratio   ':
                            graph_data_COM = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Commercial_preformance_dict"]["Trimmed Test Weighted Mean"] , st.session_state["Commercial_preformance_dict"]["Trimmed Test NAL Weighted Mean"]] }
                            graph_lab_COM = 'Weighted Mean Sale Ratio'

                        elif graph_metric_COM == 'COD   ':
                            graph_data_COM = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Commercial_preformance_dict"]["Trimmed Test COD"] , st.session_state["Commercial_preformance_dict"]["Trimmed Test NAL COD"]] }
                            graph_lab_COM = 'COD'

                        elif graph_metric_COM == 'PRD   ':
                            graph_data_COM = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Commercial_preformance_dict"]["Trimmed Test PRD"] , st.session_state["Commercial_preformance_dict"]["Trimmed Test NAL PRD"]] }
                            graph_lab_COM = 'PRD'    

                        elif graph_metric_COM == 'Mean Absolute Error   ':     
                            graph_data_COM = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Commercial_preformance_dict"]["Trimmed Test MAE"] , st.session_state["Commercial_preformance_dict"]["Trimmed Test NAL MAE"]] }
                            graph_lab_COM = 'Mean Absolute Error'


                        st.markdown('#####')

                        
                        st.bar_chart( pd.DataFrame(graph_data_COM)
                                    , x = 'Value Source'
                                    
                                    , y = 'Metric'
                                    
                                    , y_label= graph_lab_COM
                                    
                                    , horizontal = True

                    
                                    , height = 200
                                    , width= 150
                                    )


                if (st.session_state['DOR_Stats_Toggle'])  :
                    st.markdown('#####')
                    st.markdown('#####')
                    st.markdown('#### DOR Summary Statistics Tables For Stratum 1 Properties based on Model Values')
                    st.markdown('#####')

                    if st.session_state["Stratum1GroupLevelStatistics_Report"] is not None:
                        st.markdown('##### Stratum 1 Non Condo Value Summary')
                        st.dataframe(st.session_state["Stratum1_report"] )

                        st.markdown('#####')

                        st.markdown('##### Stratum 1 Non Condo Group Summary')
                        st.dataframe(st.session_state["Stratum1GroupLevelStatistics_Report"] )

                    if st.session_state["Stratum1GroupLevelStatistics_Report_Condos"] is not None:
                        st.markdown('##### Stratum 1 Condo Value Summary')
                        st.dataframe(st.session_state["Stratum1_report_Condos"] )

                        st.markdown('#####')

                        st.markdown('##### Stratum 1 Condo Group Summary')
                        st.dataframe(st.session_state["Stratum1GroupLevelStatistics_Report_Condos"] )


                    if st.session_state["Stratum4GroupLevelStatistics_Report"] is not None:
                        st.markdown('##### Stratum 4 Value Summary')
                        st.dataframe(st.session_state["Stratum4_report"] )

                        st.markdown('#####')

                        st.markdown('##### Stratum 4 Group Summary')
                        st.dataframe(st.session_state["Stratum4GroupLevelStatistics_Report"] )

                    if st.session_state["Stratum6GroupLevelStatistics_Report"] is not None:
                        st.markdown('##### Stratum 6 Value Summary')
                        st.dataframe(st.session_state["Stratum6_report"] )

                        st.markdown('#####')

                        st.markdown('##### Stratum 6 Group Summary')
                        st.dataframe(st.session_state["Stratum6GroupLevelStatistics_Report"] )


                

                st.markdown('#####')
                st.markdown('#####')
                st.markdown('#####')
                st.markdown('#####')


                if st.session_state["SFR_Model_Predicitons_For_Review"] is not None : 
          
                    st.download_button(
                                        label="Download All SFR Predictions for my county!",
                                        data = convert_df(st.session_state["SFR_Model_Predicitons_For_Review"] ),
                                        file_name = "SFR_Predictions.csv",
                                        icon="🚨",
                                        mime = "text/csv",
                                    )


                if st.session_state["Condo_Model_Predicitons_All"] is not None : 
                   
                    st.download_button(
                                        label="Download All Condo Predictions for my county!",
                                        data = convert_df(st.session_state["Condo_Model_Predicitons_All"] ),
                                        file_name = "Condo_Predictions.csv",
                                        icon="🚨",
                                        mime = "text/csv",          
                    )

                if st.session_state["Vac_Res_Model_Predicitons_All"] is not None : 
                    
                    st.download_button(
                                        label="Download All Vacant Residential Predictions for my county!",
                                        data = convert_df(st.session_state["Vac_Res_Model_Predicitons_All"] ),
                                        file_name = "Vac_Res_Predictions.csv",
                                        icon="🚨",
                                        mime = "text/csv",          
                    )

                if st.session_state["Commercial_Model_Predicitons_All"] is not None : 
                      
                    st.download_button(
                                        label="Download All Commercial Predictions for my county!",
                                        data = convert_df(st.session_state["Commercial_Model_Predicitons_All"] ),
                                        file_name = "Commercial_Predictions.csv",
                                        icon="🚨",
                                        mime = "text/csv",          
                    )
                
                if (st.session_state["Commercial_Model_Predicitons_All"] is not None ) or (st.session_state["Condo_Model_Predicitons_All"] is not None ) or (st.session_state["SFR_Model_Predicitons_All"] is not None ) or (st.session_state["Vac_Res_Model_Predicitons_All"] is not None ) :
                    
                    #all_data  =  pd.DataFrame(columns = ['PARCEL_ID','Model_Prediction']) 
                    # if st.session_state["SFR_Model_Predicitons_All"] is not None :
                    #     all_data = pd.concat([all_data, st.session_state["SFR_Model_Predicitons_All"]], ignore_index = True)    

                    # if st.session_state["Condo_Model_Predicitons_All"] is not None :
                    #     all_data = pd.concat([all_data, st.session_state["Condo_Model_Predicitons_All"]], ignore_index = True)

                    # if st.session_state["Vac_Res_Model_Predicitons_All"] is not None :
                    #     all_data = pd.concat([all_data, st.session_state["Vac_Res_Model_Predicitons_All"]], ignore_index = True)
                    
                    # if st.session_state["SFR_Model_Predicitons_All"] is not None :
                    #     all_data = pd.concat([all_data, st.session_state["SFR_Model_Predicitons_All"]], ignore_index = True)

                    all_data  = None


                    if st.session_state["SFR_Model_Predicitons_All"] is not None :
                        if all_data is None: 
                            all_data = st.session_state["SFR_Model_Predicitons_All"].copy() 
                        else:    
                            all_data = pd.concat([all_data, st.session_state["SFR_Model_Predicitons_All"]], ignore_index = True)    

                    if st.session_state["Condo_Model_Predicitons_All"] is not None :
                        if all_data is None: 
                            all_data = st.session_state["Condo_Model_Predicitons_All"].copy() 
                        else:
                            all_data = pd.concat([all_data, st.session_state["Condo_Model_Predicitons_All"]], ignore_index = True)

                    if st.session_state["Vac_Res_Model_Predicitons_All"] is not None :
                        if all_data is None:
                            all_data = st.session_state["Vac_Res_Model_Predicitons_All"].copy() 
                        else:
                            all_data = pd.concat([all_data, st.session_state["Vac_Res_Model_Predicitons_All"]], ignore_index = True)
                    
                    if st.session_state["Commercial_Model_Predicitons_All"] is not None :
                        if all_data is None:
                            all_data = st.session_state["Commercial_Model_Predicitons_All"].copy()
                        else:
                            all_data = pd.concat([all_data, st.session_state["Commercial_Model_Predicitons_All"]], ignore_index = True)
                    
                    
                    st.download_button(
                                        label="Download All Model(s) Predictions for my county!",
                                        data = convert_df(all_data),
                                        file_name = "All_Model_Predictions.csv",
                                        icon="🚨",
                                        mime = "text/csv",          
                    )

                st.markdown('#####')
                st.markdown('#####')
                st.markdown('#####')
                st.markdown('#### How About we Discuss your Data!')
                st.write("---")
                st.markdown('#####')
                
                
                if st.session_state["Chat_Launched"] == None: 

                    ch_col1, ch_col2  = st.columns(2)

                    with ch_col1:
                        st.selectbox('Pick A Chat Model '
                                    , ['gemini-1.5-flash','gemini-2.0-flash','gemini-2.0-flash-Lite', 'gemini-2.5-flash']
                                    ,placeholder = 'Models Avaialable' 
                                    , key = 'Chat_Model_Selected'
                                    , on_change = set_chatmodel)
                        #st.write(f"You selected: {st.session_state['Chat_Model_Selected']}")
                    
                    with ch_col2:
                        st.text_input('Enter Google AI Studio API Key'
                                    , key = 'Provided_Google_Key'
                                    , placeholder = 'Not Required for Most Models'
                                    ,label_visibility="hidden"
                                    , on_change = set_User_API_key)
                        

                    st.markdown('#####')

                    
                    
                    Build_Chat_Bot_Button = st.button('Start Chat!')

                    if Build_Chat_Bot_Button:
                        if st.session_state["Provided_Google_Key"] is not None:
                            with st.spinner("Hold on Tight walking up the Chat Bot!"):
                                response_container_Chat = st.empty()                                    
                                with response_container_Chat.container():
                                    progress_bar_Chat = st.columns([2,5,2])                                
                                    with progress_bar_Chat[1]:                                                                            
                                        #my_bar = st.progress(0, text='')  
                                        try:    
                                            render_animation2()
                                        except:
                                            st.spinner("Im working hard!")
                                    Build_Chat_Bot_And_Data()
                                    response_container_Chat.empty()
                        else:
                            st.warning('Please Enter a Google AI Studio API Key')

                if st.session_state["Chat_Launched"] != None: 
                 

                    st.button('Clear Chat History'
                              , on_click = Clear_Chat_History)
                    
                    # st.button('Checking Query '
                    #           , on_click = Query_Check)
                    
                    st.markdown('#####')
                    st.markdown('#####')
                    st.markdown('#####')
                    


                    for message in st.session_state["Chat_History"] :
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    # React to user input
                    if prompt := st.chat_input("What Would you like to know about your NAL?"):
                        # Display user message in chat message container
                        st.chat_message("user").markdown(prompt)
                        # Add user message to chat history
                        st.session_state["Chat_History"].append({"role": "user", "content": prompt})


                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            response = st.write_stream(response_generator(prompt))
                            if st.session_state["Display_Plot"] == 1:
                                st.pyplot(st.session_state["Chat_Bot_Plot"])
                        # Add assistant response to chat history
                        st.session_state["Chat_History"].append({"role": "assistant", "content": response})

                    
  
                

                                                



                


    if exit_app:
        ## Stop the Streamlit server.
        ##
        ## This button used to synthesize a Ctrl+W keystroke to close the
        ## browser tab as well, using a package called `keyboard`. That
        ## package has been unmaintained since March 2020, it installs a
        ## GLOBAL low-level keyboard hook that often needs administrator
        ## rights on Windows, and it sends the keystroke to whichever window
        ## currently has focus, which is not necessarily your browser.
        ##
        ## We are asking county IT departments to approve this repository,
        ## and a dependency shaped like a keylogger fails that review for
        ## good reasons. So we stop the server and let you close the tab.
        st.info('Deep Thought is shutting down. You can close this tab now.')
        pid = os.getpid()
        p = psutil.Process(pid)
        p.terminate()



### Include All Stratums -- Done
### Get 2023 and 2024 for all counties -- Done
### Test 2023 NAL to 2024 NAL -- Done
### Check Sale Coverage to pop and identify areas of weakness and warn 