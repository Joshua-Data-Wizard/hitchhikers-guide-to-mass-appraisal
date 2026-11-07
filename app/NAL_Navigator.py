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
import sys
from pathlib import Path
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
from sklearn.model_selection import KFold

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

## Used by the local model's tool-call parser further down.
import re
import json

## The hosted language model behind the chat assistant at the end of the
## app. One variable is needed - ANTHROPIC_API_KEY - and .env.example
## says where to get it. A reader who never opens the chat section never
## needs a key; everything up to and including the valuation models runs
## with no API access at all.
import anthropic
from anthropic import beta_tool

## Visualizing Data 
## Visualizing Data
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

## Wide layout, on purpose. This was commented out, and the app rendered
## in Streamlit's default narrow centred column - which is fine for a
## blog post and wrong for this. The DOR summary tables are NINE columns
## wide and the model performance tables sit side by side; in the narrow
## column they wrap into something an assessor has to decode rather than
## read.
##
## This must stay the FIRST Streamlit call in the file. set_page_config
## raises if anything else has already drawn to the page.
st.set_page_config(page_title = "NAL Navigator"
                   , page_icon = ":house:"
                   , layout = "wide")


## ── Where the app writes its own working files ───────────────────────────
##
## The chat assistant needs somewhere to put the small SQLite database it
## queries. That location must be worked out at run time.
##
## 🩹 An earlier version of this file hardcoded an absolute path:
##
##     C:\\Users\\<a name>\\...\\Contracting\\NAL Project\\Data\\NAL_Navigator.db
##
## That folder exists on exactly one computer. On that computer the chat
## worked perfectly; everywhere else it raised, which is the most
## expensive kind of bug because the person who could fix it is the one
## person who never sees it.
def find_project_root(marker = 'pyproject.toml'):
    """Climb from this file towards the root until `marker` turns up.

    Looking for a landmark beats counting `.parent` calls, because the
    answer stays right no matter which directory you launched from.
    """
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / marker).exists():
            return candidate
    return here

PROJECT_ROOT = find_project_root()

## Everything the app generates for itself goes here. It is gitignored,
## and you can delete the whole folder without losing anything you cannot
## rebuild by pressing the buttons again.
APP_WORKSPACE = PROJECT_ROOT / "app_workspace"


## ── Shared code: src/deep_thought ───────────────────────────────────────
##
## Anything the notebook and this app BOTH need lives in src/, so there is
## one copy of it. Right now that is the IAAO ratio statistics, which used
## to exist three times over in this project.
##
## 📌 Why a sys.path line rather than an installed package: pyproject.toml
## sets `package = false`, because this repository is a notebook and an app
## rather than a library somebody pip-installs. That keeps setup to a
## single `uv sync` with no build step - which is worth one explicit line
## here. The notebook does the same thing from its own project root.
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deep_thought.ratio_stats import cod as shared_cod
from deep_thought.ratio_stats import prd as shared_prd
from deep_thought.ratio_stats import weighted_mean_ratio as shared_weighted_mean
from deep_thought.ratio_stats import format_statistic
from deep_thought.ratio_study import (SALES_TABLE, build_sales_rows, run_ratio_study,
                                      describe_acceptable_ranges, range_key,
                                      PROPERTY_TYPES, STARTING_RANGES, RANGE_LABELS,
                                      THIN_GROUP_WARNING, MINIMUM_SALES_FOR_A_STATISTIC)


## ── The bundled example: Lee County, Florida ─────────────────────────────
##
## The repository ships one real county roll so that the app can be tried
## without finding data first. It is committed gzipped and unpacked once:
##
##     uv run python scripts/extract_data.py
##
## 🩹 This replaces a dropdown of all 67 Florida counties whose file paths
## were absolute locations on one person's computer. Selecting a county
## printed an encouraging message and then raised FileNotFoundError for
## everybody else - 136 lines of code that could only ever work once.
##
## Lee County is the worked EXAMPLE. Any other county brings its own roll
## through the upload boxes, which is the honest arrangement: we cannot
## ship 67 counties of data, and an assessor already has their own.
def find_sample_data():
    """Look for the bundled roll in data/, if it has been unpacked yet.

    Globbed rather than named, so next year's roll drops in without an
    edit here - and so a reader who renames a file still gets found.
    """
    data_dir = PROJECT_ROOT / "data"

    def first(pattern):
        found = sorted(data_dir.glob(pattern))
        return found[0] if found else None

    return {
        "dir": data_dir,
        "nal": first("NAL*.csv"),   # the parcels
        "sdf": first("SDF*.csv"),   # the sales
        "shp": first("*.shp"),      # the parcel shapes (optional)
    }


## ── Reading a parcel shapefile without filling your RAM ─────────────────
##
## A county shapefile carries one attribute table row per parcel, and
## counties put a LOT in it. Lee County's has 119 columns. We use exactly
## two of them: PARCEL_ID, to match against the roll, and the geometry,
## to get a point for each parcel.
##
## Measured on the bundled file:
##
##     all 119 columns      18.3 s   +1,097 MB of memory
##     just PARCEL_ID        5.1 s     +169 MB
##
## ⚠️ That 1.1 GB arrives while the 285 MB roll is ALREADY loaded. On a
## county laptop with 8 GB that is the difference between working and
## swapping. We are designing for that machine, not for a workstation.
def read_parcel_shapes(shapefile_path):
    """Read a parcel shapefile, keeping only what the merge needs.

    Falls back to a full read if the file does not have a PARCEL_ID
    column, so somebody else's shapefile still loads - it will fail later
    in the merge instead, which says plainly what is missing.
    """
    ## 📌 Expect a warning on the terminal when this runs:
    ##
    ##     RuntimeWarning: organizePolygons() received an unexpected
    ##     geometry. Either a polygon with interior rings, or a polygon
    ##     with less than 4 points, or a non-Polygon geometry.
    ##
    ## Nothing is broken. That is GDAL reporting something true about the
    ## COUNTY'S data: some parcels are drawn with holes in them, and some
    ## with too few points to be a polygon at all. Both are ordinary in a
    ## real roll - a parcel with an excluded easement, a sliver left by a
    ## survey correction.
    ##
    ## We do NOT suppress it. It is a data-quality signal about the roll
    ## you just loaded, and the centroid of a parcel with an interior ring
    ## can land somewhere surprising. The notebook keeps it for the same
    ## reason.
    try:
        return gpd.read_file(shapefile_path, columns = ["PARCEL_ID"])
    except Exception as e:
        print(f' - could not read just PARCEL_ID ({e}); reading every column')
        return gpd.read_file(shapefile_path)


## ── Which model answers questions about your roll ────────────────────────
##
## The label is what an assessor reads; the value is the model id sent to
## the API.
##
## Haiku is the default. Most questions an assessor asks of a roll are
## straightforward - a count, a median, a breakdown by neighbourhood,
## a histogram - and it answers those perfectly well for a fraction of
## the cost. An office trying this out should not discover the bill
## before it discovers whether the tool is useful.
##
## 📌 Switch up for a question worth more thought. The harder ones here
## are the multi-step ones, where a wrong SQL query becomes a confident
## wrong answer about somebody's property - so if an answer looks off,
## ask Sonnet or Opus the same question before believing either.
##
## ⚠️ Every model in this dict has been checked to accept the `effort`
## setting used in NAL_Chat_Session. If you add one, check that too -
## see the note there.
CHAT_MODELS = {
    "Claude Haiku 5.5  -  fast and cheap, the default": "claude-haiku-5-5",
    "Claude Sonnet 5.5  -  more capable"              : "claude-sonnet-5-5",
    "Claude Opus 5.5  -  most capable"                : "claude-opus-5-5",
}

## Only these three are sent the `effort` setting - see NAL_Chat_Session.
## This is captured BEFORE anything is added below, so a model that comes
## from a .env file never inherits a parameter nobody has tested with it.
VERIFIED_CHAT_MODELS = set(CHAT_MODELS.values())


## ANTHROPIC_MODEL in your .env chooses the starting model. If it names
## something not in the list above - you may have access to a model this
## app does not know about - it gets its own entry in the dropdown rather
## than being thrown away.
##
## 🩹 It WAS being thrown away. An unlisted model had no dropdown label,
## so the label lookup fell back to the first entry and the app quietly
## used Opus while a message on the terminal claimed otherwise. Honouring
## the setting and showing it on screen beats explaining it in a print
## nobody reads - and the module re-runs on every click, so that print
## appeared eight times in one short session.
_env_model = (os.environ.get("ANTHROPIC_MODEL") or "").strip()

if _env_model and _env_model not in VERIFIED_CHAT_MODELS:
    CHAT_MODELS[f"{_env_model}  -  from your .env file"] = _env_model

DEFAULT_CHAT_MODEL = _env_model or "claude-haiku-5-5"

## The dropdown needs a LABEL, not a model id. Every possible value of
## DEFAULT_CHAT_MODEL now has one, because an unlisted model just got
## added above; the fallback is belt and braces.
DEFAULT_CHAT_MODEL_LABEL = next(
    (label for label, model in CHAT_MODELS.items() if model == DEFAULT_CHAT_MODEL),
    list(CHAT_MODELS.keys())[0],
)


## ── Two ways to run the assistant ─────────────────────────────────────
##
## Hosted sends your questions, and the rows the tools return, to
## Anthropic. Local runs a small model on this computer and sends nothing
## anywhere. For many assessor offices Local is the only option their IT
## department will approve, which is exactly why it is here.
##
## 📌 Local is not a lesser version of the same thing - it is a smaller
## model with a SHORTER TOOL BELT, and the app says so on screen rather
## than letting somebody find out from a wrong number.
CHAT_ENGINE_HOSTED = "Hosted  -  Claude, over the internet"
CHAT_ENGINE_LOCAL  = "Local  -  a small model on this computer"
CHAT_ENGINES = [CHAT_ENGINE_HOSTED, CHAT_ENGINE_LOCAL]
DEFAULT_CHAT_ENGINE = CHAT_ENGINE_HOSTED


def using_local_engine():
    """True when the person has chosen to run the model on this machine."""
    return st.session_state.get("Chat_Engine_Selected") == CHAT_ENGINE_LOCAL

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

## Where the data in front of you came from, for the line that reports it.
if "Data_Source" not in st.session_state:
    st.session_state['Data_Source'] = None

if "State_Coords_Selected" not in st.session_state:
    st.session_state['State_Coords_Selected'] = None

if "EPSG_List" not in st.session_state:
    st.session_state['EPSG_List'] = ["EPSG:2236" ,"EPSG:2237","EPSG:2238" ]

if "COS_ADJ_KEY" not in st.session_state:
    st.session_state["COS_ADJ_KEY"] = .85

## 🩹 Seeded with a real code, not None.
##
## The selectbox that owns this key passed `index = 1` to start on '01'.
## A Streamlit widget whose key is ALREADY in session state uses the
## session-state value and ignores its own default, so this stayed None -
## and pressing "Update the DR 493 Adjustment" without first touching the
## dropdown raised TypeError on `'... for ' + None`.
##
## Worse if that print had not been there: the next line would have
## written None in as a dictionary key, and the adjustment would have been
## silently discarded, because every lookup uses str(row['DOR_UC']).
## A crash was the lucky outcome. FINDINGS 32.
if  "current_Adjust_dor" not in st.session_state:
    st.session_state['current_Adjust_dor'] = '01'

if "Hyperparameter_Tuning_Toggle" not in st.session_state:
    st.session_state['Hyperparameter_Tuning_Toggle'] = False 


if "Hyperparameter_Tuning_Trials" not in st.session_state:
    st.session_state['Hyperparameter_Tuning_Trials'] = 50

## How many cross-validation folds each trial is scored over. 5 is the
## usual choice and what the measurements below assume. Fewer is faster
## and noisier; more is slower and steadier.
if "Hyperparameter_Tuning_Folds" not in st.session_state:
    st.session_state['Hyperparameter_Tuning_Folds'] = 5

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

## The key read from .env at start-up (may be None - that is fine until
## somebody opens the chat section).
if  "Anthropic_Key" not in st.session_state:
    st.session_state['Anthropic_Key'] = os.environ.get('ANTHROPIC_API_KEY')

## A key typed into the box in the chat section, which overrides the above.
if  "Provided_Anthropic_Key" not in st.session_state:
    st.session_state['Provided_Anthropic_Key'] = None 

## 🩹 This used to be seeded to None, and the selectbox that owns the key
## passed `index = 0` to choose its own default. A Streamlit widget whose
## key is ALREADY in session state uses the session-state value and
## ignores its own default - so the model you picked was discarded and the
## app silently fell back to whatever .env said.
##
## Unlike `value =` on a number_input, a selectbox's `index =` does not
## warn about this, so it failed quietly. Seeding a real label is the fix.
if "Chat_Model_Selected" not in st.session_state:
        st.session_state['Chat_Model_Selected'] = DEFAULT_CHAT_MODEL_LABEL

## 🚨 Seeded here, and the radio further down passes NO `index =`. A
## Streamlit widget given BOTH a default and a pre-seeded session_state
## entry under its key silently throws its own default away - which once
## discarded the chat model a person had picked, on every run. One source
## of truth, and this is it. FINDINGS 32.
if "Chat_Engine_Selected" not in st.session_state:
        st.session_state['Chat_Engine_Selected'] = DEFAULT_CHAT_ENGINE

## Why the last Start Chat failed, in words for the person reading it. It has
## to outlive the container it happened inside, which gets cleared.
if "Chat_Start_Problem" not in st.session_state:
        st.session_state['Chat_Start_Problem'] = None

## ── The Settings tab's boxes ───────────────────────────────────────────
##
## 📌 What counts as acceptable belongs to the jurisdiction, not to this app. The
## rules differ from place to place and they change, so these are only what the
## boxes start with. Seeded here, and the boxes pass no `value =`. FINDINGS 32.
for _property_type in PROPERTY_TYPES:
    for _statistic, _limits in STARTING_RANGES.items():
        for _bound, _start in zip(("low", "high"), _limits):
            if range_key(_property_type, _statistic, _bound) not in st.session_state:
                st.session_state[range_key(_property_type, _statistic, _bound)] = float(_start)

if "Minimum_Sales_For_A_Statistic" not in st.session_state:
        st.session_state['Minimum_Sales_For_A_Statistic'] = MINIMUM_SALES_FOR_A_STATISTIC

if "Thin_Group_Warning" not in st.session_state:
        st.session_state['Thin_Group_Warning'] = THIN_GROUP_WARNING


def Current_Acceptable_Ranges():
    """The ranges as they stand on the Settings tab, read fresh every time."""
    return {property_type: {statistic: (st.session_state[range_key(property_type, statistic, "low")],
                                        st.session_state[range_key(property_type, statistic, "high")])
                            for statistic in STARTING_RANGES}
            for property_type in PROPERTY_TYPES}


def Range_Colour(value, statistic, property_type):
    """Green inside the range on the Settings tab, red outside it.

    🩹 These colours used to be hardcoded four times over, once per model, and the
    PRD test was wrong in all four. `(x < .97) and (x > 1.02)` can never be true, so
    a test PRD could not be green however good it was, and the train column only
    went green ABOVE 1.02. That is backwards: 1.00 showed orange and 1.02 showed
    green. There was also an orange band, a second pair of invented numbers per
    statistic, so now a figure is simply inside your range or it is not.
    """
    low, high = Current_Acceptable_Ranges()[property_type][statistic]
    if value is None or pd.isna(value):
        return 'gray'
    return 'green' if low <= value <= high else 'red'


## Each model's qualified sales, keyed by property type, ready to become the
## chat database's Sales table. See build_sales_rows.
if "Sales_For_Chat" not in st.session_state:
        st.session_state['Sales_For_Chat'] = {}

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


## ── What these numbers are, and what they are not ────────────────────────
##
## 🚫 Defined once and used everywhere, because this is the one thing in
## the app that must never be a footnote.
##
## This app produces a BENCHMARK: something to compare your own values
## against, and a safe place to find out what gradient boosting does. It
## is not an assessment. It is not defensible. It is not evidence in an
## appeal. The model is a deliberately simple baseline trained on one
## year of sales and a handful of features.
##
## An assessor who pastes one of these numbers into a notice of value has
## been failed by this app, so the wording appears on the results screen,
## under every table, in the sidebar on every tab, in the chat
## assistant's instructions, and in the name of the column that lands in
## the spreadsheet you download.
BENCHMARK_SHORT = "Benchmark estimates - not assessed values."

BENCHMARK_FULL = (
    "These are **benchmark estimates**, produced to compare against your "
    "own values. They are **not assessments**, not defensible values, and "
    "not evidence in an appeal.\n\n"
    "The model behind them is a deliberately simple baseline: one year of "
    "qualified sales, a handful of features, and no review of any "
    "individual parcel. It is here so you can see what these methods do "
    "and measure your own roll against something - not to replace the "
    "judgement that goes into a roll."
)

## The column name that lands in a downloaded spreadsheet. The header is
## the only part of the caveat that survives a copy-paste into somebody
## else's workbook, so it has to carry the message itself.
BENCHMARK_COLUMN = "Benchmark_Estimate"

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




## `Update_County_List_Loc()` lived here. It swapped the 67-county path
## dictionary for a second copy pointing at the previous year's files -
## another 68 lines of absolute paths on one person's machine. Both are
## gone; see find_sample_data() near the top of this file.








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

### Sales Ratio Function  ###

## These three used to be implemented here, by hand. They now forward to
## src/deep_thought/ratio_stats.py, which forwards to `pyRealEstate` - so
## the app, the notebook and the report generator all compute COD with the
## same code instead of three copies of the same formula.
##
## 🩹 Nothing can move as a result. The hand-written versions were
## character-for-character identical to pyRealEstate's apart from the
## variable names, verified across eleven cases including nullable Int64,
## pd.NA, a zero divisor and a single row.
##
## The names are kept so that the sixty-odd call sites further down this
## file are untouched - a rename would have been a chance to get one wrong.
##
## ⚠️ Argument order is NOT symmetric: COD(sale_price, appraised_value) is
## not COD(appraised_value, sale_price). Measured on real data the two give
## 14.53 and 15.59.
## 🩹 These return nan, not None, when a statistic cannot be computed.
##
## The shared module returns None - deliberately, so that a missing
## statistic cannot be mistaken for a good one. But the performance
## dictionaries below wrap every one of these in `round(...)` at about a
## hundred call sites, and rounding None RAISES. An empty trimmed frame -
## a Commercial model on a county with few commercial sales - would have
## taken the whole model build down, where previously it displayed nan.
##
## So: nan here, preserving exactly what these call sites always did.
## None is for the DOR report methods, which have Format_Statistic to
## render it as "n/a". Two different audiences, two different answers to
## "there is no number".
def _nan_if_missing(value):
    return float("nan") if value is None else value


def weighted_Mean_Sale_Ratio(sl, ap):
    return _nan_if_missing(shared_weighted_mean(sale_prices = sl, appraised_values = ap))


def COD(sl, ap):
    return _nan_if_missing(shared_cod(sale_prices = sl, appraised_values = ap))


def PRD(sl, ap):
    return _nan_if_missing(shared_prd(sale_prices = sl, appraised_values = ap))



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

    ## ── The report's ratio statistics ───────────────────────────────────
    ##
    ## 🧨 These three used to open with:
    ##
    ##     if all(i == 0 for i in ap) or all(i == 0 for i in sl):
    ##         return 0
    ##
    ## **A COD of 0.00 means perfect uniformity.** To an assessor reading a
    ## summary table, 0.00 is not a blank - it is the best score there is.
    ## So that guard turned "I have nothing to report" into "I am
    ## flawless", and it fired exactly where it was most likely to be
    ## believed: this class deliberately zeroes out value groups with fewer
    ## than `self.n` sales, so a group with four sales printed COD 0.00.
    ##
    ## They now return None, which Format_Statistic renders as "n/a". A
    ## missing statistic should look missing.
    ##
    ## 📌 Left deliberately alone: a group with four sales that was NOT
    ## zeroed upstream still gets a COD computed from four sales, which is
    ## noise wearing a decimal point. Passing `minimum_sales = self.n`
    ## below would suppress those too - but that CHANGES numbers this
    ## report already publishes, so it is a decision for whoever owns the
    ## roll, not a tidy-up to slip in here.
    def DOR_COD(self, sl, ap):
        return shared_cod(sale_prices = sl, appraised_values = ap)

    def DOR_PRD(self, sl, ap):
        return shared_prd(sale_prices = sl, appraised_values = ap)

    def DOR_weighted_Mean(self, sl, ap):
        return shared_weighted_mean(sale_prices = sl, appraised_values = ap)

    def Format_Statistic(self, value, digits = 2, scale = 1.0):
        """For the TEXT frame: a string, or "n/a" when there is no number."""
        return format_statistic(value, digits = digits, scale = scale)

    def Round_Statistic(self, value, digits = 2, scale = 1.0):
        """For the NUMERIC frame: a rounded number, or None.

        The scaling happens here rather than at the call site, because
        the call sites used to multiply by 100 before rounding, and
        multiplying None raises.

        (That sentence deliberately does not spell out the old expression.
        The script that rewrote these call sites found them by regex, and
        an earlier draft of this very docstring MATCHED - so the tool
        counted my own comment as a seventh call site and refused to run.
        Prose that quotes the code a tool is searching for becomes part of
        the search.)
        """
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return round(numeric * scale, digits)
   
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

        ## `inplace=True` is on its way out of pandas (PDEP-8) and warns
        ## today. Assigning the result back does the same job and makes it
        ## obvious which frame changed.
        self.Sorted_StratumPop = self.Sorted_StratumPop.reset_index(drop = True)

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
       
        self.sortedvalues_Untrimmed = self.sortedvalues_Untrimmed.reset_index(drop = True)

        self.G0 = self.Sorted_StratumPop.sort_values(by = self.NAL_Value_Column)
       
        self.G0 = self.G0.reset_index(drop = True)

        self.sortedvalues = self.G0[self.G0[self.NAL_Value_Column] > self.trimmed[self.NAL_Value_Column].max()]
        #self.sortedvalues = self.StratumPop_unique.sort_values(by = self.NAL_Value_Column)
        self.sortedvalues = self.sortedvalues.reset_index(drop = True)

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

        ## 🧨 `index = range(7)` and `dtype = object` are load-bearing here.
        ## Both of them. Removing either brings the warnings back.
        ##
        ## The frames below are the LAYOUT of a DOR report, not a dataset.
        ## Rows 0-4 are value groups, row 5 is a totals line, and row 6
        ## reads   "" | "COD:" | 9.36 | "PRD:" | 101.77 | "95 %" | ...
        ## so a single column genuinely holds a number in one row and a
        ## text label in the next.
        ##
        ## Left to itself, pandas types these columns float64 from the
        ## first rows and then warns that writing "COD:" into them is
        ## deprecated - and in pandas 4 that becomes an ERROR, which would
        ## take the whole summary report down with it.
        ##
        ## 🩹 `dtype = object` ALONE does not fix it, which is worth knowing
        ## because it looks like it should. Measured on the real report:
        ##
        ##     the code as it was                    8 warnings
        ##     empty frame + dtype=object            8 warnings  <- no help
        ##     rows pre-allocated + dtype=object     0 warnings  <- this
        ##
        ## Assigning `.loc[0]` to an EMPTY frame is an *enlargement*, and
        ## enlargement re-infers each column's dtype from the values being
        ## written - so the columns went straight back to float64 before
        ## row 6 ever arrived. Handing the frame its rows up front makes
        ## those writes ordinary assignments into existing object columns,
        ## which never re-infer.
        ##
        ## Every cell is byte-for-byte what it was. Only the dtype differs.
        Stratum_Columns = ['STR', 'GRP', 'LOW', 'HIGH', '#SAMP', 'COV'
                           , 'P.A SAMP VAL', 'TOT PAR', 'P.A JUST VAL']

        ## 7 rows: five value groups, a totals row, and the COD/PRD row.
        self.Stratum = pd.DataFrame(index = range(7)
                                    , columns = Stratum_Columns
                                    , dtype = object)
        self.Stratum_Text = pd.DataFrame(index = range(7)
                                         , columns = Stratum_Columns
                                         , dtype = object)

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

        ## Stratum_Text is the PRINTABLE report - every cell is a string,
        ## including the group number. It used to hold bare ints 1-5 here
        ## and then "" and "COD:" in the summary rows below, so the column
        ## was mixed and Streamlit had to coerce it before it could draw
        ## the table. (The parallel `Stratum` frame keeps real numbers;
        ## that one is for calculating with, this one is for reading.)
        self.Stratum_Text.loc[0] = [self.Stratum_Number + ' S'
                                    , str(1)
                                    , self.Format_My_Numbers(self.G1_trimmed[self.NAL_Value_Column].min() , type = 'int')
                                    , self.Format_My_Numbers(self.G1_trimmed[self.NAL_Value_Column].max(), type = 'int')
                                    , self.Format_My_Numbers(G1S.shape[0], type = 'int')
                                    , ""
                                    , self.Format_My_Numbers(G1S[self.NAL_Value_Column].sum(), type = 'int')
                                    , self.Format_My_Numbers(G1.shape[0], type = 'int')
                                    , self.Format_My_Numbers(G1[self.NAL_Value_Column].sum(), type = 'int')]
        self.Stratum_Text.loc[1] = [self.Stratum_Number + ' S'
                                    , str(2)
                                    , self.Format_My_Numbers(self.G2_trimmed[self.NAL_Value_Column].min(), type = 'int')
                                    , self.Format_My_Numbers(self.G2_trimmed[self.NAL_Value_Column].max(), type = 'int')
                                    , self.Format_My_Numbers(G2S.shape[0], type = 'int')
                                    , ""
                                    , self.Format_My_Numbers(G2S[self.NAL_Value_Column].sum(), type = 'int')
                                    , self.Format_My_Numbers(G2.shape[0], type = 'int')
                                    , self.Format_My_Numbers(G2[self.NAL_Value_Column].sum(), type = 'int')]
        self.Stratum_Text.loc[2] = [self.Stratum_Number + ' S'
                                    , str(3)
                                    , self.Format_My_Numbers(self.G3_trimmed[self.NAL_Value_Column].min(), type = 'int')
                                    , self.Format_My_Numbers(self.G3_trimmed[self.NAL_Value_Column].max(), type = 'int')
                                    , self.Format_My_Numbers(G3S.shape[0], type = 'int')
                                    , ""
                                    , self.Format_My_Numbers(G3S[self.NAL_Value_Column].sum(), type = 'int')
                                    , self.Format_My_Numbers(G3.shape[0], type = 'int')
                                    , self.Format_My_Numbers(G3[self.NAL_Value_Column].sum(), type = 'int')]
        self.Stratum_Text.loc[3] = [self.Stratum_Number + ' S', str(4)
                                    , self.Format_My_Numbers(self.G4_trimmed[self.NAL_Value_Column].min(), type = 'int')
                                    , self.Format_My_Numbers(self.G4_trimmed[self.NAL_Value_Column].max(), type = 'int')
                                    , self.Format_My_Numbers(G4S.shape[0], type = 'int')
                                    , ""
                                    , self.Format_My_Numbers(G4S[self.NAL_Value_Column].sum(), type = 'int')
                                    , self.Format_My_Numbers(G4.shape[0], type = 'int')
                                    , self.Format_My_Numbers(G4[self.NAL_Value_Column].sum(), type = 'int')]
        self.Stratum_Text.loc[4] = [self.Stratum_Number + ' S'
                                    , str(5)
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
                               , self.Round_Statistic(self.DOR_COD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]))
                               ,"PRD:"
                               , self.Round_Statistic(self.DOR_PRD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]), scale = 100)
                               , "95 %"
                               , "Conf Intvl"
                               , round(self.confidence_interval[0],2)
                               , round(self.confidence_interval[1],2)]
       

        self.Stratum_Text.loc[6] = [""
                               , "COD:"
                               , self.Format_Statistic(self.DOR_COD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]))
                               , "PRD:"
                               , self.Format_Statistic(self.DOR_PRD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]), scale = 100)
                               , "95 %"
                               , "Conf Intvl"
                               , str(round(self.confidence_interval[0],2))
                               , str(round(self.confidence_interval[1],2))]
       

        ## Same reason as the Stratum frames above, and the same two-part
        ## fix: the last row's Group cell reads 'Total' while the four rows
        ## above it hold 1, 2, 3, 4.
        Group_Columns = ['Stratum', 'Group', 'N', 'Median', 'Mean'
                         , 'COD', 'PRD', 'Weighted Mean']

        ## 5 rows: four value groups and a total.
        self.GroupLevelStatistics = pd.DataFrame(index = range(5)
                                                 , columns = Group_Columns
                                                 , dtype = object)
        self.GroupLevelStatistics_Text = pd.DataFrame(index = range(5)
                                                      , columns = Group_Columns
                                                      , dtype = object)

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
                                        , self.Round_Statistic(self.DOR_COD(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column]))
                                        , self.Round_Statistic(self.DOR_PRD(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column]), scale = 100)
                                        , self.Round_Statistic(self.DOR_weighted_Mean(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column]), scale = 100)]
        self.GroupLevelStatistics.loc[1] = [self.Stratum_Number
                                        , 2
                                        , G2S.shape[0]
                                        , Group_2_Median_Sale_Ratio
                                        , Group_2_Mean_Sale_Ratio
                                        , self.Round_Statistic(self.DOR_COD(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column]))
                                        , self.Round_Statistic(self.DOR_PRD(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column]), scale = 100)
                                        , self.Round_Statistic(self.DOR_weighted_Mean(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column]), scale = 100)]
        self.GroupLevelStatistics.loc[2] = [self.Stratum_Number
                                        , 3
                                        , G3S.shape[0]
                                        , Group_3_Median_Sale_Ratio
                                        , Group_3_Mean_Sale_Ratio
                                        , self.Round_Statistic(self.DOR_COD(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column]))
                                        , self.Round_Statistic(self.DOR_PRD(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column]), scale = 100)
                                        , self.Round_Statistic(self.DOR_weighted_Mean(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column]), scale = 100)]
        self.GroupLevelStatistics.loc[3] = [self.Stratum_Number
                                        , 4
                                        , G4S.shape[0]
                                        , Group_4_Median_Sale_Ratio
                                        , Group_4_Mean_Sale_Ratio
                                        , self.Round_Statistic(self.DOR_COD(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column]))
                                        , self.Round_Statistic(self.DOR_PRD(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column]), scale = 100)
                                        , self.Round_Statistic(self.DOR_weighted_Mean(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column]), scale = 100)]
       
        self.GroupLevelStatistics.loc[4] = [self.Stratum_Number
                                        , 'Total'
                                        , self.StratumSales_trimmed.shape[0]
                                        , round(((self.StratumSales_trimmed[self.NAL_Value_Column]/self.StratumSales_trimmed[self.Sale_Value_Column]).median())*100,2)
                                        , round(((self.StratumSales_trimmed[self.NAL_Value_Column]/self.StratumSales_trimmed[self.Sale_Value_Column]).mean())*100,2)
                                        , self.Round_Statistic(self.DOR_COD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]))
                                        , self.Round_Statistic(self.DOR_PRD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]), scale = 100)
                                        , self.Round_Statistic(self.DOR_weighted_Mean(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]), scale = 100)]
       

        ## Fill out each row manually
        ## Every cell in the _Text frame is a STRING - it is the printable
        ## version of the report. Two of these rows used to write the group
        ## number as a bare int while the others stringified it, which left
        ## the column holding ['1', '2', 3, 4, 'Total'].
        ##
        ## Streamlit could not convert that to Arrow, logged a full
        ## ArrowTypeError traceback to the terminal, then recovered by
        ## coercing the types - so the table DID render and the only cost
        ## was a traceback that looks fatal and is not. Still worth not
        ## printing: a reader who sees it reasonably assumes something broke.
        self.GroupLevelStatistics_Text.loc[0] = [self.Stratum_Number
                                                    , str(1)
                                                    , self.Format_My_Numbers(G1S.shape[0] , type = 'int')
                                                    , str(Group_1_Median_Sale_Ratio)
                                                    , str(Group_1_Mean_Sale_Ratio)
                                                    , self.Format_Statistic(self.DOR_COD(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column]))
                                                    , self.Format_Statistic(self.DOR_PRD(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column]), scale = 100)
                                                    , self.Format_Statistic(self.DOR_weighted_Mean(G1S[self.Sale_Value_Column],G1S[self.NAL_Value_Column]), scale = 100)]
        self.GroupLevelStatistics_Text.loc[1] = [self.Stratum_Number
                                                    , str(2)
                                                    , self.Format_My_Numbers(G2S.shape[0], type = 'int')
                                                    , str(Group_2_Median_Sale_Ratio)
                                                    , str(Group_2_Mean_Sale_Ratio)
                                                    , self.Format_Statistic(self.DOR_COD(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column]))
                                                    , self.Format_Statistic(self.DOR_PRD(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column]), scale = 100)
                                                    , self.Format_Statistic(self.DOR_weighted_Mean(G2S[self.Sale_Value_Column],G2S[self.NAL_Value_Column]), scale = 100)]
        self.GroupLevelStatistics_Text.loc[2] = [self.Stratum_Number
                                                    , str(3)
                                                    , self.Format_My_Numbers(G3S.shape[0], type = 'int')
                                                    , str(Group_3_Median_Sale_Ratio)
                                                    , str(Group_3_Mean_Sale_Ratio)
                                                    , self.Format_Statistic(self.DOR_COD(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column]))
                                                    , self.Format_Statistic(self.DOR_PRD(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column]), scale = 100)
                                                    , self.Format_Statistic(self.DOR_weighted_Mean(G3S[self.Sale_Value_Column],G3S[self.NAL_Value_Column]), scale = 100)]
        self.GroupLevelStatistics_Text.loc[3] = [self.Stratum_Number
                                                    , str(4)
                                                    , self.Format_My_Numbers(G4S.shape[0], type = 'int')
                                                    , str(Group_4_Median_Sale_Ratio)
                                                    , str(Group_4_Mean_Sale_Ratio)
                                                    , self.Format_Statistic(self.DOR_COD(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column]))
                                                    , self.Format_Statistic(self.DOR_PRD(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column]), scale = 100)
                                                    , self.Format_Statistic(self.DOR_weighted_Mean(G4S[self.Sale_Value_Column],G4S[self.NAL_Value_Column]), scale = 100)]
        self.GroupLevelStatistics_Text.loc[4] = [self.Stratum_Number
                                                    , 'Total'
                                                    , self.Format_My_Numbers(self.StratumSales_trimmed.shape[0], type = 'int')
                                                    , str(round(((self.StratumSales_trimmed[self.NAL_Value_Column]/self.StratumSales_trimmed[self.Sale_Value_Column]).median())*100,2))
                                                    , str(round(((self.StratumSales_trimmed[self.NAL_Value_Column]/self.StratumSales_trimmed[self.Sale_Value_Column]).mean())*100,2))
                                                    , self.Format_Statistic(self.DOR_COD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]))
                                                    , self.Format_Statistic(self.DOR_PRD(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]), scale = 100)
                                                    , self.Format_Statistic(self.DOR_weighted_Mean(self.StratumSales_trimmed[self.Sale_Value_Column],self.StratumSales_trimmed[self.NAL_Value_Column]), scale = 100)]

        if self.Return_Type.lower() == 'text':
            return self.Stratum_Text , self.GroupLevelStatistics_Text
        else:
            return self.Stratum , self.GroupLevelStatistics


### GUI Functions ###
def render_animation():
    #animation_response = requests.get('https://assets1.lottiefiles.com/packages/lf20_vykpwvdsvsdvsdt8b.json')
    animation_response = requests.get('https://assets1.lottiefiles.com/packages/lf20_vykpwt8b.json', timeout = 3)
    animation_json = dict()
    
    if animation_response.status_code == 200:
        animation_json = animation_response.json()
    else:
        ## 🩹 This used to be a bare `1/0`, raising ZeroDivisionError to
        ## signal a failed download. The call sites catch it, so the app
        ## degraded rather than died - but a deliberate ZeroDivisionError
        ## in a traceback tells whoever reads it nothing at all.
        ##
        ## This matters on a county laptop: these animations are fetched
        ## from the internet, and behind a corporate proxy the fetch fails
        ## every time. A decoration must never be able to stop the work.
        raise RuntimeError(
            f"Could not fetch the loading animation "
            f"(HTTP {animation_response.status_code}). This is decoration "
            f"only - the app works without it."
        )
                           
    return st_lottie(animation_json,height=300,width=400)

def render_animation2():
    animation_response = requests.get('https://lottie.host/864a3252-e826-4a65-ab0d-7fe545b55100/GKjQEfQguG.json', timeout = 3)
    animation_json = dict()
    
    if animation_response.status_code == 200:
        animation_json = animation_response.json()
    else:
        ## See render_animation above - same reasoning.
        raise RuntimeError(
            f"Could not fetch the chat animation "
            f"(HTTP {animation_response.status_code}). Decoration only."
        )
                           
    return st_lottie(animation_json,height=300,width=400)

@st.cache_data
def convert_df(df):
    """Turn a results frame into CSV bytes for the download button.

    🚫 Every download in the app goes through here, which makes it the one
    place that can label a number on its way out of the building.

    That matters more than the on-screen wording. A spreadsheet gets
    emailed, pasted into another workbook, and opened months later by
    somebody who never saw this app - and not one pixel of the caveat on
    the results screen travels with it. The column HEADER does.

    So `Model_Prediction` becomes `Benchmark_Estimate`. It is the only
    part of the warning that survives being copied into a different file.

    📌 Why not a text column repeating the full caveat on every row:
    `All_Model_Predictions.csv` has 268,269 rows, and an 80-character
    sentence on each is 21 MB of identical text. The header carries it for
    nothing.
    """
    labelled = df.rename(columns = {
        "Model_Prediction": BENCHMARK_COLUMN,
        "Model_Value": BENCHMARK_COLUMN,
    })
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return labelled.to_csv(index = False).encode("utf-8")


def Password_Entry(u,p):
    global User_Password_List
    p_entry = [u,p]

    if p_entry in User_Password_List:
        st.session_state['User_Name_Entry'] = u
        st.session_state['PSWRD_Entry'] = p

    st.session_state['Password_Count'] = st.session_state['Password_Count'] + 1




### Data Manipulation Functions ###

def Merge_data(NAL_Data , SDF_Data , GIS_Data):
    ## Both frames are already loaded by the time we get here - the caller
    ## is gated on it. There used to be an `else` branch that read a roll
    ## off disk using the 67-county path dictionary; it is gone with the
    ## dictionary, along with the `using_last_NAL` flag it set.
    SDF = SDF_Data.copy()
    NAL = NAL_Data.copy()

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

        ## ── One shape per parcel, before anything joins on it ───────────
        ##
        ## A county shapefile is not guaranteed to hold exactly one polygon
        ## per parcel. Measured on the bundled Lee County file:
        ##
        ##     total shapes                      559,141
        ##     shapes with a NULL PARCEL_ID        2,915
        ##     parcels owning more than one shape     11   -> 17 extra rows
        ##
        ## 🧨 Both matter, because this frame is INNER JOINED to the sales.
        ## A parcel with five polygons joins five times, so its sale shows
        ## up five times in the data the model learns from - and a
        ## duplicated sale is a sale counted twice.
        ##
        ## It is small on this roll. It is a property of the COUNTY'S file,
        ## though, not of our code, so another roll could be far worse and
        ## nothing here would have told you.
        ##
        ## 📌 Which polygon to keep, for a parcel that has several? The
        ## biggest one. We want a point that says where the property IS,
        ## and the largest polygon is the main parcel rather than a sliver
        ## left by a survey correction or an excluded easement. Keeping
        ## whichever happened to be first in the file would be arbitrary.
        GIS_Data = GIS_Data[GIS_Data['PARCEL_ID'].notna()].copy()

        _shapes_before = len(GIS_Data)

        GIS_Data['_shape_area'] = GIS_Data['geometry'].area
        GIS_Data = (GIS_Data
                    .sort_values('_shape_area', ascending = False)
                    .drop_duplicates(subset = 'PARCEL_ID', keep = 'first')
                    .drop(columns = '_shape_area'))

        if _shapes_before != len(GIS_Data):
            print(' - GIS: %d shapes -> %d, one per parcel (largest kept)'
                  % (_shapes_before, len(GIS_Data)))

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

        ## 🩹 Swapping neighbourhood code for coordinates, done so it can
        ## run twice.
        ##
        ## This used to be, for each of the four lists:
        ##
        ##     st.session_state[k] = st.session_state[k] + ['Latitude', 'Longitude']
        ##     st.session_state[k].remove('NBRHD_CD')
        ##
        ## Press "Merge and Preprocess" a second time and the `.remove()`
        ## raises ValueError: list.remove(x): x not in list, because
        ## NBRHD_CD went the first time - leaving the list holding
        ## ['Latitude', 'Longitude', 'Latitude', 'Longitude']. A reader who
        ## clicks a button twice should not have to restart the app.
        ##
        ## Why swap at all: with real coordinates the model can learn
        ## location directly, which is strictly better than a neighbourhood
        ## code standing in for it. Keeping both would hand the model two
        ## versions of the same thing.
        for _feature_list in ('Model_Features'
                              , 'Model_Features_Condos'
                              , 'Model_Features_Vac_Res'
                              , 'Model_Features_Commercial'):
            _features = list(st.session_state[_feature_list])

            for _coordinate in ('Latitude', 'Longitude'):
                if _coordinate not in _features:
                    _features.append(_coordinate)

            if 'NBRHD_CD' in _features:
                _features.remove('NBRHD_CD')

            st.session_state[_feature_list] = _features

        ## 🩹 This function writes its own result back into
        ## session_state["NAL"], so a SECOND press of "Merge and
        ## Preprocess" hands it a frame that already has coordinates.
        ## Merging again would produce Latitude_x and Latitude_y, and the
        ## model build would then fail with a KeyError on 'Latitude'.
        ##
        ## Dropping them first makes a re-run land in exactly the same
        ## place as the first run. `errors = 'ignore'` because on the first
        ## run there is nothing to drop.
        All_NAL_Data = (st.session_state["NAL"]
                        .copy()
                        .drop(columns = ['Latitude', 'Longitude']
                              , errors = 'ignore'))

        All_NAL_Data_With_GIS = pd.merge(All_NAL_Data
                                        , GIS_Data_Clean[['PARCEL_ID','Latitude', 'Longitude']]
                                        , on = 'PARCEL_ID'
                                        , how = 'inner').copy()

        st.session_state["NAL"] = All_NAL_Data_With_GIS.copy()
    

    ## A block here used to drop rows whose use code and vacant/improved
    ## flag disagreed - a parcel coded vacant that sold as improved, and
    ## the reverse. It ran ONLY when the roll had been read off disk from
    ## the old county dictionary, i.e. only for last year's file, so it
    ## never ran for an uploaded roll. It is gone with that branch.
    ##
    ## 📌 Worth revisiting on its own merits: that disagreement is a real
    ## data-quality signal on any roll, not only an old one. But it was
    ## never applied to the data a reader actually loads, so turning it on
    ## for everyone would be a NEW filter, not a preserved one - and it
    ## would silently change every number in the app.

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

    ## Writing back into session_state["SDF"] is deliberate: the DOR
    ## summary report needs the adjusted sale price, and it reads the SDF
    ## rather than the merged frame.
    ##
    ## Safe to re-run: this OVERWRITES the column from SALE_PRC each time
    ## rather than adjusting what is already there, so pressing the button
    ## twice does not apply the adjustment twice.
    st.session_state["SDF"]['Adjusted_Sale_Price']= st.session_state["SDF"].apply(lambda row : round(row['SALE_PRC'] * SDF_Fix_Dict[row['PARCEL_ID']],0) , axis = 1)

    
    return NAL_merged
        
    

def Score_Model_Predictions(performance_metric, actual_values, predicted_values):
    """Turn predictions into a single number, in one place.

    ⚠️ Two things this function exists to get right.

    **COD's argument order matters.** `COD(sale_price, appraised_value)` is
    NOT symmetric - swapping the two changes the answer (measured: 14.53
    against 15.59 on the same data). Writing the order down once beats
    remembering it at four call sites.

    **R squared is returned NEGATED.** The Optuna study minimises, and a
    higher R squared is better - so minimising it unnegated would search
    for the WORST model. Nothing currently asks for 'r2', which is exactly
    why it would have gone unnoticed.
    """
    metric = performance_metric.lower().strip()

    if metric == 'rmse':
        return np.sqrt(mean_squared_error(actual_values, predicted_values))
    if metric == 'mae':
        return mean_absolute_error(actual_values, predicted_values)
    if metric == 'cod':
        return COD(actual_values, predicted_values)
    if metric == 'r2':
        return -r2_score(actual_values, predicted_values)

    raise ValueError(
        f"Unknown performance metric {performance_metric!r}. "
        f"Use one of: RMSE, MAE, COD, R2."
    )


def Model_Objective(trial
                    , Model_Train_Data
                    , model_features
                    , performance_metric = 'RMSE'
                    , n_folds = 5):
    """Score ONE set of hyperparameters, honestly.

    Optuna calls this once per trial. It has to answer: how good is this
    combination of settings? The answer decides which settings get used,
    so how it is measured matters more than almost anything else here.

    🛑 What this used to do, and why it was wrong
    -------------------------------------------
    It fitted on the training split and scored on the TEST split. Then the
    finished model's "test" performance was reported from that same split.

    That is **hyperparameter leakage**. Run 150 trials against a test set
    and you are no longer measuring how the model does on data it has not
    seen - you are picking whichever settings happen to suit those 5,000
    particular sales. The reported COD comes out flattering, the reader
    believes it, and the first time the model meets a genuinely new year
    of sales it does worse than advertised. In appraisal that is not an
    academic problem: somebody's value is in that number.

    ✅ What it does now
    ------------------
    **The test split is never touched during tuning.** Each trial splits
    the TRAINING data into `n_folds` parts, then for each part: fit on the
    others, score on this one. The trial's score is the average. So every
    score comes from data that particular fit had not seen, and the test
    split stays sealed until the final model is scored once.

    It costs `n_folds` fits per trial instead of one. Measured on the real
    SFR training set, 5 folds, 24-core machine: 50 trials took 192 s and 150
    took 640 s. A laptop will take longer.

    🩹 This used to say 150 trials was roughly 4 minutes. That came from random
    parameter draws, and TPE settles on the expensive parameters, so the real
    cost was three times that. The default is 50 now, for that reason.

    📌 An open question, deliberately left open
    ------------------------------------------
    The folds are stratified by **sale year**, matching how the app's own
    train/test split works, so each fold sees a similar mix of years.

    A valuation model predicts FORWARD, though, and a stricter design
    would train on earlier sales and validate on later ones. That is a
    different and arguably better choice, and it would change the headline
    numbers - so it is not being made quietly here. See CLAUDE.md section 7.
    """


  
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



    ## ── Choose how to cut the training data into folds ──────────────────
    ##
    ## A fold needs at least one row of every sale year to stratify on it.
    ## Residential rolls have thousands per year; a Commercial model on a
    ## small county might have three sales in 2021. So check, and fall back
    ## to plain random folds rather than raising at the reader.
    n_folds = int(n_folds)
    n_folds = max(2, min(n_folds, len(Model_Train_Data) // 2))

    fold_strata = Model_Train_Data[Stratification_Variable]

    if fold_strata.value_counts().min() >= n_folds:
        fold_splitter = StratifiedKFold(n_splits = n_folds
                                        , shuffle = True
                                        , random_state = 42)
        fold_indices = fold_splitter.split(Model_Train_Data, fold_strata)
    else:
        fold_splitter = KFold(n_splits = n_folds
                              , shuffle = True
                              , random_state = 42)
        fold_indices = fold_splitter.split(Model_Train_Data)

    ## ── Fit once per fold, scoring on the part left out ─────────────────
    fold_scores = []

    for fold_number, (fold_train_rows, fold_val_rows) in enumerate(fold_indices):
        fold_train = Model_Train_Data.iloc[fold_train_rows]
        fold_validation = Model_Train_Data.iloc[fold_val_rows]

        fold_model = ltb.LGBMRegressor(**Opt_model_params)
        fold_model.fit(fold_train[model_features]
                       , fold_train[Model_Outcome[0]])

        fold_scores.append(Score_Model_Predictions(
            performance_metric
            , fold_validation[Model_Outcome[0]].values.ravel()
            , fold_model.predict(fold_validation[model_features])
        ))

        ## Tell Optuna how this trial is going so far. If the running
        ## average is already worse than the median trial at this point,
        ## there is no sense fitting the remaining folds - the pruner says
        ## so and we stop. On a wide search space most trials are poor, so
        ## this is where a good chunk of the k-fold cost comes back.
        trial.report(float(np.mean(fold_scores)), fold_number)

        if trial.should_prune():
            raise optuna.TrialPruned()

    ## The average across folds. One fold getting lucky cannot carry a
    ## trial, which is the whole point of doing it this way.
    return float(np.mean(fold_scores))


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
            ## 🎲 A seeded sampler, so the search is REPRODUCIBLE.
            ## Without `seed`, Optuna explores differently every run and
            ## two people following the same steps get different models -
            ## which makes any number either of them quotes unverifiable.
            ##
            ## ✂️ The pruner abandons a trial once its running score is
            ## worse than the median trial at the same fold. It is what
            ## pays for cross-validation: hopeless trials cost one or two
            ## fits instead of all of them. `n_startup_trials` lets the
            ## first 10 finish, so there is something to take a median of.
            LGBM_study = optuna.create_study(
                direction = 'minimize'
                , sampler = optuna.samplers.TPESampler(seed = 42)
                , pruner = optuna.pruners.MedianPruner(n_startup_trials = 10
                                                       , n_warmup_steps = 1)
            )
            ## No test data is passed any more - that is the point. The
            ## objective cross-validates within the training split.
            LGBM_study_obj_func = lambda trial: Model_Objective(trial
                                                    , Model_Train_Data = NAL_Data_Train_SFR
                                                    , model_features = st.session_state['Model_Features']
                                                    , performance_metric = 'COD'
                                                    , n_folds = st.session_state['Hyperparameter_Tuning_Folds'])
            
            LGBM_study.optimize(LGBM_study_obj_func
                                , n_trials = st.session_state['Hyperparameter_Tuning_Trials'])
            
            LGBM_Best_Params = LGBM_study.best_params.copy()
            LGBM_Best_Params = {**LGBM_Best_Params, **{'random_state': 42}}
            #LGBM_Best_Params = {**LGBM_Best_Params, **{'n_estimators': 1000}}
            LGBM_Best_Params = {**LGBM_Best_Params, **{'verbose': -1}}

            NAL_Model = ltb.LGBMRegressor(**LGBM_Best_Params)

            ## 🛑 No `eval_set` on the test split here, deliberately.
            ##
            ## It used to early-stop against the test data: train until the
            ## test score stops improving. That reads like good practice and
            ## is the same leak as before in a quieter form - the test split
            ## decides how many trees the model keeps, so the test score it
            ## then reports is not a clean measurement.
            ##
            ## `n_estimators` comes from the cross-validated search instead,
            ## which chose it without ever seeing the test data. It also
            ## clears a deprecation: `eval_set` is on its way out of
            ## LightGBM in favour of `eval_X` / `eval_y`.
            NAL_Model.fit(NAL_Data_Train_SFR_Features
                          , NAL_Data_Train_SFR[Model_Outcome[0]])
            

        NAL_Model_Train_SFR_results = NAL_Data_Train_SFR.copy()

        NAL_Model_Train_SFR_results['Model_Prediction'] = NAL_Model.predict(NAL_Data_Train_SFR_Features)

        NAL_Model_Train_SFR_results['sale_ratio'] = NAL_Model_Train_SFR_results['Model_Prediction'] / NAL_Model_Train_SFR_results[Model_Outcome[0]] 

        NAL_Model_Train_SFR_results_Trimmed = NAL_Model_Train_SFR_results[(NAL_Model_Train_SFR_results['sale_ratio'] > .3) & (NAL_Model_Train_SFR_results['sale_ratio'] < 1.7)]

        #print('Number of Records Before Trimming: ', NAL_Model_Train_SFR_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Train_SFR_results_Trimmed.shape[0])

        NAL_Model_Test_SFR_results = NAL_Data_Test_SFR.copy()

        NAL_Model_Test_SFR_results['Model_Prediction'] = NAL_Model.predict(NAL_Data_Test_SFR_Features)

        NAL_Model_Test_SFR_results['sale_ratio'] = NAL_Model_Test_SFR_results['Model_Prediction'] / NAL_Model_Test_SFR_results[Model_Outcome[0]] 

        ## ── The county's own ratio, on the SAME footing as the model's ──
        ##
        ## A sale ratio is  value / sale price. Both numbers have to mean
        ## the same thing, or the comparison is meaningless.
        ##
        ## `Adjusted_Sale_Price` is the sale price with the DR-493
        ## cost-of-sale adjustment applied - what Florida compares a just
        ## value against. `JV` is already the county's just value. So the
        ## ratio is simply one divided by the other.
        ##
        ## 🩹 This line used to read (JV / 0.85) / Adjusted_Sale_Price,
        ## which applied the cost-of-sale adjustment TWICE in opposite
        ## directions - once dividing JV up, once multiplying the sale
        ## down. Measured on 25,491 qualified improved sales:
        ##
        ##     (JV/0.85) / (SALE*0.85)   median 1.1090   <- the old line
        ##      JV       / (SALE*0.85)   median 0.9442   <- correct
        ##
        ## The county looked as though it was over-assessing, by about 11%, when
        ## it sits about 6% below its sale prices. ✅ COD is
        ## scale-free, so the UNIFORMITY numbers never moved - only the
        ## LEVEL ones, which is the half an assessor is judged on.
        ##
        ## 📌 And because it no longer hardcodes 0.85, it follows whatever
        ## per-DOR-code adjustment you set in the sidebar. Before, changing
        ## an adjustment moved the model's ratio and left the county's behind.
        NAL_Model_Test_SFR_results['sale_ratio_JV'] = NAL_Model_Test_SFR_results['JV'] / NAL_Model_Test_SFR_results[Model_Outcome[0]] 

        NAL_Model_Test_results_Trimmed = NAL_Model_Test_SFR_results[(NAL_Model_Test_SFR_results['sale_ratio'] > .3) & (NAL_Model_Test_SFR_results['sale_ratio'] < 1.7)]

        ## ── Phase 4.4: ONE row set for both sides of the comparison ─────────
        ##
        ## 🩹 This used to keep only the rows whose COUNTY ratio was in range,
        ## while the model's metrics came from a frame keeping the rows whose
        ## MODEL ratio was in range. Two different row sets - and the app then
        ## drew them side by side in a chart labelled "Model" vs "NAL Values".
        ##
        ## That flatters both. Trimming on the model's ratio drops the sales the
        ## MODEL did worst on; trimming on the county's drops the sales the
        ## COUNTY did worst on. Each side was scored on the subset kindest to
        ## it, and a reader was invited to subtract one from the other.
        ##
        ## Now: the rows where BOTH ratios are in range. Same rows, same count,
        ## and the model's own numbers are computed on that set too, so the
        ## comparison is like-for-like on both axes.
        ##
        ## 📌 The `Trimmed Test *` keys are untouched. The model's own trimmed
        ## performance is a fair standalone figure - it just is not a comparison.
        _model_ratio_in_range = ((NAL_Model_Test_SFR_results['sale_ratio'] > .3)
                                 & (NAL_Model_Test_SFR_results['sale_ratio'] < 1.7))
        _county_ratio_in_range = ((NAL_Model_Test_SFR_results['sale_ratio_JV'] > .3)
                                  & (NAL_Model_Test_SFR_results['sale_ratio_JV'] < 1.7))
        
        NAL_Model_Test_results_Comparable = NAL_Model_Test_SFR_results[_model_ratio_in_range & _county_ratio_in_range]

        ## ── Keep these sales for the chat assistant's ratio study ───────────
        ##
        ## 🛑 Both frames, and deliberately not `SFR_Model_Predicitons_All`.
        ## Those are predictions at the VALUATION date, while these are at each
        ## sale's own date - the numbers this tab reports. And
        ## build_sales_rows leaves the model's value EMPTY on training sales,
        ## so nothing can score the model on what it learned from. Measured:
        ## doing it the obvious way gives a COD of 9.98 against an honest 11.12,
        ## which flatters the model by more than a full point.
        st.session_state["Sales_For_Chat"]["Single Family"] = build_sales_rows(
            "Single Family", NAL_Model_Train_SFR_results, NAL_Model_Test_SFR_results)
        
        ## Pulled out once, so the metric lines below read as comparisons
        ## rather than as a wall of indexing.
        _cmp_sale_price = NAL_Model_Test_results_Comparable[Model_Outcome[0]].values.ravel()
        _cmp_model_value = NAL_Model_Test_results_Comparable['Model_Prediction'].values.ravel()
        _cmp_county_value = NAL_Model_Test_results_Comparable['JV'].values.ravel()

        #print(NAL_Model_Test_results_Trimmed.info())

        #print('Number of Records Before Trimming: ', NAL_Model_Test_SFR_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Test_results_Trimmed.shape[0])

        NAL_Model_Data_SFR_For_Preds = st.session_state["NAL"].copy()

        if st.session_state['NBHD_Encoder'] is not None:
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

        Pop_Data_SFR_For_Review['Model_Value_Pct_Diff_Just_Value'] = round((Pop_Data_SFR_For_Review['Model_Prediction'] - (Pop_Data_SFR_For_Review['JV'])) / (Pop_Data_SFR_For_Review['JV']) ,3)


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






                                                    ## ── The head-to-head, on identical rows ─────────────────────────
                                                    ## Both sides measured on the same sales. See the note where
                                                    ## _Comparable is built.
                                                    , 'Comparable Records' : NAL_Model_Test_results_Comparable.shape[0]

                                                    , 'Head to Head Model R Squared' : round(r2_score(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model Weighted Mean' : round(weighted_Mean_Sale_Ratio(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model COD' : round(COD(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model PRD' : round(PRD(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model MAE' : round(mean_absolute_error(_cmp_sale_price, _cmp_model_value), Result_Prescision)

                                                    , 'Head to Head County R Squared' : round(r2_score(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County Weighted Mean' : round(weighted_Mean_Sale_Ratio(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County COD' : round(COD(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County PRD' : round(PRD(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County MAE' : round(mean_absolute_error(_cmp_sale_price, _cmp_county_value), Result_Prescision)

                                                    
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
            ## 🎲 A seeded sampler, so the search is REPRODUCIBLE.
            ## Without `seed`, Optuna explores differently every run and
            ## two people following the same steps get different models -
            ## which makes any number either of them quotes unverifiable.
            ##
            ## ✂️ The pruner abandons a trial once its running score is
            ## worse than the median trial at the same fold. It is what
            ## pays for cross-validation: hopeless trials cost one or two
            ## fits instead of all of them. `n_startup_trials` lets the
            ## first 10 finish, so there is something to take a median of.
            LGBM_study = optuna.create_study(
                direction = 'minimize'
                , sampler = optuna.samplers.TPESampler(seed = 42)
                , pruner = optuna.pruners.MedianPruner(n_startup_trials = 10
                                                       , n_warmup_steps = 1)
            )
            ## No test data is passed any more - the objective
            ## cross-validates within the training split. That also
            ## retires the bug this branch used to have, where the FULL
            ## qualified-sales set was handed over as the test split.
            LGBM_study_obj_func = lambda trial: Model_Objective(trial
                                                    , Model_Train_Data = NAL_Data_Train_Vac_Res
                                                    , model_features = st.session_state['Model_Features_Vac_Res']
                                                    , performance_metric = 'COD'
                                                    , n_folds = st.session_state['Hyperparameter_Tuning_Folds'])
            
            LGBM_study.optimize(LGBM_study_obj_func
                                , n_trials = st.session_state['Hyperparameter_Tuning_Trials'])
            
            LGBM_Best_Params = LGBM_study.best_params.copy()
            LGBM_Best_Params = {**LGBM_Best_Params, **{'random_state': 42}}
            #LGBM_Best_Params = {**LGBM_Best_Params, **{'n_estimators': 1000}}
            LGBM_Best_Params = {**LGBM_Best_Params, **{'verbose': -1}}

            NAL_Model_Vac_Res = ltb.LGBMRegressor(**LGBM_Best_Params)
            ## See the SFR fit above for why there is no eval_set here.
            ## This branch already had it commented out.
            NAL_Model_Vac_Res.fit(NAL_Data_Train_Vac_Res_Features
                                  , NAL_Data_Train_Vac_Res[Model_Outcome[0]])
            

        

        NAL_Model_Train_Vac_Res_results = NAL_Data_Train_Vac_Res.copy()

        NAL_Model_Train_Vac_Res_results['Model_Prediction'] = NAL_Model_Vac_Res.predict(NAL_Data_Train_Vac_Res_Features)

        NAL_Model_Train_Vac_Res_results['sale_ratio'] = NAL_Model_Train_Vac_Res_results['Model_Prediction'] / NAL_Model_Train_Vac_Res_results[Model_Outcome[0]] 

        NAL_Model_Train_Vac_Res_results_Trimmed = NAL_Model_Train_Vac_Res_results[(NAL_Model_Train_Vac_Res_results['sale_ratio'] > .3) & (NAL_Model_Train_Vac_Res_results['sale_ratio'] < 1.7)]

        #print('Number of Records Before Trimming: ', NAL_Model_Train_Vac_Res_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Train_Vac_Res_results_Trimmed.shape[0])

        NAL_Model_Test_Vac_Res_results = NAL_Data_Test_Vac_Res.copy()

        NAL_Model_Test_Vac_Res_results['Model_Prediction'] = NAL_Model_Vac_Res.predict(NAL_Data_Test_Vac_Res_Features)

        NAL_Model_Test_Vac_Res_results['sale_ratio'] = NAL_Model_Test_Vac_Res_results['Model_Prediction'] / NAL_Model_Test_Vac_Res_results[Model_Outcome[0]] 

        NAL_Model_Test_Vac_Res_results['sale_ratio_JV'] = NAL_Model_Test_Vac_Res_results['JV'] / NAL_Model_Test_Vac_Res_results[Model_Outcome[0]] 

        NAL_Model_Test_Vac_Res_results_Trimmed = NAL_Model_Test_Vac_Res_results[(NAL_Model_Test_Vac_Res_results['sale_ratio'] > .3) & (NAL_Model_Test_Vac_Res_results['sale_ratio'] < 1.7)]

        ## ── Phase 4.4: ONE row set for both sides of the comparison ─────────
        ##
        ## 🩹 This used to keep only the rows whose COUNTY ratio was in range,
        ## while the model's metrics came from a frame keeping the rows whose
        ## MODEL ratio was in range. Two different row sets - and the app then
        ## drew them side by side in a chart labelled "Model" vs "NAL Values".
        ##
        ## That flatters both. Trimming on the model's ratio drops the sales the
        ## MODEL did worst on; trimming on the county's drops the sales the
        ## COUNTY did worst on. Each side was scored on the subset kindest to
        ## it, and a reader was invited to subtract one from the other.
        ##
        ## Now: the rows where BOTH ratios are in range. Same rows, same count,
        ## and the model's own numbers are computed on that set too, so the
        ## comparison is like-for-like on both axes.
        ##
        ## 📌 The `Trimmed Test *` keys are untouched. The model's own trimmed
        ## performance is a fair standalone figure - it just is not a comparison.
        _model_ratio_in_range = ((NAL_Model_Test_Vac_Res_results['sale_ratio'] > .3)
                                 & (NAL_Model_Test_Vac_Res_results['sale_ratio'] < 1.7))
        _county_ratio_in_range = ((NAL_Model_Test_Vac_Res_results['sale_ratio_JV'] > .3)
                                  & (NAL_Model_Test_Vac_Res_results['sale_ratio_JV'] < 1.7))
        
        NAL_Model_Test_Vac_Res_results_Comparable = NAL_Model_Test_Vac_Res_results[_model_ratio_in_range & _county_ratio_in_range]

        ## See the Single Family block for why this is not the all-parcel frame.
        st.session_state["Sales_For_Chat"]["Vacant Residential"] = build_sales_rows(
            "Vacant Residential", NAL_Model_Train_Vac_Res_results, NAL_Model_Test_Vac_Res_results)
        
        ## Pulled out once, so the metric lines below read as comparisons
        ## rather than as a wall of indexing.
        _cmp_sale_price = NAL_Model_Test_Vac_Res_results_Comparable[Model_Outcome[0]].values.ravel()
        _cmp_model_value = NAL_Model_Test_Vac_Res_results_Comparable['Model_Prediction'].values.ravel()
        _cmp_county_value = NAL_Model_Test_Vac_Res_results_Comparable['JV'].values.ravel()

        #print(NAL_Model_Test_results_Trimmed.info())

        #print('Number of Records Before Trimming: ', NAL_Model_Test_Vac_Res_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Test_Vac_Res_results_Trimmed.shape[0])

        NAL_Model_Data_Vac_Res_For_Preds = st.session_state["NAL"].copy()

        if st.session_state['NBHD_Encoder'] is not None:
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






                                                    ## ── The head-to-head, on identical rows ─────────────────────────
                                                    ## Both sides measured on the same sales. See the note where
                                                    ## _Comparable is built.
                                                    , 'Comparable Records' : NAL_Model_Test_Vac_Res_results_Comparable.shape[0]

                                                    , 'Head to Head Model R Squared' : round(r2_score(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model Weighted Mean' : round(weighted_Mean_Sale_Ratio(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model COD' : round(COD(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model PRD' : round(PRD(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model MAE' : round(mean_absolute_error(_cmp_sale_price, _cmp_model_value), Result_Prescision)

                                                    , 'Head to Head County R Squared' : round(r2_score(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County Weighted Mean' : round(weighted_Mean_Sale_Ratio(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County COD' : round(COD(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County PRD' : round(PRD(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County MAE' : round(mean_absolute_error(_cmp_sale_price, _cmp_county_value), Result_Prescision)

                                                    
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
            ## 🎲 A seeded sampler, so the search is REPRODUCIBLE.
            ## Without `seed`, Optuna explores differently every run and
            ## two people following the same steps get different models -
            ## which makes any number either of them quotes unverifiable.
            ##
            ## ✂️ The pruner abandons a trial once its running score is
            ## worse than the median trial at the same fold. It is what
            ## pays for cross-validation: hopeless trials cost one or two
            ## fits instead of all of them. `n_startup_trials` lets the
            ## first 10 finish, so there is something to take a median of.
            LGBM_study = optuna.create_study(
                direction = 'minimize'
                , sampler = optuna.samplers.TPESampler(seed = 42)
                , pruner = optuna.pruners.MedianPruner(n_startup_trials = 10
                                                       , n_warmup_steps = 1)
            )
            ## No test data is passed any more - that is the point. The
            ## objective cross-validates within the training split.
            LGBM_study_obj_func = lambda trial: Model_Objective(trial
                                                    , Model_Train_Data = NAL_Data_Train_Condo
                                                    , model_features = st.session_state['Model_Features_Condos']
                                                    , performance_metric = 'COD'
                                                    , n_folds = st.session_state['Hyperparameter_Tuning_Folds'])
            
            LGBM_study.optimize(LGBM_study_obj_func
                                , n_trials = st.session_state['Hyperparameter_Tuning_Trials'])
            
            LGBM_Best_Params = LGBM_study.best_params.copy()
            LGBM_Best_Params = {**LGBM_Best_Params, **{'random_state': 42}}
            #LGBM_Best_Params = {**LGBM_Best_Params, **{'n_estimators': 1000}}
            LGBM_Best_Params = {**LGBM_Best_Params, **{'verbose': -1}}

            NAL_Model_Condo = ltb.LGBMRegressor(**LGBM_Best_Params)
            

            ## 🛑 No `eval_set` on the test split here, deliberately.
            ##
            ## It used to early-stop against the test data: train until the
            ## test score stops improving. That reads like good practice and
            ## is the same leak as before in a quieter form - the test split
            ## decides how many trees the model keeps, so the test score it
            ## then reports is not a clean measurement.
            ##
            ## `n_estimators` comes from the cross-validated search instead,
            ## which chose it without ever seeing the test data. It also
            ## clears a deprecation: `eval_set` is on its way out of
            ## LightGBM in favour of `eval_X` / `eval_y`.
            NAL_Model_Condo.fit(NAL_Data_Train_Condo_Features
                                , NAL_Data_Train_Condo[Model_Outcome[0]])
        

        NAL_Model_Train_Condo_results = NAL_Data_Train_Condo.copy()

        NAL_Model_Train_Condo_results['Model_Prediction'] = NAL_Model_Condo.predict(NAL_Data_Train_Condo_Features)

        NAL_Model_Train_Condo_results['sale_ratio'] = NAL_Model_Train_Condo_results['Model_Prediction'] / NAL_Model_Train_Condo_results[Model_Outcome[0]] 


        NAL_Model_Train_Condo_results_Trimmed = NAL_Model_Train_Condo_results[(NAL_Model_Train_Condo_results['sale_ratio'] > .3) & (NAL_Model_Train_Condo_results['sale_ratio'] < 1.7)]


        #print('Number of Records Before Trimming: ', NAL_Model_Train_Condo_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Train_Condo_results_Trimmed.shape[0])

        NAL_Model_Test_Condo_results = NAL_Data_Test_Condo.copy()

        NAL_Model_Test_Condo_results['Model_Prediction'] = NAL_Model_Condo.predict(NAL_Data_Test_Condo_Features)

        NAL_Model_Test_Condo_results['sale_ratio'] = NAL_Model_Test_Condo_results['Model_Prediction'] / NAL_Model_Test_Condo_results[Model_Outcome[0]] 

        NAL_Model_Test_Condo_results_Trimmed = NAL_Model_Test_Condo_results[(NAL_Model_Test_Condo_results['sale_ratio'] > .3) & (NAL_Model_Test_Condo_results['sale_ratio'] < 1.7)]

        ## 🩹 These two lines used to sit 15 lines higher, attached to the
        ## TRAIN frame - so the Condo model's "Trimmed Test NAL" figures
        ## were measured on data the model had been fitted to, and labelled
        ## Test. SFR, Vacant Residential and Commercial all used their test
        ## split; Condo was the odd one out, which is the hardest kind of
        ## inconsistency to notice because every number still looked sane.
        NAL_Model_Test_Condo_results['sale_ratio_JV'] = NAL_Model_Test_Condo_results['JV'] / NAL_Model_Test_Condo_results[Model_Outcome[0]]

        ## ── Phase 4.4: ONE row set for both sides of the comparison ─────────
        ##
        ## 🩹 This used to keep only the rows whose COUNTY ratio was in range,
        ## while the model's metrics came from a frame keeping the rows whose
        ## MODEL ratio was in range. Two different row sets - and the app then
        ## drew them side by side in a chart labelled "Model" vs "NAL Values".
        ##
        ## That flatters both. Trimming on the model's ratio drops the sales the
        ## MODEL did worst on; trimming on the county's drops the sales the
        ## COUNTY did worst on. Each side was scored on the subset kindest to
        ## it, and a reader was invited to subtract one from the other.
        ##
        ## Now: the rows where BOTH ratios are in range. Same rows, same count,
        ## and the model's own numbers are computed on that set too, so the
        ## comparison is like-for-like on both axes.
        ##
        ## 📌 The `Trimmed Test *` keys are untouched. The model's own trimmed
        ## performance is a fair standalone figure - it just is not a comparison.
        _model_ratio_in_range = ((NAL_Model_Test_Condo_results['sale_ratio'] > .3)
                                 & (NAL_Model_Test_Condo_results['sale_ratio'] < 1.7))
        _county_ratio_in_range = ((NAL_Model_Test_Condo_results['sale_ratio_JV'] > .3)
                                  & (NAL_Model_Test_Condo_results['sale_ratio_JV'] < 1.7))
        
        NAL_Model_Test_Condo_results_Comparable = NAL_Model_Test_Condo_results[_model_ratio_in_range & _county_ratio_in_range]

        ## See the Single Family block for why this is not the all-parcel frame.
        st.session_state["Sales_For_Chat"]["Condo"] = build_sales_rows(
            "Condo", NAL_Model_Train_Condo_results, NAL_Model_Test_Condo_results)
        
        ## Pulled out once, so the metric lines below read as comparisons
        ## rather than as a wall of indexing.
        _cmp_sale_price = NAL_Model_Test_Condo_results_Comparable[Model_Outcome[0]].values.ravel()
        _cmp_model_value = NAL_Model_Test_Condo_results_Comparable['Model_Prediction'].values.ravel()
        _cmp_county_value = NAL_Model_Test_Condo_results_Comparable['JV'].values.ravel()

        #print(NAL_Model_Test_results_Trimmed.info())

        #print('Number of Records Before Trimming: ', NAL_Model_Test_Condo_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Test_Condo_results_Trimmed.shape[0])

        NAL_Model_Data_Condo_For_Preds = st.session_state["NAL"].copy()

        if st.session_state['NBHD_Encoder'] is not None:
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





                                                    
                                                    ## ── The head-to-head, on identical rows ─────────────────────────
                                                    ## Both sides measured on the same sales. See the note where
                                                    ## _Comparable is built.
                                                    , 'Comparable Records' : NAL_Model_Test_Condo_results_Comparable.shape[0]

                                                    , 'Head to Head Model R Squared' : round(r2_score(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model Weighted Mean' : round(weighted_Mean_Sale_Ratio(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model COD' : round(COD(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model PRD' : round(PRD(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model MAE' : round(mean_absolute_error(_cmp_sale_price, _cmp_model_value), Result_Prescision)

                                                    , 'Head to Head County R Squared' : round(r2_score(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County Weighted Mean' : round(weighted_Mean_Sale_Ratio(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County COD' : round(COD(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County PRD' : round(PRD(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County MAE' : round(mean_absolute_error(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    
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
            ## 🎲 A seeded sampler, so the search is REPRODUCIBLE.
            ## Without `seed`, Optuna explores differently every run and
            ## two people following the same steps get different models -
            ## which makes any number either of them quotes unverifiable.
            ##
            ## ✂️ The pruner abandons a trial once its running score is
            ## worse than the median trial at the same fold. It is what
            ## pays for cross-validation: hopeless trials cost one or two
            ## fits instead of all of them. `n_startup_trials` lets the
            ## first 10 finish, so there is something to take a median of.
            LGBM_study = optuna.create_study(
                direction = 'minimize'
                , sampler = optuna.samplers.TPESampler(seed = 42)
                , pruner = optuna.pruners.MedianPruner(n_startup_trials = 10
                                                       , n_warmup_steps = 1)
            )
            ## No test data is passed any more - that is the point. The
            ## objective cross-validates within the training split.
            LGBM_study_obj_func = lambda trial: Model_Objective(trial
                                                    , Model_Train_Data = NAL_Data_Train_Commercial
                                                    , model_features = st.session_state['Model_Features_Commercial']
                                                    , performance_metric = 'COD'
                                                    , n_folds = st.session_state['Hyperparameter_Tuning_Folds'])
            
            LGBM_study.optimize(LGBM_study_obj_func
                                , n_trials = st.session_state['Hyperparameter_Tuning_Trials'])
            
            LGBM_Best_Params = LGBM_study.best_params.copy()
            LGBM_Best_Params = {**LGBM_Best_Params, **{'random_state': 42}}
            #LGBM_Best_Params = {**LGBM_Best_Params, **{'n_estimators': 1000}}
            LGBM_Best_Params = {**LGBM_Best_Params, **{'verbose': -1}}

            NAL_Model_Commercial = ltb.LGBMRegressor(**LGBM_Best_Params)
            

            ## 🛑 No `eval_set` on the test split here, deliberately.
            ##
            ## It used to early-stop against the test data: train until the
            ## test score stops improving. That reads like good practice and
            ## is the same leak as before in a quieter form - the test split
            ## decides how many trees the model keeps, so the test score it
            ## then reports is not a clean measurement.
            ##
            ## `n_estimators` comes from the cross-validated search instead,
            ## which chose it without ever seeing the test data. It also
            ## clears a deprecation: `eval_set` is on its way out of
            ## LightGBM in favour of `eval_X` / `eval_y`.
            NAL_Model_Commercial.fit(NAL_Data_Train_Commercial_Features
                                     , NAL_Data_Train_Commercial[Model_Outcome[0]])


        NAL_Model_Train_Commercial_results = NAL_Data_Train_Commercial.copy()

        NAL_Model_Train_Commercial_results['Model_Prediction'] = NAL_Model_Commercial.predict(NAL_Data_Train_Commercial_Features)

        NAL_Model_Train_Commercial_results['sale_ratio'] = NAL_Model_Train_Commercial_results['Model_Prediction'] / NAL_Model_Train_Commercial_results[Model_Outcome[0]] 

        NAL_Model_Train_Commercial_results_Trimmed = NAL_Model_Train_Commercial_results[(NAL_Model_Train_Commercial_results['sale_ratio'] > .3) & (NAL_Model_Train_Commercial_results['sale_ratio'] < 1.7)]

        #print('Number of Records Before Trimming: ', NAL_Model_Train_Commercial_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Train_Commercial_results_Trimmed.shape[0])

        NAL_Model_Test_Commercial_results = NAL_Data_Test_Commercial.copy()

        NAL_Model_Test_Commercial_results['Model_Prediction'] = NAL_Model_Commercial.predict(NAL_Data_Test_Commercial_Features)

        NAL_Model_Test_Commercial_results['sale_ratio'] = NAL_Model_Test_Commercial_results['Model_Prediction'] / NAL_Model_Test_Commercial_results[Model_Outcome[0]] 

        NAL_Model_Test_Commercial_results['sale_ratio_JV'] = NAL_Model_Test_Commercial_results['JV'] / NAL_Model_Test_Commercial_results[Model_Outcome[0]] 

        NAL_Model_Test_Commercial_results_Trimmed = NAL_Model_Test_Commercial_results[(NAL_Model_Test_Commercial_results['sale_ratio'] > .3) & (NAL_Model_Test_Commercial_results['sale_ratio'] < 1.7)]

        ## ── Phase 4.4: ONE row set for both sides of the comparison ─────────
        ##
        ## 🩹 This used to keep only the rows whose COUNTY ratio was in range,
        ## while the model's metrics came from a frame keeping the rows whose
        ## MODEL ratio was in range. Two different row sets - and the app then
        ## drew them side by side in a chart labelled "Model" vs "NAL Values".
        ##
        ## That flatters both. Trimming on the model's ratio drops the sales the
        ## MODEL did worst on; trimming on the county's drops the sales the
        ## COUNTY did worst on. Each side was scored on the subset kindest to
        ## it, and a reader was invited to subtract one from the other.
        ##
        ## Now: the rows where BOTH ratios are in range. Same rows, same count,
        ## and the model's own numbers are computed on that set too, so the
        ## comparison is like-for-like on both axes.
        ##
        ## 📌 The `Trimmed Test *` keys are untouched. The model's own trimmed
        ## performance is a fair standalone figure - it just is not a comparison.
        _model_ratio_in_range = ((NAL_Model_Test_Commercial_results['sale_ratio'] > .3)
                                 & (NAL_Model_Test_Commercial_results['sale_ratio'] < 1.7))
        _county_ratio_in_range = ((NAL_Model_Test_Commercial_results['sale_ratio_JV'] > .3)
                                  & (NAL_Model_Test_Commercial_results['sale_ratio_JV'] < 1.7))
        
        NAL_Model_Test_Commercial_results_Comparable = NAL_Model_Test_Commercial_results[_model_ratio_in_range & _county_ratio_in_range]

        ## See the Single Family block for why this is not the all-parcel frame.
        st.session_state["Sales_For_Chat"]["Commercial"] = build_sales_rows(
            "Commercial", NAL_Model_Train_Commercial_results, NAL_Model_Test_Commercial_results)
        
        ## Pulled out once, so the metric lines below read as comparisons
        ## rather than as a wall of indexing.
        _cmp_sale_price = NAL_Model_Test_Commercial_results_Comparable[Model_Outcome[0]].values.ravel()
        _cmp_model_value = NAL_Model_Test_Commercial_results_Comparable['Model_Prediction'].values.ravel()
        _cmp_county_value = NAL_Model_Test_Commercial_results_Comparable['JV'].values.ravel()

        #print(NAL_Model_Test_results_Trimmed.info())

        #print('Number of Records Before Trimming: ', NAL_Model_Test_Commercial_results.shape[0])
        #print('Number of Records After Trimming: ', NAL_Model_Test_Commercial_results_Trimmed.shape[0])

        NAL_Model_Data_Commercial_For_Preds = st.session_state["NAL"].copy()

        if st.session_state['NBHD_Encoder'] is not None:
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





                                                    
                                                    ## ── The head-to-head, on identical rows ─────────────────────────
                                                    ## Both sides measured on the same sales. See the note where
                                                    ## _Comparable is built.
                                                    , 'Comparable Records' : NAL_Model_Test_Commercial_results_Comparable.shape[0]

                                                    , 'Head to Head Model R Squared' : round(r2_score(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model Weighted Mean' : round(weighted_Mean_Sale_Ratio(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model COD' : round(COD(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model PRD' : round(PRD(_cmp_sale_price, _cmp_model_value), Result_Prescision)
                                                    , 'Head to Head Model MAE' : round(mean_absolute_error(_cmp_sale_price, _cmp_model_value), Result_Prescision)

                                                    , 'Head to Head County R Squared' : round(r2_score(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County Weighted Mean' : round(weighted_Mean_Sale_Ratio(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County COD' : round(COD(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County PRD' : round(PRD(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    , 'Head to Head County MAE' : round(mean_absolute_error(_cmp_sale_price, _cmp_county_value), Result_Prescision)
                                                    
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

## There used to be a hand-rolled retry wrapper monkey-patched onto the
## model client here, so that HTTP 429 (rate limited) and 503 (busy) were
## retried. The Anthropic SDK already does that - it retries connection
## errors, 408, 409, 429 and 5xx twice with exponential backoff - so the
## wrapper is gone. Set max_retries on the client to change it.
##
## It also ran at IMPORT time, which meant a broken language-model
## dependency took down the entire valuation app, 2,200 lines away from
## anything to do with chat.
  
## ⚠️ The docstrings below are not ordinary comments. The tool runner
## builds each tool's schema from the function signature, and Claude picks
## which tool to call by READING THE DESCRIPTION. A vague docstring here
## shows up as a model that keeps choosing the wrong tool.
@beta_tool
def list_tables() -> str:
    """List the tables available in the parcel database.

    Returns:
        One table name per line.
    """
    # These prints are deliberate: they go to the terminal you launched
    # from, so you can watch which tools the model reaches for.
    print(' - DB CALL: list_tables()')

    cursor = st.session_state["Chat_Bot_Data_Base"].cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    return "\n".join(t[0] for t in cursor.fetchall())

  
@beta_tool
def describe_table(table_name: str) -> str:
    """Look up the columns of one table and their types.

    Args:
        table_name: The name of the table to describe.

    Returns:
        One "column_name: type" line per column.
    """
    print(f' - DB CALL: describe_table({table_name})')

    cursor = st.session_state["Chat_Bot_Data_Base"].cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")

    # PRAGMA table_info gives (index, name, type, ...) per column.
    return "\n".join("%s: %s" % (col[1], col[2]) for col in cursor.fetchall())

@beta_tool
def execute_query(sql: str) -> str:
    """Run a read-only SQL SELECT query against the parcel database.

    Args:
        sql: A single SQL SELECT statement.

    Returns:
        The rows, one per line, or an error message explaining what went
        wrong so the query can be corrected and retried.
    """
    print(f' - DB CALL: execute_query({sql})')

    # 🔒 This runs SQL that a language model wrote. The database connection
    # is opened READ-ONLY (see Build_Chat_Bot_And_Data), which is what
    # actually makes `DROP TABLE Parcel_Data` impossible rather than
    # merely unlikely. Asking the model nicely is not a security control.
    try:
        cursor = st.session_state["Chat_Bot_Data_Base"].cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
    except Exception as e:
        # Hand the error back as a normal tool result rather than raising.
        # The model can then read it, fix the query and try again - which
        # is far better for the person waiting than a crashed app.
        print(f' - DB CALL FAILED: {e}')
        return f"That query failed: {e}. Check the column names with "\
               f"describe_table and try again."

    if not rows:
        return "The query ran but matched no rows."

    # Keep a lid on it: a county roll has hundreds of thousands of parcels,
    # and a SELECT * would otherwise be sent to the model in full.
    LIMIT = 200
    body = "\n".join(str(list(r)) for r in rows[:LIMIT])
    if len(rows) > LIMIT:
        body += f"\n... {len(rows) - LIMIT:,} more rows not shown "\
                f"(ask for a COUNT or an aggregate instead)."
    return body



@beta_tool
def describe_Data(sql_query: str) -> str:
    """Summarise one numeric column: count, mean, standard deviation,
    minimum, quartiles and maximum.

    Args:
        sql_query: A SQL SELECT returning exactly ONE numeric column,
            for example "SELECT Model_Value FROM Parcel_Data".

    Returns:
        The descriptive statistics as text.
    """
    print(f' - DB CALL: describe_Data({sql_query})')
    try:
        df = pd.read_sql_query(sql_query, st.session_state["Chat_Bot_Data_Base"])
        return df.iloc[:, 0].describe().to_string()
    except Exception as e:
        return f"Could not summarise that: {e}. The query must return one "\
               f"numeric column."

@beta_tool
def query_plotter(sql_query: str) -> str:
    """Draw a histogram of one numeric column and show it to the user.

    Args:
        sql_query: A SQL SELECT returning exactly ONE numeric column,
            for example "SELECT Model_Value FROM Parcel_Data".

    Returns:
        Confirmation that the chart was drawn.
    """
    print(f' - DB CALL: query_plotter({sql_query})')
    try:
        df = pd.read_sql_query(sql_query, st.session_state["Chat_Bot_Data_Base"])
        st_fig, st_ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df.iloc[:, 0], kde = True, ax = st_ax)
    except Exception as e:
        return f"Could not draw that: {e}. The query must return one "\
               f"numeric column."

    # The figure is stashed rather than drawn here, because this function
    # runs inside the tool loop and Streamlit paints the page afterwards.
    st.session_state["Display_Plot"] = 1
    st.session_state["Chat_Bot_Plot"] = st_fig
    return ("The histogram has been drawn and is shown below the answer. "
            "Describe its shape for the user.")


@beta_tool
def ratio_study(compare: str = "model_vs_county", group_by: str = "",
                property_type: str = "", city: str = "",
                neighborhood_code: str = "") -> str:
    """Run a sales ratio study: the median ratio, weighted mean ratio, COD and PRD.

    This is the only correct way to get those statistics. It chooses the right
    sales and the right sale prices. Never work them out with execute_query. It
    does not compute PRB, which is a different statistic from PRD.

    Args:
        compare: "model_vs_county" scores the model and the county roll on the
            SAME held-out sales - use it for the model, or for any comparison.
            "county_only" scores the county roll on every qualified sale - use
            it when the question is only about the county's roll.
        group_by: Leave blank for one overall answer, or use "city",
            "neighborhood" or "sale_year" for one line per group, worst COD
            first, with small groups listed apart.
        property_type: Leave blank for every modelled type, or name one:
            "Single Family", "Condo", "Vacant Residential", "Commercial".
        city: Leave blank, or name one city to study only that city.
        neighborhood_code: Leave blank, or give one neighborhood code.

    Returns:
        The statistics, labelled, with how many sales they cover and which ones.
    """
    print(f' - DB CALL: ratio_study(compare={compare}, group_by={group_by}, '
          f'property_type={property_type}, city={city}, '
          f'neighborhood_code={neighborhood_code})')

    ## 📌 The arithmetic and the choice of rows live in
    ## src/deep_thought/ratio_study.py rather than here. A tool that reads
    ## st.session_state cannot be tested from outside a running app, and this
    ## is the one tool whose answer has to match the Results tab to the digit.
    try:
        sales = pd.read_sql_query(f"SELECT * FROM {SALES_TABLE}",
                                  st.session_state["Chat_Bot_Data_Base"])
        return run_ratio_study(sales, compare = compare, group_by = group_by,
                               property_type = property_type, city = city,
                               neighborhood_code = neighborhood_code,
                               minimum_sales = st.session_state["Minimum_Sales_For_A_Statistic"],
                               acceptable_ranges = Current_Acceptable_Ranges(),
                               warn_below = st.session_state["Thin_Group_Warning"])
    except Exception as e:
        print(f' - DB CALL FAILED: {e}')
        return f"The ratio study could not run: {e}"


@beta_tool
def acceptable_ranges() -> str:
    """Look up the acceptable ranges this office has set for level, COD and PRD.

    Use this whenever you are asked what an acceptable range is. Never quote one
    from the IAAO, from any other body, or from memory.

    Returns:
        The range for each statistic and property type, as set on the Settings tab.
    """
    print(' - DB CALL: acceptable_ranges()')
    return describe_acceptable_ranges(Current_Acceptable_Ranges())


HOSTED_HISTORY_NOTE = """

    YOUR EARLIER ANSWERS. This conversation shows your earlier answers but not
    the tool calls behind them. Where one used tools, it ends with a note in square
    brackets saying which. Those figures were looked up properly at the time, so do
    not apologise for them or re-check them unless the person asks. Never write
    that note yourself. For each new question, call the tools that question needs."""


class NAL_Chat_Session:
    """One conversation with Claude about your roll.

    The API is **stateless**: there is no conversation living on a server
    somewhere. Every turn sends the whole history back. This class is
    simply the thing that holds that history, so the rest of the app can
    call `.send_message("...")` and not think about it.

    📌 Note what is kept and what is not. Each turn stores the question
    and the final answer, but NOT the tool calls and the SQL rows that
    produced it. That is deliberate: a `SELECT` against a county roll can
    return a great deal of text, and carrying all of it forward would
    make every later question slower and more expensive for no benefit.
    The model still has its own answer to refer back to.
    """

    def __init__(self, client, model, system_instruction, tools):
        self.client = client
        self.model = model
        self.system_instruction = system_instruction
        self.tools = tools
        self.messages = []
        self.last_tool_calls = []

    def send_message(self, question):
        """Ask a question, let Claude use the tools, return the answer.

        `tool_runner` drives the whole back-and-forth: it sends the
        question, and when Claude asks for a tool it runs the matching
        Python function, feeds the result back, and goes round again until
        Claude has an answer. Each pass of the loop below is one of those
        rounds, which is why we can report the tools that were used.
        """
        self.last_tool_calls = []

        ## 🩹 This stores each answer but not the tool calls behind it, so an
        ## earlier answer looks to the model as though nothing backed it. Told to
        ## get every figure from a tool, it started apologising for correct
        ## answers: "I gave figures without running any queries", which was false,
        ## and re-running queries nobody had asked about. What it lacked was a
        ## fact, so it gets told the fact.
        system = self.system_instruction
        if self.messages:
            system += HOSTED_HISTORY_NOTE

        request = dict(
            model = self.model,
            max_tokens = 16000,
            system = system,
            tools = self.tools,
            messages = self.messages + [{"role": "user", "content": question}],
        )

        ## 🩹 `effort` - how hard Claude works before answering - is only
        ## sent to the models we have actually checked. Not every model
        ## accepts it, and one that does not returns
        ##
        ##     400 - This model does not support the effort parameter.
        ##
        ## on EVERY request, which turns the whole chat into an error
        ## message. That really happened: a leftover ANTHROPIC_MODEL value
        ## copied from another project named an older model, and nothing
        ## worked until this check existed. A reader is free to put any
        ## model in .env, so this cannot be an assumption.
        if self.model in VERIFIED_CHAT_MODELS:
            request["output_config"] = {"effort": "medium"}

        ## Errors are returned as sentences, not raised. A county laptop
        ## may sit behind a proxy, a key may be mistyped, and a rate limit
        ## is routine - none of those should look like a crashed app to
        ## somebody trying this for the first time.
        try:
            runner = self.client.beta.messages.tool_runner(**request)

            final_message = None
            for message in runner:
                final_message = message
                for block in message.content:
                    if block.type == "tool_use":
                        self.last_tool_calls.append(block.name)

        except anthropic.AuthenticationError:
            return ("That API key was rejected. Check ANTHROPIC_API_KEY in "
                    "your .env file, or paste a key into the box above.")
        except anthropic.NotFoundError:
            return (f"The model '{self.model}' was not found on your account. "
                    f"Pick one from the dropdown.")
        except anthropic.RateLimitError:
            return ("The API is rate limiting us. Wait a moment and ask "
                    "again - nothing is broken.")
        except anthropic.BadRequestError as e:
            return (f"The API rejected that request: {e}. If you changed "
                    f"ANTHROPIC_MODEL in your .env file, try one of the "
                    f"models in the dropdown instead.")
        except anthropic.APIConnectionError:
            return ("Could not reach the API. Check your internet "
                    "connection, and whether your network needs a proxy.")
        except anthropic.APIStatusError as e:
            return f"The API returned an error: {e}"

        if final_message is None:
            return "I did not get a reply from the model. Please try again."

        ## A safety classifier can decline a request outright. It is not an
        ## error and there is no exception to catch - the response simply
        ## comes back with stop_reason 'refusal' and no text.
        if final_message.stop_reason == "refusal":
            return ("The model declined to answer that one. Try rephrasing "
                    "the question around the roll data.")

        answer = "".join(
            block.text for block in final_message.content if block.type == "text"
        ).strip()

        if not answer:
            answer = "I could not find an answer to that in your data."

        ## 🩹 The model copies the note below into its own answers, even when told
        ## not to, so a reader was seeing "[Looked up with: ...]". Asking again would
        ## be asking again. It is simply taken off.
        answer = re.sub(r"\s*\[Looked up with:[^\]]*\]", "", answer).strip()

        ## The note goes in the history only, never on screen, so the model can
        ## see that an earlier answer was looked up. See HOSTED_HISTORY_NOTE.
        remembered = answer
        if self.last_tool_calls:
            remembered += ("\n\n[Looked up with: %s]"
                           % ", ".join(dict.fromkeys(self.last_tool_calls)))

        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": remembered})
        return answer


## ══════════════════════════════════════════════════════════════════════
##  THE LOCAL TWIN - the same assistant, on your own machine
## ══════════════════════════════════════════════════════════════════════
##
## Everything above talks to Anthropic over the internet. Everything below
## does the same job with a model running on this computer, so that no
## parcel data leaves the building. For a lot of assessor offices that is
## the only version IT will approve, which is the whole reason it exists.
##
## ⚠️ It is a SMALLER model with a SHORTER tool belt, and it shows. See
## TOOL_BELT_RULES below for what it can and cannot do.

## The model is read from the environment so a reader can try another one
## without editing code. It is NOT downloaded by `uv sync` - see README.
LOCAL_MODEL_ID = (os.environ.get("LOCAL_MODEL") or "").strip() \
                 or "Qwen/Qwen3.5-4B"


LOCAL_PACKAGES_MISSING = (
    "The local model needs some optional packages that are not installed. "
    "Close the app, run this once (about a 3 GB download), and start it again:"
    "\n\n`uv sync --extra slm`\n\nHosted works without them."
)


def local_weights_missing(model_id):
    """What to tell somebody whose model has not been downloaded yet."""
    return (
        f"The model `{model_id}` is not on this computer yet, and the app will "
        f"not start a large download without being asked. Run this once while "
        f"you have an internet connection (about 9 GB for the default model):"
        f"\n\n`uv run python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{model_id}')\"`\n\n"
        f"After that it runs with no connection at all."
    )


def local_model_facts(config):
    """Everything the app wants to know about a local model, in one place.

    🧨 This helper exists because of one specific trap. Qwen3.5-4B is a
    MULTIMODAL model, so its text settings do not live on the config - they
    live on `config.text_config`, and

        config.vocab_size

    raises `AttributeError`, it does not return a number. Ask it through
    here and a different model later cannot break the app.

    📌 And note what is NOT here: a hardcoded vocabulary size. Writing
    "248,320" into the app would be a number that is correct until the
    first person changes LOCAL_MODEL, and then it is a lie that no
    test catches.
    """
    text = getattr(config, "text_config", config)
    return {
        "vocab_size"  : getattr(text, "vocab_size", None),
        "layers"      : getattr(text, "num_hidden_layers", None),
        "hidden_size" : getattr(text, "hidden_size", None),
        "context"     : getattr(text, "max_position_embeddings", None),
        "multimodal"  : hasattr(config, "vision_config"),
    }


class Local_Model_Unavailable(Exception):
    """Why the local model cannot run here, in words for the person reading it."""


## 🔢 What the default model needs, measured on the Lee County roll. A different
## LOCAL_MODEL will differ, and these are then only a rough guide.
##
##     where        needs              speed          (24-core desktop)
##     GPU, 4-bit    3.07 GiB VRAM     ~15 tokens/sec
##     CPU, float32 16.46 GiB RAM      ~2.6 tokens/sec
##     CPU, bf16     8.60 GiB RAM      ~0.6 tokens/sec
##
## ⚠️ There is no free option on a processor. bf16 halves the memory and is more
## than four times slower, because most CPUs have to emulate it. So the app uses
## the faster one when it fits and the smaller one when it does not, and says
## which - and these are best-case speeds, so a laptop will be slower.
LOCAL_VRAM_NEEDED_GIB = 3.5
LOCAL_RAM_NEEDED_FOR_FLOAT32_GIB = 17.0
LOCAL_RAM_NEEDED_FOR_BF16_GIB = 9.5


@st.cache_resource(show_spinner = False)
def load_local_model(model_id = LOCAL_MODEL_ID, prefer_4bit = True):
    """Load the local model once and keep it. Returns (tokenizer, model, note).

    Raises `Local_Model_Unavailable`, with a message meant to be shown, when it
    cannot run here - which for most readers is the expected case, since the
    packages are an optional extra and the weights are a separate download.

    ⚠️ `transformers` and `torch` are imported INSIDE this function on
    purpose. They are the optional `slm` extra, roughly 3 GB, and the
    valuation half of this app needs none of it. A module-level import
    here would mean a reader who ran a plain `uv sync` could not open the
    app at all - which is exactly how this app used to be broken, by a
    `from google import genai` on line 74.

    🔒 Everything loads with `local_files_only = True`. Without it, this quietly
    does two things this feature promises not to:

      - with the weights missing, it starts a 9 GB download nobody asked for
      - with the weights present, it still contacts huggingface.co on every
        load. Measured on a blocked network: 25 refused connections, and 88
        seconds to start instead of 17.

    Quantising is the DEFAULT, not a fallback. bf16 needs 8.57 GiB of video
    memory, and CLAUDE.md says design to an 8 GB card - so bf16 simply does not
    fit the machine we are designing for, while costing only about 20% of the
    speed.
    """
    ## 🔒 And this, because `local_files_only` turned out not to be enough. Even
    ## with it, huggingface_hub fetches a list from huggingface.co just to build
    ## a User-Agent string. Only offline mode stops that, and the library reads
    ## the setting once, when first imported - so it is set before the import,
    ## and on the constant in case something imported the library earlier. This
    ## affects this process only: the one-time download in the README is a
    ## separate command and still works.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        import torch
        from transformers import (AutoConfig, AutoModelForCausalLM,
                                  AutoTokenizer, BitsAndBytesConfig)
    except ImportError as e:
        raise Local_Model_Unavailable(LOCAL_PACKAGES_MISSING) from e

    try:
        config = AutoConfig.from_pretrained(model_id, local_files_only = True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only = True)
    except OSError as e:
        raise Local_Model_Unavailable(local_weights_missing(model_id)) from e

    facts = local_model_facts(config)

    ## ⚠️ AutoModelForCausalLM resolves to the TEXT-only class, even though
    ## Qwen3.5-4B declares the image-and-text one. That skips 297 vision
    ## tensors we would never use, 3.07 GiB against 3.15 for the full load.
    kwargs = dict(local_files_only = True)

    free_vram_gib = 0.0
    if torch.cuda.is_available():
        free_vram_gib = torch.cuda.mem_get_info()[0] / 2 ** 30
        print(" - Local model: GPU available, %.2f GiB free" % free_vram_gib)

    if free_vram_gib >= LOCAL_VRAM_NEEDED_GIB:
        kwargs["device_map"] = "auto"
        if prefer_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit             = True,
                bnb_4bit_quant_type      = "nf4",
                bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype   = torch.bfloat16,
            )
            note = ("Running on your %s, quantised to 4-bit (about 3 GB of "
                    "video memory)." % torch.cuda.get_device_name(0))
        else:
            kwargs["dtype"] = torch.bfloat16
            note = ("Running on your %s at full bf16 precision (about 8.6 GB "
                    "of video memory)." % torch.cuda.get_device_name(0))
    else:
        free_ram_gib = psutil.virtual_memory().available / 2 ** 30
        print(" - Local model: no usable GPU, %.2f GiB of RAM free" % free_ram_gib)
        kwargs["device_map"] = "cpu"

        if free_ram_gib >= LOCAL_RAM_NEEDED_FOR_FLOAT32_GIB:
            kwargs["dtype"] = torch.float32
            note = ("No usable graphics card, so this is running on your "
                    "processor, using about 16.5 GB of memory. Expect a minute "
                    "or more per answer on a fast desktop, and several times "
                    "that on a laptop.")
        elif free_ram_gib >= LOCAL_RAM_NEEDED_FOR_BF16_GIB:
            kwargs["dtype"] = torch.bfloat16
            note = ("No usable graphics card, so this is running on your "
                    "processor, in a smaller format because memory is tight: "
                    "about 8.6 GB. It is slow. Expect several minutes per "
                    "answer on a fast desktop, and longer on a laptop.")
        else:
            raise Local_Model_Unavailable(
                "This computer has %.1f GB of memory free and no usable "
                "graphics card, and the local model needs about %.1f GB to "
                "run. Close some programs and try again, or use Hosted."
                % (free_ram_gib, LOCAL_RAM_NEEDED_FOR_BF16_GIB))

    print(" - Local model: loading %s ..." % model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ImportError as e:
        raise Local_Model_Unavailable(LOCAL_PACKAGES_MISSING) from e
    except OSError as e:
        raise Local_Model_Unavailable(local_weights_missing(model_id)) from e
    model.eval()

    ## Report the shape of the model from the config, never from a literal.
    print(" - Local model ready: %s layers, hidden %s, context %s, vocab %s"
          % (facts["layers"], facts["hidden_size"], facts["context"],
             facts["vocab_size"]))
    return tokenizer, model, note


## ── Letting the local model use the app's own five tools ──────────────
##
## 📌 The five tools above are decorated with `@beta_tool`, which is part
## of the Anthropic SDK - so the obvious assumption is that a local model
## needs its own copy of all five. It does not. `@beta_tool` returns a
## `BetaFunctionTool` that exposes
##
##     .name           the function name
##     .description    the first line of the docstring
##     .input_schema   a plain JSON Schema built from the signature
##     .call({...})    runs the function
##
## all of which are ordinary data. Only the DISPATCH LOOP is
## Anthropic-specific. So both engines share one set of tools, and a tool
## fixed once is fixed for both.
def tool_to_local_schema(tool):
    """One `@beta_tool` function, as a local chat template wants to see it."""
    return {"type": "function",
            "function": {"name": tool.name,
                         "description": tool.description,
                         "parameters": tool.input_schema}}


## ── Reading the model's tool calls ────────────────────────────────────
##
## ⚠️ Qwen3.5 does NOT emit JSON tool calls. Qwen2.5 did, which makes this
## an easy thing to get wrong from memory. The format its chat template
## asks for is:
##
##     <tool_call>
##     <function=execute_query>
##     <parameter=sql>
##     SELECT COUNT(*) FROM Parcel_Data
##     </parameter>
##     </function>
##     </tool_call>
_LOCAL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)(?:</function>|\Z)", re.DOTALL)
_LOCAL_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>\n?(.*?)\n?(?:</parameter>|\Z)", re.DOTALL)

## Any shred of tool-call markup. If one of these survives into text we
## were about to show somebody, what we have is a FRAGMENT, not an answer.
_LOCAL_STRAY_RE = re.compile(
    r"</?tool_call>|</?function[=>]|</?parameter[=>]")


def parse_local_tool_calls(reply, tools_by_name):
    """Pull every tool call out of a reply. Returns (calls, remaining_text).

    Parameter values arrive as plain text, so each one is coerced using the
    tool's OWN JSON Schema. A parameter declared `integer` must not reach
    the function as the string "5".
    """
    calls = []
    for match in _LOCAL_CALL_RE.finditer(reply):
        name = match.group(1).strip()
        tool = tools_by_name.get(name)
        properties = (tool.input_schema or {}).get("properties", {}) if tool else {}

        arguments = {}
        for param in _LOCAL_PARAM_RE.finditer(match.group(2)):
            key, raw = param.group(1).strip(), param.group(2)
            wanted = (properties.get(key) or {}).get("type", "string")
            if wanted in ("integer", "number", "boolean", "object", "array"):
                try:
                    arguments[key] = json.loads(raw.strip())
                except Exception:
                    arguments[key] = raw.strip()
            else:
                arguments[key] = raw.strip()
        calls.append({"name": name, "arguments": arguments})

    return calls, _LOCAL_CALL_RE.sub("", reply).strip()


def scrub_tool_markup(text):
    """Never show tool-call markup to a person."""
    return _LOCAL_STRAY_RE.sub("", text).strip()


class Local_NAL_Chat_Session:
    """One conversation with a model running on this machine.

    Deliberately the same shape as `NAL_Chat_Session` above: it holds the
    history and exposes **`send_message(question)`** returning a string,
    plus `last_tool_calls`. That is the entire contract the rest of the app
    uses, so the two engines are interchangeable and nothing downstream
    needs to know which one answered.

    What is different is the loop. The hosted version hands the whole
    back-and-forth to `tool_runner`. Here we run it ourselves, which is
    about forty lines and is worth reading once if you want to see what a
    tool-calling loop actually is:

        generate -> did it ask for a tool? -> run it -> hand back the
        result -> generate again -> ... -> plain answer
    """

    ## How many earlier questions it is reminded of. See send_message.
    QUESTIONS_REMEMBERED = 5

    ## How many times round that loop before we give up. Six is enough for
    ## list_tables -> describe_table -> query -> answer with room to spare,
    ## and it stops a model that keeps retrying a query it cannot fix from
    ## looping forever while somebody watches a spinner.
    MAX_TOOL_ROUNDS = 6

    def __init__(self, tokenizer, model, system_instruction, tools,
                 device_note = ""):
        self.tokenizer = tokenizer
        self.model = model
        self.system_instruction = system_instruction
        self.tools = tools
        self.device_note = device_note
        self.messages = []
        self.last_tool_calls = []

    def _generate(self, messages, schemas, stop_at_tool_call = True):
        """One pass of the model over the conversation so far."""
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools = schemas,
            tokenize = False,
            add_generation_prompt = True,
            ## 🤔 Thinking OFF. Qwen3.5 opens a `<think>` block by default
            ## and will reason for a couple of hundred tokens before
            ## answering. With the definitions below in the prompt it gets
            ## these questions right WITHOUT thinking, so the reasoning
            ## buys nothing here and costs a reader real seconds - more on
            ## a laptop with no GPU.
            enable_thinking = False,
        )
        inputs = self.tokenizer(prompt, return_tensors = "pt").to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]

        generate_kwargs = dict(max_new_tokens = 400, do_sample = False)
        if stop_at_tool_call:
            ## Stop as soon as a tool call closes - there is nothing useful
            ## after it, and generating anyway costs seconds per round.
            generate_kwargs["stop_strings"] = ["</tool_call>"]
            generate_kwargs["tokenizer"] = self.tokenizer

        with torch.inference_mode():
            output = self.model.generate(**inputs, **generate_kwargs)
        return self.tokenizer.decode(output[0][prompt_length:],
                                     skip_special_tokens = True).strip()

    def send_message(self, question):
        """Ask a question, let the model use the tools, return the answer."""
        self.last_tool_calls = []
        tools_by_name = {tool.name: tool for tool in self.tools}
        schemas = [tool_to_local_schema(tool) for tool in self.tools]

        ## 🩹 It remembers your earlier QUESTIONS, not its own earlier ANSWERS.
        ##
        ## With its answers in view, in one long conversation, this model stopped
        ## looking things up. Asked for the mean Just_Value it made no tool call
        ## and said $474,296 - the mean of Model_Value, from an answer further up.
        ## The real figure is $460,044. It also set a county figure beside a model
        ## figure from different sales, which is the mistake FINDINGS 39 records.
        ##
        ## Asking it not to is one more prohibition (FINDINGS 41). This way there
        ## is no old number to reach for, and "what about in Cape Coral?" still
        ## makes sense, because the question it follows up on is still there.
        ##
        ## ⚠️ The cost is that "explain that more simply" cannot work here, and the
        ## chat tab says so. The hosted model keeps the full conversation.
        system = self.system_instruction
        earlier_questions = [m["content"] for m in self.messages
                             if m["role"] == "user"][-self.QUESTIONS_REMEMBERED:]
        if earlier_questions:
            system += ("\n\n    EARLIER IN THIS CONVERSATION the person asked the "
                       "following, oldest first. This is only so you can follow "
                       "what they mean. You do not have your earlier answers, so "
                       "look up any figure again.\n"
                       + "\n".join("    - %s" % q for q in earlier_questions))

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": question}]

        answer = None
        try:
            for _ in range(self.MAX_TOOL_ROUNDS):
                reply = self._generate(messages, schemas)
                calls, spoken = parse_local_tool_calls(reply, tools_by_name)

                ## 🩹 A real failure, found by testing: the model sometimes
                ## narrates ("Now I will describe the table...") and emits a
                ## bare `</tool_call>` with no opening tag. The stop string
                ## fires on it, nothing parses as a call, and the loop used
                ## to hand that half-sentence to the user AS THE ANSWER,
                ## raw markup included.
                ##
                ## Re-running the same prompt would change nothing - this is
                ## greedy decoding, so it is deterministic. Re-running it
                ## WITHOUT the stop string lets the model carry on past the
                ## stray tag and produce the call it was reaching for.
                if not calls and _LOCAL_STRAY_RE.search(reply):
                    print(' - Local model: incomplete tool call, retrying')
                    reply = self._generate(messages, schemas,
                                           stop_at_tool_call = False)
                    calls, spoken = parse_local_tool_calls(reply, tools_by_name)

                if not calls:
                    answer = scrub_tool_markup(reply)
                    break

                messages.append({"role": "assistant",
                                 "content": spoken,
                                 "tool_calls": [{"function": c} for c in calls]})

                for call in calls:
                    self.last_tool_calls.append(call["name"])
                    tool = tools_by_name.get(call["name"])
                    if tool is None:
                        result = ("There is no tool called %s. The tools are: "
                                  "%s." % (call["name"], ", ".join(tools_by_name)))
                    else:
                        try:
                            result = tool.call(call["arguments"])
                        except Exception as e:
                            ## Hand the error back as a tool result rather
                            ## than raising, exactly as the hosted path
                            ## does, so the model can correct itself.
                            result = "That tool failed: %s: %s" % (type(e).__name__, e)
                    messages.append({"role": "tool", "content": str(result)})

        except Exception as e:
            ## A local model fails differently from an API: out of memory,
            ## a missing package, a driver problem. None of those should
            ## look like a crashed app to somebody trying this once.
            print(' - Local model error: %s: %s' % (type(e).__name__, e))
            return ("The local model ran into a problem: %s. If this says "
                    "something about memory, try closing other programs, or "
                    "press Start Over and choose Hosted." % e)

        if not answer:
            answer = ("I could not reach an answer with the tools I have. "
                      "Try a narrower question, or press Start Over and "
                      "choose Hosted.")

        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": answer})
        return answer


## ══════════════════════════════════════════════════════════════════════
##  What BOTH engines are told - the prompt that stops confident nonsense
## ══════════════════════════════════════════════════════════════════════
##
## 🛑 Every line of this was earned by a measured wrong answer, not added
## as a precaution. Asked to read a ratio study, Qwen3.5-4B invented FIVE
## different expansions of "PRD" across four runs - "Percent of Residential
## Determination", "Property Ratio Deviation", "Percentile Ratio
## Difference" - and described a sales ratio upside down. It was fluent and
## confident every time.
##
## It is in the SHARED prompt rather than the local one because a
## definition cannot hurt the hosted model, and because the alternative is
## two prompts that drift.
##
## 🩹 This used to say the mean Model_Value is "about 40% higher" than the
## median. That was a fact about Lee County, sent to the model for every
## county's roll. The assistant then repeated it, and it read like a measured
## result because it happened to be true here. A number about the data does
## not belong in the prompt. It comes from a tool, or it does not get said.
RATIO_STUDY_GROUNDING = """

    THE STATISTICS, DEFINED. Use these exactly. Do not invent others, and
    do not guess at what an abbreviation stands for - if you are not
    certain, say so.

    - A SALES RATIO is the appraised value DIVIDED BY the sale price. A
      ratio BELOW 1.00 means the property is assessed BELOW what it sold
      for. Getting this the wrong way round reverses the conclusion.
    - MEDIAN RATIO measures the LEVEL of assessment.
    - COD is the COEFFICIENT OF DISPERSION. It measures UNIFORMITY - how
      far individual ratios scatter around the median. LOWER IS BETTER.
      COD is NOT a percentage difference between two numbers.
    - PRD is the PRICE-RELATED DIFFERENTIAL. It measures VERTICAL EQUITY,
      and it is the MEAN ratio divided by the WEIGHTED MEAN ratio. Well above
      1.00 suggests regressivity - cheaper homes assessed relatively higher
      than expensive ones.

    ACCEPTABLE RANGES. Never quote an acceptable range from the IAAO, from any
    other body, or from memory. Each jurisdiction sets its own, and standards
    change. The only ranges that count are the ones this office has entered on
    the Settings tab. Call acceptable_ranges to see them.

    NEVER SUBSTITUTE ONE STATISTIC FOR ANOTHER. If you are asked for a
    MEDIAN, do not compute an average and call it a median. Reporting one
    statistic under the name of another is the most damaging thing you can
    do here, because an assessor will act on it.

    This database is SQLite, which has NO median, percentile or mode
    function. AVG is the MEAN, and the mean is NOT the median: property values
    are strongly right-skewed, so the two can be far apart. To get a median,
    use the describe_Data tool, which reports the 50% quartile.

    If the data needed for an answer is not in the database, say so plainly
    instead of approximating it with something else.

    NEVER STATE A NUMBER YOU DID NOT GET FROM A TOOL. Every figure in your
    answer about this roll must come from a tool you called for THIS question.
    Do not reuse a figure from earlier in the conversation, because it may
    describe a different column or different sales. Do not estimate one, do not
    infer one from the shape of a distribution, and do not reach for a word like
    "approximately", "roughly" or "typically" to cover a number you have not
    looked up. Do not work out a percentage in your head either: answer what was
    asked, or let execute_query do the arithmetic. If you want a mean, a median,
    a quartile or a standard deviation, call describe_Data and read it off the
    result.

    This matters most when you describe a chart. Draw it with query_plotter,
    then call describe_Data on the SAME query, and describe the chart using
    those real numbers. A number you invented is worse than no number,
    because the person reading it cannot tell the difference."""


## 🧰 And the tool belt itself: when to reach for the ratio study, and the
## ways of getting a ratio that look right and are not. Each "never" here is a
## route to a real wrong number, measured on the Lee County roll.
TOOL_BELT_RULES = """

    RATIO STUDIES. For the median ratio, weighted mean ratio, COD or PRD, call
    the ratio_study tool. It uses the right sales and the right sale prices.

    - How does the model do, or does it beat the county: ratio_study with no
      arguments. It scores both on the same sales.
    - Only about the county's own roll: compare = "county_only".
    - Broken down by city, neighborhood or year: use group_by.
    - One city: city = "CAPE CORAL" or whichever city was asked about.

    When you report those figures, copy them exactly as the tool gave them, and
    say how many sales they cover and which sales they are. The tool states whether
    each figure is inside this office's range, and who is ahead. Repeat what it
    states. Do not judge those for yourself. When the tool gives
    figures for both the model and the county, report both, each with its name,
    unless the question is only about one of them. Figures from
    compare = "county_only" cover different sales from the model's figures, so
    never put them side by side.

    SMALL GROUPS. A figure from a handful of sales mostly measures the handful. When
    ratio_study warns that a figure rests on few sales, pass the warning on plainly
    and say how many sales it is. If you are asked which group is best or worst,
    answer from the list of groups with enough sales, and mention that smaller groups
    are listed apart because their figures cannot be relied on.

    NEVER DECLARE AN OVERALL WINNER between the model and the county. Level
    (median ratio), uniformity (COD) and vertical equity (PRD) are separate
    questions, and they often point different ways. Say which is ahead on each
    statistic and stop there. The model is a benchmark, not a better roll.

    THE SALES TABLE. Every row is one qualified sale, so COUNT(*) on its own
    counts all of them. Held_Out says what the model did with that sale: 1 means
    it was kept back to test the model, 0 means the model was trained on it. Both
    are qualified sales, so do not filter on Held_Out to count qualified sales.

    THE MODEL ON ITS OWN TRAINING SALES. You have no figures for this and
    ratio_study never reports them. If you are asked, say that you do not have
    them, because scoring a model on sales it learned from flatters it, and that
    the Train column on the Results tab shows them. Do NOT answer with the
    held-out figures as though they were the training figures.

    NEVER work out a ratio statistic yourself with execute_query or
    describe_Data. In particular:

    - Never divide Model_Value by a sale price. Model_Value is predicted as of
      the valuation date, for every parcel, and is not comparable with a sale.
    - Model_Value_At_Sale_Date is empty on the sales the model trained on. That
      is deliberate. Scoring a model on its own training sales flatters it, so
      do not look for a way around it.
    - Never divide Just_Value by Model_Value and call it a sales ratio. Those
      are two estimates of value, and neither one is a sale.

    WHAT YOU CANNOT DO. PRB, the Price-Related Bias, is not computed in this
    app. The Sales table holds only the property types a model was built for,
    so other types have no ratio study here. If you are asked for something you
    have no tool for, say so plainly."""


def describe_chat_database(connection):
    """The real schema, read from the database we just built.

    📌 Written rather than hardcoded for a reason. The prompt used to list
    eight columns from memory while the table actually has thirteen - so
    the model was never told that City, Address, ZIP_Code or
    Neighborhood_Code existed, and could only find them by going looking.
    Reading it here means the prompt cannot drift from the table.
    """
    lines = ["\n\n    THE DATABASE. These are all of the tables and columns "
             "there are. The list is complete - if a column you need is not "
             "here, it does not exist, so say so rather than improvising one."]
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for (table_name,) in cursor.fetchall():
        columns = cursor.execute("PRAGMA table_info(%s);" % table_name).fetchall()
        row_count = cursor.execute("SELECT COUNT(*) FROM %s;" % table_name).fetchone()[0]
        lines.append("\n    Table %s - %s rows:" % (table_name, format(row_count, ",")))
        lines.append("      " + ", ".join("%s %s" % (c[1], c[2]) for c in columns))
    return "\n".join(lines)


## ── What this assistant can be asked ──────────────────────────────────
##
## 📋 Not a guess. `scripts/check_chat_assistant.py` puts every question below to
## the assistant, through its own chat box, on the real Lee County roll, and
## checks every number in the answer against the same figure computed in pandas
## or returned by the tool. Re-run it after changing the prompt or the model.
##
## 📌 The limits are the more useful half. Knowing where a tool stops being
## trustworthy is the most valuable thing a reader can be told about it.
CHAT_EXAMPLE_QUESTIONS = [
    "What columns are in the database?",
    "How many parcels are in the roll?",
    "How many parcels have a Just_Value over 500,000?",
    "What is the median Model_Value?",
    "What is the mean Just_Value?",
    "Which city has the most parcels, and how many?",
    "What is the average Model_Value for parcels built after 2000?",
    "Draw me a histogram of Model_Value and describe its shape.",
    "What does COD stand for, and what is an acceptable range?",
    "If a sales ratio is 0.94, is the property assessed above or below "
    "its sale price?",
    "How does the model compare with the county roll on COD and PRD?",
    "What is the county roll's COD on its own, using every qualified sale?",
    "How do the model and the county roll compare on COD in Cape Coral?",
    "Which city has the worst COD for the model?",
    "How many qualified sales are there?",
]

CHAT_KNOWN_LIMITS = [
    "PRB. It is not in the app yet, while new standards are pending. The "
    "notebook reports it.",
    "A ratio study for a property type you did not build a model for. The "
    "sales table only holds the types you modelled.",
    "Scoring the model on the sales it trained on. That is impossible on "
    "purpose, because it would flatter the model.",
    "Drawing a conclusion from a small group. It will give you the figures for a "
    "group with a handful of sales, with a warning, and keeps those groups out of "
    "any ranking. The Settings tab says how small is small.",
    "Questions it has to write its own multi-step SQL for. These depend on the "
    "model getting the SQL right, so check the figures.",
    "Following up on an answer, on Local. It remembers your earlier questions "
    "but not its own answers, so \"explain that more simply\" will not work. Ask "
    "the question in full. Hosted does not have this limit.",
    "Legal, policy or appeal advice. These are benchmark estimates.",
]


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

    ## There used to be a `Parcel_Data.to_csv(...)` here, writing every
    ## parcel to a hardcoded folder. Nothing ever read it back, and the
    ## app already offers proper download buttons further up, so it was a
    ## copy of real people's property records left on disk for no reason.
    ## It is gone. (CLAUDE.md: public does not mean careless.)

    ## Let go of any earlier chat's connection before rewriting its file.
    Reset_Chat_Session()

    ## ── Build the little database the assistant will query ──────────────
    APP_WORKSPACE.mkdir(parents = True, exist_ok = True)
    local_db_file = APP_WORKSPACE / "NAL_Navigator.db"

    ## Write it with an ordinary read-write connection, then CLOSE that
    ## connection. Nothing after this point needs to write.
    builder_connection = sqlite3.connect(local_db_file)
    ## 🧨 Dollar columns are stored as REAL, on purpose. SQLite divides two INTEGER
    ## columns as whole numbers and throws away the remainder, row by row, with no
    ## error. Measured: a model asked to divide Model_Value by Just_Value got 0.71,
    ## when the true mean is 1.0971 and the true median 1.0611. A rule asking it to
    ## CAST would be one more thing to forget, so the trap is simply not there.
    Parcel_Data.to_sql('Parcel_Data', builder_connection, if_exists = 'replace'
                       , dtype = {'Model_Value': 'REAL', 'Just_Value': 'REAL'})

    ## 🛑 Sales get a table of their own instead of a column on Parcel_Data. A
    ## parcel can sell more than once - 482 did - and joining those on would turn
    ## 268,269 parcels into 268,760, so "how many parcels?" would come back wrong.
    ##
    ## City and neighbourhood come from the roll by PARCEL_ID rather than from
    ## the model frames, which may hold label-encoded neighbourhood codes.
    if st.session_state["Sales_For_Chat"]:
        Sales_Data = pd.concat(list(st.session_state["Sales_For_Chat"].values())
                               , ignore_index = True)
        Sales_Data = pd.merge(Sales_Data
                              , NAL_For_DB[['PARCEL_ID', 'City', 'Neighborhood_Code']]
                                  .drop_duplicates(subset = 'PARCEL_ID')
                              , on = 'PARCEL_ID'
                              , how = 'left'
                              , validate = 'many_to_one')
        Sales_Data.to_sql(SALES_TABLE, builder_connection
                          , if_exists = 'replace', index = False
                          , dtype = {'Sale_Price': 'REAL', 'Just_Value': 'REAL'})

    builder_connection.commit()
    builder_connection.close()

    ## 🔒 Now reopen it READ-ONLY for the conversation.
    ##
    ## The `execute_query` tool runs SQL written by a language model. Any
    ## instruction we put in the system prompt is a request, not a
    ## control - and the people most likely to type something strange into
    ## the chat box are the ones exploring it for the first time.
    ##
    ## `mode=ro` moves that from unlikely to impossible: SQLite itself
    ## refuses the write, so DROP TABLE, DELETE and UPDATE all fail no
    ## matter what the model asks for. `uri=True` is required for the
    ## file:...?mode=ro form to be read as a URI rather than a filename.
    st.session_state["Chat_Bot_Data_Base"] = sqlite3.connect(
        f"file:{local_db_file.as_posix()}?mode=ro"
        , uri = True
        ## Streamlit runs the script on its own thread, so the connection
        ## is used from a different thread than the one that opened it.
        , check_same_thread = False
    )
    print(f' - Chat database ready (read-only) at {local_db_file}')

    ## Setting up google API key 

    ## The key comes from .env, and a key typed into the box overrides it.
    ## ── Which engine? ─────────────────────────────────────────────────
    ## The local model needs no key, no client and no internet, so none of
    ## that is set up for it. Building an `anthropic.Anthropic` client we
    ## would never use would also mean demanding a key from somebody who
    ## chose Local precisely because they do not have one.
    use_local_engine = using_local_engine()
    client = None
    Model_Name = None

    ANTHROPIC_API_KEY = st.session_state["Anthropic_Key"]

    typed_key = (st.session_state["Provided_Anthropic_Key"] or "").strip()
    if len(typed_key) > 12:
        ANTHROPIC_API_KEY = typed_key

    ## 🔒 NEVER print the key itself. An earlier version did, which wrote a
    ## live credential to the terminal every time the chatbot started.
    ## To check a key arrived, report only its SHAPE:
    if ANTHROPIC_API_KEY:
        print(f'API key loaded: yes ({len(str(ANTHROPIC_API_KEY))} characters)')
    else:
        print('API key loaded: NO - none provided')

    if use_local_engine:
        print(f'Chat engine: LOCAL - {LOCAL_MODEL_ID} (no API key needed)')
    else:
        client = anthropic.Anthropic(api_key = ANTHROPIC_API_KEY)

        ## Whatever was picked in the dropdown, falling back to the default.
        Model_Name = CHAT_MODELS.get(
            st.session_state['Chat_Model_Selected'], DEFAULT_CHAT_MODEL
        )
        print(f'Chat engine: HOSTED - {Model_Name}')

    db_tools = [list_tables, describe_table, execute_query, describe_Data,
                query_plotter, ratio_study, acceptable_ranges]

    ## 🩹 Two dead near-copies of this prompt sat here for a long time, under
    ## a comment saying they were gone. One of them nearly took a patch meant
    ## for the live prompt, because all three ended with the same line. They
    ## were deleted when the tool belt arrived, along with a hand-written column
    ## list that named 8 of the 13 columns. The real schema is read from the
    ## database below, so it cannot drift from the table.
    instruction = """You are a helpful chatbot that can interact with a SQL database containing property data. You will take the users questions and turn them into SQL
    queries using the tools available. Once you have the information you need, you will
    answer the user's question using the data returned.

    IMPORTANT - what Model_Value is.

    Model_Value is a BENCHMARK ESTIMATE from a deliberately simple baseline
    model, trained on one year of qualified sales with a handful of
    features. It is NOT an assessment, NOT a defensible value, and NOT
    evidence in an appeal. Just_Value is the county's own assessed value and
    is a different thing entirely.

    When you quote a Model_Value for a parcel, or an average or total of
    them, say once in that answer that these are benchmark estimates rather
    than assessed values. Do not repeat it in every sentence - say it clearly
    once and move on.

    If the user appears to be about to rely on one of these numbers as an
    assessment - asking what to put on a notice of value, what to tell a
    taxpayer, or what a parcel should be assessed at - tell them plainly that
    this is not what these numbers are for.

    You can use the following tools:
        - list_tables
        - describe_table
        - execute_query
        - describe_Data
        - query_plotter
        - ratio_study
        - acceptable_ranges

    Use list_tables to see what tables are present, describe_table to understand the
    schema, and execute_query to issue an SQL SELECT query. The tool query_plotter can be used to plot a histogram of a single continuous variable when it is given a valid SQL query with one continuous variable.
    The describe_Data tool will return the descriptive statistics as a string for a single continuous variable when it is given a valid SQL query contianing one continuous variable.

    When creating descriptive statistics you should format the output for the user in a way that is easy to read and provide a summary. If the user asks you to create a plot and to describe the plot
    use the describe_Data function to create the descriptive statistics for the plot.

    If you do not know the answer, you can say "I don't know"
    """

    ## ── The shared additions ──────────────────────────────────────────
    ## Both engines get the same definitions, the same prohibition on
    ## swapping one statistic for another, the REAL schema read from the
    ## database we just built, and the same rules for the ratio study. See
    ## the constants above for why each block exists - every one of them is
    ## a measured wrong answer.
    instruction = (instruction
                   + RATIO_STUDY_GROUNDING
                   + describe_chat_database(st.session_state["Chat_Bot_Data_Base"])
                   + TOOL_BELT_RULES)

    ## One conversation, holding its own history. See NAL_Chat_Session.
    ## Both classes expose the same `send_message(question) -> str`, so
    ## nothing downstream - not `response_generator`, not the chat box -
    ## needs to know which engine answered.
    if use_local_engine:
        ## 🩹 This was unguarded. Picking Local on a plain `uv sync` - which is
        ## most readers - put a raw `No module named 'torch'` traceback on the
        ## page. The local model not being able to run here is an ordinary
        ## situation, so it gets an ordinary sentence saying what to do.
        try:
            with st.spinner("Loading the local model - this takes a moment the "
                            "first time."):
                local_tokenizer, local_model, device_note = load_local_model()
        except Local_Model_Unavailable as e:
            st.session_state["Chat_Start_Problem"] = str(e)
            return
        except Exception as e:
            print(f' - Local model failed to load: {type(e).__name__}: {e}')
            st.session_state["Chat_Start_Problem"] = (
                f"The local model could not start: {e}. If this mentions "
                f"memory, try closing other programs, or use Hosted."
            )
            return
        chat = Local_NAL_Chat_Session(
            tokenizer = local_tokenizer,
            model = local_model,
            system_instruction = instruction,
            tools = db_tools,
            device_note = device_note,
        )
    else:
        device_note = ""
        chat = NAL_Chat_Session(
            client = client,
            model = Model_Name,
            system_instruction = instruction,
            tools = db_tools,
        )
    greeting_text = (
        "Hi there! I can answer questions about the roll you just modelled - "
        "counts, medians, breakdowns by neighbourhood, charts and ratio studies. "
        "A good first "
        "question is *'What columns are in the database?'* so I can tell you "
        "exactly what I can see.\n\n"
        ":warning: One thing to know before we start: the `Model_Value` I "
        "report is a **benchmark estimate** from a simple baseline model, not "
        "an assessed value. `Just_Value` is your county's own number."
    )

    ## The local model gets its own second paragraph, because the honest
    ## thing to say about it is specific: it is not that the model is bad,
    ## it is that its TOOL BELT is short. The ratio study made it longer, and
    ## anything outside the tools still depends on SQL a small model wrote.
    ## So the wording is about checking figures, and it says "yet".
    if use_local_engine:
        greeting_text += (
            "\n\n---\n\n**I am running on this computer** - nothing you "
            "type, and no parcel data, leaves the building. " + device_note +
            "\n\n:toolbox: **My tool belt is still limited, and this is a "
            "concept rather than a finished product, so please check my figures "
            "on anything complicated.** I am good at running a ratio study, "
            "counting parcels, filtering and grouping them, and giving you "
            "means, medians and quartiles. Anything beyond those tools relies on "
            "SQL I write myself, which is where I can go wrong. Compare against "
            "the **4. Results** tab. As more tools are added, this warning gets "
            "shorter."
        )
    with st.chat_message("assistant"):
        st.write(greeting_text)
    
    st.session_state["Chat_History"] = [{"role": "assistant", "content": greeting_text}]

    st.session_state["NAL_Chat_Bot"] = chat

    st.session_state['Chat_Launched'] = 'ready'

def Clear_Chat_History():
    st.session_state["Chat_History"] = []
    ## 🩹 This used to clear only what is on screen. The model kept every earlier
    ## question and answer in its own history, so "clearing" a conversation left
    ## it answering in the light of one the reader believed was gone.
    if st.session_state["NAL_Chat_Bot"] is not None:
        st.session_state["NAL_Chat_Bot"].messages = []


def Reset_Chat_Session():
    """Forget the running chat, so the next one starts from the current model.

    🩹 Nothing used to reset this. So the Hosted / Local choice could be made
    once per session, and rebuilding a model left the assistant reading the
    PREVIOUS model's values from a database nobody had rebuilt - while the
    Results tab showed the new ones.

    The local model stays in `st.cache_resource` on purpose: reloading it costs
    far more than keeping it, and only a named cache should ever be cleared.
    """
    connection = st.session_state["Chat_Bot_Data_Base"]
    if connection is not None:
        try:
            connection.close()
        except Exception as e:
            print(f' - Could not close the chat database: {e}')
    st.session_state["Chat_Bot_Data_Base"] = None
    st.session_state["NAL_Chat_Bot"] = None
    st.session_state["Chat_History"] = []
    st.session_state["Chat_Launched"] = None
    st.session_state["Chat_Start_Problem"] = None    

def response_generator( question ):
    """Ask the question and hand the answer to Streamlit.

    This used to yield one word at a time with `time.sleep(0.05)` between
    them, to imitate typing. On a 200-word answer that is **ten seconds**
    of deliberate waiting added to a reply that had already arrived. The
    time this app actually spends is in the tool calls - querying the
    roll - and Streamlit shows its own running indicator for that.
    """
    st.session_state["Display_Plot"] = 0
    st.session_state["Display_Chart"] = 0

    print('The submitted question is:', question)

    answer = st.session_state["NAL_Chat_Bot"].send_message(question)

    tools_used = st.session_state["NAL_Chat_Bot"].last_tool_calls
    if tools_used:
        print(' - tools used this turn:', ', '.join(tools_used))

    yield answer
    

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


    ## No `value =` here on purpose. The starting value lives in
    ## st.session_state['Valuation_Year'] further up this file, and passing
    ## BOTH a default and a session-state entry makes Streamlit warn:
    ##
    ##     The widget with key "Valuation_Year" was created with a default
    ##     value but also had its value set via the Session State API.
    ##
    ## Only one of the two can win, and which one is not obvious from
    ## reading the code - so we keep the single source of truth.
    ## The year you are valuing TO. It sets Time_to_Sale for every parcel,
    ## so the model prices them as at this year rather than at their sale
    ## date. There is no callback on it any more - the one that was here
    ## swapped a dictionary of dead file paths.
    st.sidebar.number_input("Valuation Year"
                            , key = 'Valuation_Year'
                            , step = 1)

    st.session_state['State_Coords_Selected'] = st.sidebar.selectbox("Select Your Coordinate System"
                                                                    , st.session_state['EPSG_List']
                                                                    , index = None
                                                                    , placeholder = "Select EPSG"
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

    ## 🚫 In the sidebar, so it is on screen whichever tab you are reading.
    ## The results screen carries the full version; this is the one that
    ## never scrolls away.
    st.sidebar.markdown('##')
    st.sidebar.divider()
    st.sidebar.warning(BENCHMARK_SHORT)





    st.markdown('##')

    ## ---- Five tabs, one per step of the job -------------------------------
    ##
    ## This used to be one continuous scroll about 1,400 lines long, with each
    ## stage revealed by a nested `if` - so the page grew downwards as you
    ## worked and you could not tell what was coming next.
    ##
    ## Tabs make the shape of the job visible: load, prepare, build, read,
    ## ask. The order is the order you do them in.
    ##
    ## WARNING: the gating moved INSIDE the tabs, not onto the tab bar. Every
    ## tab is always clickable, and each one checks its own precondition and
    ## says what to do first. Disabling tabs would hide the workflow; this
    ## shows it and still makes it impossible to build a model before merging.
    tab_data, tab_prep, tab_model, tab_results, tab_chat, tab_settings = st.tabs([
        "1. Load Data"
        , "2. Prepare"
        , "3. Build Model"
        , "4. Results"
        , "5. Ask Questions"
        , "6. Settings"
    ])

    with tab_data:
        st.markdown('## Load Your Data')
        st.write("---")

        ## ── Option 1: the bundled Lee County example ─────────────────────────
        st.markdown('#### :package: Option 1 - try it on the bundled example')

        _sample = find_sample_data()

        if _sample["nal"] and _sample["sdf"]:
            st.markdown(
                "One real county roll ships with this repository: **Lee County, "
                "Florida**, used with the county's permission. It is the same data "
                "the notebook walks through, so you can compare the two."
            )
            if st.button("Load the Lee County example"
                         , use_container_width = True
                         , type = "primary"):
                with st.status("Reading the roll... the parcel shapes take a few "
                               "seconds", expanded = True) as _status:
                    st.write("parcels...")
                    st.session_state["NAL"] = pd.read_csv(_sample["nal"], low_memory = False)
                    st.write("sales...")
                    st.session_state["SDF"] = pd.read_csv(_sample["sdf"], low_memory = False)
                    if _sample["shp"]:
                        st.write("parcel shapes... (a GDAL warning on the "
                                 "terminal here is normal - see below)")
                        st.session_state["GIS"] = read_parcel_shapes(_sample["shp"])
                    st.session_state["Data_Source"] = "the bundled Lee County example"
                    _status.update(label = "Lee County loaded.", state = "complete"
                                   , expanded = False)
        else:
            ## The archives are committed; the extracted files are not. So a
            ## fresh clone lands here, and this is the one setup step.
            st.info(
                "The bundled Lee County data has not been unpacked yet. It ships "
                "gzipped to fit on GitHub, and one command unpacks it:\n\n"
                "```\nuv run python scripts/extract_data.py\n```\n\n"
                f"Looking in `{_sample['dir']}`. Nothing is downloaded - the "
                "archives are already in your clone."
            )

        st.markdown('#####')

        ## ── Option 2: the reader's own county ────────────────────────────────
        st.markdown('#### :open_file_folder: Option 2 - use your own county')
        st.markdown(
            "Florida DOR publishes the NAL and SDF files for every county. "
            "Upload yours below. The parcel shapefile is optional - without it "
            "the model uses neighbourhood codes instead of coordinates."
        )

        ##Loading NAL File

        NAL_Files = st.file_uploader("Upload NAL file"
                                    , type=["csv"]) 
    
    
        if NAL_Files is not None:
        # st.markdown(":green[NAL File Loaded!]")
            if st.session_state["NAL"] is None:
                with st.spinner("One second just a little tidying up to do!"):
                    st.session_state["NAL"]  = pd.read_csv(NAL_Files
                                                        , low_memory= False)
            #print(type(NAL))
            st.markdown(""":green[ Your NAL loaded successfully! There are """ + str(st.session_state["NAL"].shape[0]) + """ rows of data in your NAL File ]""")

        st.markdown('#####')
        SDF_Files = st.file_uploader("Upload SDF file"
                                    , type=["csv"]) 

        if SDF_Files is not None:
            if st.session_state["SDF"] is None:
                with st.spinner("One second just a little tidying up to do!"):
                    st.session_state["SDF"]  = pd.read_csv(SDF_Files
                                                            , low_memory= False)
        
            st.markdown(""":green[ Your SDF loaded successfully! There are """ + str(st.session_state["SDF"].shape[0]) + """ rows of data in your SDF File ]""")


        st.markdown('#####')

        ## ⚠️ Why this is a PATH box and not an upload box.
        ##
        ## A shapefile is not one file. It is a set that must stay together:
        ##
        ##     .shp   the shapes themselves
        ##     .dbf   the attribute table - this is where PARCEL_ID lives
        ##     .shx   the index that ties the two together
        ##     .prj   the projection: WHERE on the planet those shapes are
        ##
        ## Upload only the .shp and there is no PARCEL_ID to match against the
        ## roll, so it cannot be used at all. Pointing at the .shp on disk lets
        ## the library pick up its siblings, which is what it expects.
        ##
        ## 📌 If your county did not ship a .prj - ours did not - the shapes
        ## load with no idea where they are. Pick your coordinate system in the
        ## sidebar; for west Florida that is EPSG:2237.
        GIS_File_Path = st.text_input(
            "Path to your parcel shapefile (the .shp)"
            , placeholder = r"C:\path\to\your_county_parcels.shp"
            , help = "Keep the .shp, .dbf, .shx and .prj together in one folder."
        )
    
        if st.button("Load GIS File"):
            with st.status("Loading your GIS File... This may take a few seconds", expanded=True) as status_message:
                ## 🩹 This used to do GIS_File_Path.replace("/", "\\"), which
                ## hardcoded Windows separators and broke any path a macOS or
                ## Linux reader typed. Path() handles either on every platform.
                st.session_state["GIS"] = read_parcel_shapes(Path(GIS_File_Path.strip()))
                status_message.update(label = "Parcel shapes loaded."
                                      , state = "complete"
                                      , expanded = False)

        if st.session_state["GIS"] is not None:
            st.markdown(
                ":green[Parcel shapes loaded: "
                + f"{st.session_state['GIS'].shape[0]:,} parcels, "
                + f"CRS {st.session_state['GIS'].crs}]"
            )
            st.caption(
                "If your terminal showed an `organizePolygons()` warning while "
                "these loaded, nothing is wrong. It means some parcels in your "
                "county are drawn with holes in them, or with too few points to "
                "be a polygon - both normal on a real roll. We leave the warning "
                "visible because it tells you something about your data."
            )


        ## ── What is actually loaded right now ────────────────────────────────
        ## One line, whichever route was taken. It survives a rerun because it
        ## reads session state rather than the button that was clicked.
        if st.session_state["NAL"] is not None or st.session_state["SDF"] is not None:
            _loaded = []
            if st.session_state["NAL"] is not None:
                _loaded.append(f"**{st.session_state['NAL'].shape[0]:,}** parcels")
            if st.session_state["SDF"] is not None:
                _loaded.append(f"**{st.session_state['SDF'].shape[0]:,}** sales")
            if st.session_state["GIS"] is not None:
                _loaded.append(f"**{st.session_state['GIS'].shape[0]:,}** parcel shapes")

            _from = st.session_state["Data_Source"]
            st.success("Loaded" + (f" from {_from}" if _from else "") + ": "
                       + " | ".join(_loaded))

            if st.session_state["SDF"] is None:
                st.warning("A sales file (SDF) is still needed before the data "
                           "can be merged - a model needs sales to learn from.")

        st.markdown('#####')

        ## Everything below needs BOTH files. There is no longer a path where
        ## the app reads a roll off disk by itself, so both must really be here.

    with tab_prep:
        if st.session_state['Cost_Of_Sale_Adjustment_Toggle'] == True:
            st.markdown('##')
            st.markdown('## Set DR 493 Adjustments')
            st.write("---")

            ## No `index =` - the starting code comes from session state, which
            ## is seeded further up. Passing both means this default is silently
            ## discarded; see the note by that seed.
            st.selectbox("Select DOR Code to Adjust"
                            , list(st.session_state["DOR_COS_ADJ_DICT"].keys())
                            , key = 'current_Adjust_dor'
                            , on_change = update_COS_Input
                            , help = "DOR use code. The cost-of-sale adjustment "
                                     "is applied per code, so you can set a "
                                     "different one for vacant land than for "
                                     "single-family homes."
                            )

            st.markdown('##')

            new_COS_ADJ =    st.number_input("Set Adjustments"
                                , key = "COS_ADJ_KEY"
                                , min_value = 0.0
                                , max_value = 2.0
                                , step = .01)                                                    

            st.markdown('##')

            if(st.button("Update the DR 493 Adjustment")):
                with st.spinner("Applying the Update!"):
                    print('The New COS adjustment is ' + str(new_COS_ADJ) + ' for ' + st.session_state['current_Adjust_dor'] )
                    st.session_state["DOR_COS_ADJ_DICT"][st.session_state['current_Adjust_dor'] ] = new_COS_ADJ













            st.markdown('#####')



        st.markdown('##')

        if (st.session_state["SDF"] is None) or (st.session_state["NAL"] is None):
            st.info(
                "Load a parcel file (NAL) and a sales file (SDF) on the "
                "**1. Load Data** tab first. A model needs sales to learn from."
            )
        else:
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



    with tab_model:

        if st.session_state["preprocesing_ran"] != True:
            st.info(
                "Merge your files on the **2. Prepare** tab first. That is "
                "where the parcels, the sales and the parcel shapes are joined "
                "into the single table the model trains on."
            )
        else:
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
                    st.checkbox(
                        "Run Hyperparameter Tuning"
                        , key = "Hyperparameter_Tuning_Toggle"
                        , help = "Searches for better model settings instead of "
                                 "using the defaults. Each trial is scored by "
                                 "cross-validation on the TRAINING data, so the "
                                 "test set stays sealed and the final numbers "
                                 "stay honest. This is the slow part of the app "
                                 "- see the time estimate below."
                    )
                    st.number_input(
                        "Number of Trials"
                        , min_value = 10
                        , max_value = 400
                        , step = 1
                        , key = "Hyperparameter_Tuning_Trials"
                        , help = "How many different settings combinations to "
                                 "try. More is better and slower; poor trials "
                                 "are abandoned early, so the cost is less than "
                                 "it looks."
                    )
                    st.number_input(
                        "Cross-Validation Folds"
                        , min_value = 2
                        , max_value = 10
                        , step = 1
                        , key = "Hyperparameter_Tuning_Folds"
                        , help = "Each trial splits the training data into this "
                                 "many parts, fits on all but one and scores on "
                                 "the part left out, then averages. 5 is the "
                                 "usual choice. Fewer is faster and noisier."
                    )

                    ## ⏱️ Say what this will cost, in minutes, before anybody
                    ## presses the button.
                    ##
                    ## The 0.85 s is MEASURED: 150 trials x 5 folds took 640 s
                    ## and 623 s on two runs over 20,381 training sales, on a
                    ## 24-logical-core desktop. That is 0.85 s per single fold
                    ## fit. At the default 50 trials it was 192 s, or 0.77 s per
                    ## fit, so this errs slightly long at the default.
                    ##
                    ## ⚠️ It is a rough figure and will be optimistic for most
                    ## readers. It scales with the number of training sales,
                    ## and LightGBM uses every core it is given - so a 4-core
                    ## laptop can take several times longer. Better a stated
                    ## estimate that is honest about being rough than a
                    ## progress bar that says "hang on tight" for 40 minutes.
                    SECONDS_PER_FOLD_FIT = 0.85

                    _estimated_minutes = (
                        st.session_state['Hyperparameter_Tuning_Trials']
                        * st.session_state['Hyperparameter_Tuning_Folds']
                        * SECONDS_PER_FOLD_FIT / 60
                    )

                    st.caption(
                        f"⏱️ These settings mean "
                        f"{st.session_state['Hyperparameter_Tuning_Trials']} trials "
                        f"x {st.session_state['Hyperparameter_Tuning_Folds']} folds = "
                        f"up to "
                        f"{st.session_state['Hyperparameter_Tuning_Trials'] * st.session_state['Hyperparameter_Tuning_Folds']:,} "
                        f"model fits, **roughly {_estimated_minutes:.0f} minutes** on a "
                        f"fast desktop and longer on a laptop. Poor trials are "
                        f"abandoned early, so the real time is usually less. "
                        f"Halve the trials to halve the wait."
                    )
                    st.markdown('##')
                    st.caption(
                        "The search range for each setting. Optuna picks "
                        "values between the minimum and maximum you give it. "
                        "Widening a range explores more and needs more trials "
                        "to cover it."
                    )

                    ## None of the boxes below pass a `value =`. Their
                    ## starting numbers are seeded into session_state at the
                    ## top of this file, and passing both makes Streamlit warn
                    ## that only one of them can win. FINDINGS 32.
                    Model_Expand_col1, Model_Expand_col2 = st.columns(2)
                    
                    with Model_Expand_col1:
                        st.number_input("Minimum Number of Estimators"
                                        , key = "HyperParametet_Tuning_N_Estimators_Low")
                        st.number_input("Minimum Number of Leaves"
                                        , key = "HyperParametet_Tuning_num_leaves_Low")
                        st.number_input("Minimum Learning Rate"
                                        , key = "HyperParametet_Tuning_learning_rate_Low")
                        st.number_input("Minimum Subsample"
                                        , key = "HyperParametet_Tuning_subsample_Low")
                        st.number_input("Minimum Feature Fraction"
                                        , key = "HyperParametet_Tuning_feature_fraction_Low")
                        st.number_input("Minimum Max Depth"
                                        , key = "HyperParametet_Tuning_max_depth_Low")
                        st.number_input("Minimum Min Data in Leaf"
                                        , key = "HyperParametet_Tuning_min_data_in_leaf_Low")
                        st.number_input("Minimum Min Sum Hessian in Leaf"
                                        , key = "HyperParametet_Tuning_min_sum_hessian_in_leaf_Low")
                        

                    with Model_Expand_col2:
                        st.number_input("Maximum Number of Estimators"
                                        , key = "HyperParametet_Tuning_N_Estimators_High")
                        st.number_input("Maximum Number of Leaves"
                                        , key = "HyperParametet_Tuning_num_leaves_High")
                        st.number_input("Maximum Learning Rate"
                                        , key = "HyperParametet_Tuning_learning_rate_High")
                        st.number_input("Maximum Subsample"
                                        , key = "HyperParametet_Tuning_subsample_High")
                        st.number_input("Maximum Feature Fraction"
                                        , key = "HyperParametet_Tuning_feature_fraction_High")
                        st.number_input("Maximum Max Depth"
                                        , key = "HyperParametet_Tuning_max_depth_High")
                        st.number_input("Maximum Min Data in Leaf"
                                        , key = "HyperParametet_Tuning_min_data_in_leaf_High")
                        st.number_input("Maximum Min Sum Hessian in Leaf"
                                        , key = "HyperParametet_Tuning_min_sum_hessian_in_leaf_High")
                        

                    st.markdown('##')

                st.markdown('##')

                build_model_button = st.form_submit_button("Build Model")

                ## 🩹 Pressing this with nothing ticked used to divide by zero
                ## and show a raw traceback, because number_of_models is 0. That
                ## is an easy first click for somebody new, so say what to do.
                nothing_to_build = not any(
                    st.session_state[_model_flag]
                    for _model_flag in ("SFR_model_binary"
                                        , "Vac_Res_model_binary"
                                        , "Condo_model_binary"
                                        , "Commercial_model_binary")
                )

                if build_model_button and nothing_to_build:
                    st.warning("Tick at least one kind of property above, then "
                               "press **Build Model** again.")

                if build_model_button and not nothing_to_build:

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
                            ## A chat started before this build would go on
                            ## answering from the previous model's values.
                            Reset_Chat_Session()
                            


                        # for percent_complete in range(100):
                        #     time.sleep(0.05)
                            # my_bar.progress(percent_complete + 1, text='**Please hang on tight we are building something cool together!**')



                            
                    with response_container.container():
                        st.success(f'✅  We have successfully built some really cool models!  :tada:')  
                        st.session_state["Model_ran"] = True


    with tab_results:

        if st.session_state["Model_ran"] != True:
            st.info(
                "Build a model on the **3. Build Model** tab first, and its "
                "performance will appear here."
            )
        else:
            # with st.spinner('Wait for it....'):
            #     time.sleep(5)
            st.markdown('##')
            ## 🚫 The first thing on the results screen, before any number.
            st.warning(BENCHMARK_FULL)

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
                sfr_m_middle.markdown("**Training Performance**")
                sfr_m_right.markdown("**Test Performance**")
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
                sfr_perf_color = Range_Colour(train_stat_weighted_mean_sale_ratio, "level", "Single Family")
                sfr_m_middle.markdown( ":" + sfr_perf_color + "["  + str(train_stat_weighted_mean_sale_ratio) + "]"   )
                sfr_perf_color = Range_Colour(test_stat_weighted_mean_sale_ratio, "level", "Single Family")
                sfr_m_right.markdown( ":" + sfr_perf_color + "["  + str(test_stat_weighted_mean_sale_ratio) + "]"   )

                sfr_m_left.markdown("**COD**")
                sfr_perf_color = Range_Colour(train_stat_COD, "cod", "Single Family")
                sfr_m_middle.markdown( ":" + sfr_perf_color + "["  + str(train_stat_COD) + "]"   )
                sfr_perf_color = Range_Colour(test_stat_COD, "cod", "Single Family")
                sfr_m_right.markdown( ":" + sfr_perf_color + "["  + str(test_stat_COD) + "]"   )

                sfr_m_left.markdown("**PRD**")
                sfr_perf_color = Range_Colour(train_stat_PRD, "prd", "Single Family")
                sfr_m_middle.markdown( ":" + sfr_perf_color + "["  + str(train_stat_PRD ) + "]"   )
                sfr_perf_color = Range_Colour(test_stat_PRD, "prd", "Single Family")
                sfr_m_right.markdown( ":" + sfr_perf_color + "["  + str(test_stat_PRD) + "]"   )

                sfr_m_left.markdown("**Mean Absolute Error**")
            
                sfr_m_middle.markdown(  str(round(train_stat_MAE,0))  )
        
                sfr_m_right.markdown( str(round(test_stat_MAE,0)) )


                if  st.session_state['Model_Performance_Toggle']:
                    st.markdown('#####')
                    st.markdown('#####')

                    st.markdown('#### Model vs county roll, on identical sales (SFR Model)')

                    ## ⚖️ Both bars come from the SAME sales - the ones
                    ## where the model's ratio AND the county's ratio are
                    ## both inside 0.3 to 1.7. Before, each side was
                    ## measured on the subset kindest to itself.
                    st.caption(
                        "Both bars are measured on the same "
                        + f"{st.session_state['sfr_preformance_dict']['Comparable Records']:,} "
                        + "sales - those where neither the model's ratio "
                        + "nor the county's is an outlier. That is what "
                        + "makes the two comparable."
                    )

                    graph_metric = st.radio(
                                            "Model Metric:"
                                            ,['R squared' , 'Weighted Mean Sale Ratio' , 'COD' , 'PRD' , 'Mean Absolute Error']
                                            , horizontal = True
                                            , index = 0
                                        )                    


                    if graph_metric == 'R squared':  
                            
                        graph_data = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["sfr_preformance_dict"]["Head to Head Model R Squared"] , st.session_state["sfr_preformance_dict"]["Head to Head County R Squared"]] }
                        graph_lab = 'R Squared'

                    elif graph_metric == 'Weighted Mean Sale Ratio':
                        graph_data = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["sfr_preformance_dict"]["Head to Head Model Weighted Mean"] , st.session_state["sfr_preformance_dict"]["Head to Head County Weighted Mean"]] }
                        graph_lab = 'Weighted Mean Sale Ratio'

                    elif graph_metric == 'COD':
                        graph_data = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["sfr_preformance_dict"]["Head to Head Model COD"] , st.session_state["sfr_preformance_dict"]["Head to Head County COD"]] }
                        graph_lab = 'COD'

                    elif graph_metric == 'PRD':
                        graph_data = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["sfr_preformance_dict"]["Head to Head Model PRD"] , st.session_state["sfr_preformance_dict"]["Head to Head County PRD"]] }
                        graph_lab = 'PRD'    

                    elif graph_metric == 'Mean Absolute Error':     
                        graph_data = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["sfr_preformance_dict"]["Head to Head Model MAE"] , st.session_state["sfr_preformance_dict"]["Head to Head County MAE"]] }
                        graph_lab = 'Mean Absolute Error'


                    st.markdown('#####')

                        
                    st.bar_chart( pd.DataFrame(graph_data)
                                , x = 'Value Source'
                                    
                                , y = 'Metric'
                                    
                                , y_label= graph_lab
                                    
                                , horizontal = True

                    
                                , height = 200
                                , width= 600
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
                Condo_m_middle.markdown("**Training Performance**")
                Condo_m_right.markdown("**Test Performance**")
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
                Condo_perf_color = Range_Colour(Condo_train_stat_weighted_mean_sale_ratio, "level", "Condo")
                Condo_m_middle.markdown( ":" + Condo_perf_color + "["  + str(Condo_train_stat_weighted_mean_sale_ratio) + "]"   )
                Condo_perf_color = Range_Colour(Condo_test_stat_weighted_mean_sale_ratio, "level", "Condo")
                Condo_m_right.markdown( ":" + Condo_perf_color + "["  + str(Condo_test_stat_weighted_mean_sale_ratio) + "]"   )

                Condo_m_left.markdown("**COD**")
                Condo_perf_color = Range_Colour(Condo_train_stat_COD, "cod", "Condo")
                Condo_m_middle.markdown( ":" + Condo_perf_color + "["  + str(Condo_train_stat_COD) + "]"   )
                Condo_perf_color = Range_Colour(Condo_test_stat_COD, "cod", "Condo")
                Condo_m_right.markdown( ":" + Condo_perf_color + "["  + str(Condo_test_stat_COD) + "]"   )

                Condo_m_left.markdown("**PRD**")
                Condo_perf_color = Range_Colour(Condo_train_stat_PRD, "prd", "Condo")
                Condo_m_middle.markdown( ":" + Condo_perf_color + "["  + str(Condo_train_stat_PRD ) + "]"   )
                Condo_perf_color = Range_Colour(Condo_test_stat_PRD, "prd", "Condo")
                Condo_m_right.markdown( ":" + Condo_perf_color + "["  + str(Condo_test_stat_PRD) + "]"   )

                Condo_m_left.markdown("**Mean Absolute Error**")
            
                Condo_m_middle.markdown(  str(round(Condo_train_stat_MAE,0))  )
        
                Condo_m_right.markdown( str(round(Condo_test_stat_MAE,0)) )


                if  st.session_state['Model_Performance_Toggle']:
                    st.markdown('#####')
                    st.markdown('#####')

                    st.markdown('#### Model vs county roll, on identical sales (Condo Model)')

                    ## ⚖️ Both bars come from the SAME sales - the ones
                    ## where the model's ratio AND the county's ratio are
                    ## both inside 0.3 to 1.7. Before, each side was
                    ## measured on the subset kindest to itself.
                    st.caption(
                        "Both bars are measured on the same "
                        + f"{st.session_state['Condo_preformance_dict']['Comparable Records']:,} "
                        + "sales - those where neither the model's ratio "
                        + "nor the county's is an outlier. That is what "
                        + "makes the two comparable."
                    )

                    graph_metric_con = st.radio(
                                            "Model Metric:"
                                            ,['R squared ' , 'Weighted Mean Sale Ratio ' , 'COD ' , 'PRD ' , 'Mean Absolute Error ']
                                            , horizontal = True
                                            , index = 0
                                        )                    


                    if graph_metric_con == 'R squared ':  
                            
                        graph_data_con = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Condo_preformance_dict"]["Head to Head Model R Squared"] , st.session_state["Condo_preformance_dict"]["Head to Head County R Squared"]] }
                        graph_lab_con = 'R Squared'

                    elif graph_metric_con == 'Weighted Mean Sale Ratio ':
                        graph_data_con = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Condo_preformance_dict"]["Head to Head Model Weighted Mean"] , st.session_state["Condo_preformance_dict"]["Head to Head County Weighted Mean"]] }
                        graph_lab_con = 'Weighted Mean Sale Ratio'

                    elif graph_metric_con == 'COD ':
                        graph_data_con = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Condo_preformance_dict"]["Head to Head Model COD"] , st.session_state["Condo_preformance_dict"]["Head to Head County COD"]] }
                        graph_lab_con = 'COD'

                    elif graph_metric_con == 'PRD ':
                        graph_data_con = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Condo_preformance_dict"]["Head to Head Model PRD"] , st.session_state["Condo_preformance_dict"]["Head to Head County PRD"]] }
                        graph_lab_con = 'PRD'    

                    elif graph_metric_con == 'Mean Absolute Error ':     
                        graph_data_con = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Condo_preformance_dict"]["Head to Head Model MAE"] , st.session_state["Condo_preformance_dict"]["Head to Head County MAE"]] }
                        graph_lab_con = 'Mean Absolute Error'


                    st.markdown('#####')

                        
                    st.bar_chart( pd.DataFrame(graph_data_con)
                                , x = 'Value Source'
                                    
                                , y = 'Metric'
                                    
                                , y_label= graph_lab_con
                                    
                                , horizontal = True

                    
                                , height = 200
                                , width= 600
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
                VR_m_middle.markdown("**Training Performance**")
                VR_m_right.markdown("**Test Performance**")
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
                VR_perf_color = Range_Colour(VR_train_stat_weighted_mean_sale_ratio, "level", "Vacant Residential")
                VR_m_middle.markdown( ":" + VR_perf_color + "["  + str(VR_train_stat_weighted_mean_sale_ratio) + "]"   )
                VR_perf_color = Range_Colour(VR_test_stat_weighted_mean_sale_ratio, "level", "Vacant Residential")
                VR_m_right.markdown( ":" + VR_perf_color + "["  + str(VR_test_stat_weighted_mean_sale_ratio) + "]"   )

                VR_m_left.markdown("**COD**")
                VR_perf_color = Range_Colour(VR_train_stat_COD, "cod", "Vacant Residential")
                VR_m_middle.markdown( ":" + VR_perf_color + "["  + str(VR_train_stat_COD) + "]"   )
                VR_perf_color = Range_Colour(VR_test_stat_COD, "cod", "Vacant Residential")
                VR_m_right.markdown( ":" + VR_perf_color + "["  + str(VR_test_stat_COD) + "]"   )

                VR_m_left.markdown("**PRD**")
                VR_perf_color = Range_Colour(VR_train_stat_PRD, "prd", "Vacant Residential")
                VR_m_middle.markdown( ":" + VR_perf_color + "["  + str(VR_train_stat_PRD ) + "]"   )
                VR_perf_color = Range_Colour(VR_test_stat_PRD, "prd", "Vacant Residential")
                VR_m_right.markdown( ":" + VR_perf_color + "["  + str(VR_test_stat_PRD) + "]"   )

                VR_m_left.markdown("**Mean Absolute Error**")
            
                VR_m_middle.markdown(  str(round(VR_train_stat_MAE,0))  )
        
                VR_m_right.markdown( str(round(VR_test_stat_MAE,0)) )

                if  st.session_state['Model_Performance_Toggle']:
                    st.markdown('#####')
                    st.markdown('#####')

                    st.markdown('#### Model vs county roll, on identical sales (Vacant Residential Model)')

                    ## ⚖️ Both bars come from the SAME sales - the ones
                    ## where the model's ratio AND the county's ratio are
                    ## both inside 0.3 to 1.7. Before, each side was
                    ## measured on the subset kindest to itself.
                    st.caption(
                        "Both bars are measured on the same "
                        + f"{st.session_state['Vac_Res_preformance_dict']['Comparable Records']:,} "
                        + "sales - those where neither the model's ratio "
                        + "nor the county's is an outlier. That is what "
                        + "makes the two comparable."
                    )

                    graph_metric_VR = st.radio(
                                            "Model Metric:"
                                            ,['R squared  ' , 'Weighted Mean Sale Ratio  ' , 'COD  ' , 'PRD  ' , 'Mean Absolute Error  ']
                                            , horizontal = True
                                            , index = 0
                                        )                    


                    if graph_metric_VR == 'R squared  ':  
                            
                        graph_data_VR = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Vac_Res_preformance_dict"]["Head to Head Model R Squared"] , st.session_state["Vac_Res_preformance_dict"]["Head to Head County R Squared"]] }
                        graph_lab_VR = 'R Squared'

                    elif graph_metric_VR == 'Weighted Mean Sale Ratio  ':
                        graph_data_VR = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Vac_Res_preformance_dict"]["Head to Head Model Weighted Mean"] , st.session_state["Vac_Res_preformance_dict"]["Head to Head County Weighted Mean"]] }
                        graph_lab_VR = 'Weighted Mean Sale Ratio'

                    elif graph_metric_VR== 'COD  ':
                        graph_data_VR = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Vac_Res_preformance_dict"]["Head to Head Model COD"] , st.session_state["Vac_Res_preformance_dict"]["Head to Head County COD"]] }
                        graph_lab_VR = 'COD'

                    elif graph_metric_VR == 'PRD  ':
                        graph_data_VR = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Vac_Res_preformance_dict"]["Head to Head Model PRD"] , st.session_state["Vac_Res_preformance_dict"]["Head to Head County PRD"]] }
                        graph_lab_VR = 'PRD'    

                    elif graph_metric_VR == 'Mean Absolute Error  ':     
                        graph_data_VR = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Vac_Res_preformance_dict"]["Head to Head Model MAE"] , st.session_state["Vac_Res_preformance_dict"]["Head to Head County MAE"]] }
                        graph_lab_VR = 'Mean Absolute Error'


                    st.markdown('#####')

                        
                    st.bar_chart( pd.DataFrame(graph_data_VR)
                                , x = 'Value Source'
                                    
                                , y = 'Metric'
                                    
                                , y_label= graph_lab_VR
                                    
                                , horizontal = True

                    
                                , height = 200
                                , width= 600
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
                Com_m_middle.markdown("**Training Performance**")
                Com_m_right.markdown("**Test Performance**")
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
                Com_perf_color = Range_Colour(Com_train_stat_weighted_mean_sale_ratio, "level", "Commercial")
                Com_m_middle.markdown( ":" + Com_perf_color + "["  + str(Com_train_stat_weighted_mean_sale_ratio) + "]"   )
                Com_perf_color = Range_Colour(Com_test_stat_weighted_mean_sale_ratio, "level", "Commercial")
                Com_m_right.markdown( ":" + Com_perf_color + "["  + str(Com_test_stat_weighted_mean_sale_ratio) + "]"   )

                Com_m_left.markdown("**COD**")
                Com_perf_color = Range_Colour(Com_train_stat_COD, "cod", "Commercial")
                Com_m_middle.markdown( ":" + Com_perf_color + "["  + str(Com_train_stat_COD) + "]"   )
                Com_perf_color = Range_Colour(Com_test_stat_COD, "cod", "Commercial")
                Com_m_right.markdown( ":" + Com_perf_color + "["  + str(Com_test_stat_COD) + "]"   )

                Com_m_left.markdown("**PRD**")
                Com_perf_color = Range_Colour(Com_train_stat_PRD, "prd", "Commercial")
                Com_m_middle.markdown( ":" + Com_perf_color + "["  + str(Com_train_stat_PRD ) + "]"   )
                Com_perf_color = Range_Colour(Com_test_stat_PRD, "prd", "Commercial")
                Com_m_right.markdown( ":" + Com_perf_color + "["  + str(Com_test_stat_PRD) + "]"   )

                Com_m_left.markdown("**Mean Absolute Error**")
            
                Com_m_middle.markdown(  str(round(Com_train_stat_MAE,0))  )
        
                Com_m_right.markdown( str(round(Com_test_stat_MAE,0)) )


                if  st.session_state['Model_Performance_Toggle']:
                    st.markdown('#####')
                    st.markdown('#####')

                    st.markdown('#### Model vs county roll, on identical sales (Commercial Model)')

                    ## ⚖️ Both bars come from the SAME sales - the ones
                    ## where the model's ratio AND the county's ratio are
                    ## both inside 0.3 to 1.7. Before, each side was
                    ## measured on the subset kindest to itself.
                    st.caption(
                        "Both bars are measured on the same "
                        + f"{st.session_state['Commercial_preformance_dict']['Comparable Records']:,} "
                        + "sales - those where neither the model's ratio "
                        + "nor the county's is an outlier. That is what "
                        + "makes the two comparable."
                    )

                    graph_metric_COM = st.radio(
                                            "Model Metric:"
                                            ,['R squared   ' , 'Weighted Mean Sale Ratio   ' , 'COD   ' , 'PRD   ' , 'Mean Absolute Error   ']
                                            , horizontal = True
                                            , index = 0
                                        )                    


                    if graph_metric_COM == 'R squared   ':  
                            
                        graph_data_COM = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Commercial_preformance_dict"]["Head to Head Model R Squared"] , st.session_state["Commercial_preformance_dict"]["Head to Head County R Squared"]] }
                        graph_lab_COM = 'R Squared'

                    elif graph_metric_COM == 'Weighted Mean Sale Ratio   ':
                        graph_data_COM = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Commercial_preformance_dict"]["Head to Head Model Weighted Mean"] , st.session_state["Commercial_preformance_dict"]["Head to Head County Weighted Mean"]] }
                        graph_lab_COM = 'Weighted Mean Sale Ratio'

                    elif graph_metric_COM == 'COD   ':
                        graph_data_COM = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Commercial_preformance_dict"]["Head to Head Model COD"] , st.session_state["Commercial_preformance_dict"]["Head to Head County COD"]] }
                        graph_lab_COM = 'COD'

                    elif graph_metric_COM == 'PRD   ':
                        graph_data_COM = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Commercial_preformance_dict"]["Head to Head Model PRD"] , st.session_state["Commercial_preformance_dict"]["Head to Head County PRD"]] }
                        graph_lab_COM = 'PRD'    

                    elif graph_metric_COM == 'Mean Absolute Error   ':     
                        graph_data_COM = { 'Value Source': ["Model" ,"NAL Values"] , 'Metric' : [st.session_state["Commercial_preformance_dict"]["Head to Head Model MAE"] , st.session_state["Commercial_preformance_dict"]["Head to Head County MAE"]] }
                        graph_lab_COM = 'Mean Absolute Error'


                    st.markdown('#####')

                        
                    st.bar_chart( pd.DataFrame(graph_data_COM)
                                , x = 'Value Source'
                                    
                                , y = 'Metric'
                                    
                                , y_label= graph_lab_COM
                                    
                                , horizontal = True

                    
                                , height = 200
                                , width= 600
                                )


            if (st.session_state['DOR_Stats_Toggle'])  :
                st.markdown('#####')
                st.markdown('#####')
                st.markdown('#### DOR Summary Statistics Tables For Stratum 1 Properties based on Model Values')

                ## ⚠️ These tables are the most CONFUSABLE output in the app,
                ## because they are laid out like the DOR report an assessor
                ## actually files. Same columns, same group structure. So the
                ## reminder goes directly under the heading, not at the
                ## bottom of the page.
                st.warning(
                    "**These are not DOR submissions.** The layout matches "
                    "the report you file, but every value in it is a "
                    "benchmark estimate from this baseline model - not an "
                    "assessed value. " + BENCHMARK_SHORT
                )

                ## 📌 Scoring all qualified sales is CORRECT for this report, and this
                ## note is an explanation rather than a caveat. When a jurisdiction
                ## submits, the Department of Revenue calculates these statistics on
                ## ALL the sales, not on a testing set the jurisdiction kept back.
                ##
                ## 🩹 This note first said these tables "flatter the model", which
                ## framed a correct report as a defect. The user, who does this work,
                ## pointed out the above, and that many offices come from linear
                ## regression, where a test set or out-of-fold scoring can seem
                ## foreign. That second part is what is worth teaching here. Measured
                ## on Lee County: COD 9.36 here, 11.12 on sales the model never saw.
                st.info(
                    "**These tables use all of your qualified sales**, which is how "
                    "the Department of Revenue calculates these statistics when a "
                    "jurisdiction submits its values and sales.\n\n"
                    "**If you are used to linear regression, one thing is different "
                    "here.** About four in every five of these sales were used to "
                    "train this model, and a gradient boosting model can fit the sales "
                    "it learned from far more closely than it can predict new ones. So "
                    "its COD and PRD read better here than on sales it has never seen. "
                    "To see those, use the test column and the model vs county "
                    "comparison above."
                )
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


            ## 🚫 Said again here, at the moment a number is about to leave
            ## the app. On screen the caveat is one scroll away; in a
            ## spreadsheet it is gone entirely, so the last chance to say it
            ## is the second before the click.
            st.warning(
                "**" + BENCHMARK_SHORT + "** The value column in these files "
                "is named `" + BENCHMARK_COLUMN + "` for that reason. If you "
                "share one of these files, the column heading is the only "
                "part of this warning that travels with it."
            )

            if st.session_state["SFR_Model_Predicitons_For_Review"] is not None : 
          
                st.download_button(
                                    label="Download SFR benchmark estimates",
                                    data = convert_df(st.session_state["SFR_Model_Predicitons_For_Review"] ),
                                    file_name = "SFR_Benchmark_Estimates.csv",
                                    icon="🚨",
                                    mime = "text/csv",
                                )


            if st.session_state["Condo_Model_Predicitons_All"] is not None : 
                   
                st.download_button(
                                    label="Download condominium benchmark estimates",
                                    data = convert_df(st.session_state["Condo_Model_Predicitons_All"] ),
                                    file_name = "Condo_Benchmark_Estimates.csv",
                                    icon="🚨",
                                    mime = "text/csv",          
                )

            if st.session_state["Vac_Res_Model_Predicitons_All"] is not None : 
                    
                st.download_button(
                                    label="Download vacant residential benchmark estimates",
                                    data = convert_df(st.session_state["Vac_Res_Model_Predicitons_All"] ),
                                    file_name = "Vac_Res_Benchmark_Estimates.csv",
                                    icon="🚨",
                                    mime = "text/csv",          
                )

            if st.session_state["Commercial_Model_Predicitons_All"] is not None : 
                      
                st.download_button(
                                    label="Download commercial benchmark estimates",
                                    data = convert_df(st.session_state["Commercial_Model_Predicitons_All"] ),
                                    file_name = "Commercial_Benchmark_Estimates.csv",
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
                                    label="Download every benchmark estimate",
                                    data = convert_df(all_data),
                                    file_name = "All_Benchmark_Estimates.csv",
                                    icon="🚨",
                                    mime = "text/csv",          
                )

            st.markdown('#####')
            st.markdown('#####')
            st.markdown('#####')

    with tab_chat:

        if st.session_state["Model_ran"] != True:
            st.info(
                "Build a model on the **3. Build Model** tab first. The "
                "assistant answers questions about the values your model "
                "produced, so it needs those values to exist."
            )
        else:
            st.markdown('#### How About we Discuss your Data!')
            st.write("---")

            ## 📋 These are not suggestions somebody thought sounded good. See
            ## CHAT_EXAMPLE_QUESTIONS for how they are checked.
            with st.expander("What can I ask it?"):
                q_col, limit_col = st.columns(2)
                with q_col:
                    st.markdown("**Questions it has been checked on**")
                    for _example in CHAT_EXAMPLE_QUESTIONS:
                        st.markdown("- %s" % _example)
                with limit_col:
                    st.markdown("**What it cannot do yet**")
                    for _limit in CHAT_KNOWN_LIMITS:
                        st.markdown("- %s" % _limit)
                    st.caption(
                        "Checked on the Lee County roll by "
                        "`scripts/check_chat_assistant.py`, with every number "
                        "compared against the same figure computed separately. "
                        "A check on one roll is not a guarantee on yours."
                    )
            st.markdown('#####')
                
                
            if st.session_state["Chat_Launched"] == None: 

                ## ── Where should the model run? ───────────────────
                ## No `index =` here on purpose, same as the dropdown
                ## below: the starting value comes from session state,
                ## which is seeded near the top of this file. Passing
                ## both means the widget's own default is thrown away,
                ## silently. FINDINGS 32.
                st.radio('Where should the assistant run?'
                         , CHAT_ENGINES
                         , key = 'Chat_Engine_Selected'
                         , horizontal = True)

                if using_local_engine():
                    st.info(
                        "**No parcel data will leave this computer.** The model "
                        "runs here, reads the roll you just modelled, and "
                        "needs no API key and no internet connection - so the "
                        "two boxes below are greyed out.\n\n"
                        ":toolbox: It is a small model with a **tool belt that "
                        "is growing but still limited**, and this is a concept "
                        "rather than a finished product. It can run a ratio "
                        "study, and count, filter, group and summarise your "
                        "roll. **Check its figures on anything more "
                        "complicated.** More tools are planned."
                    )
                    st.caption(
                        f"Model: `{LOCAL_MODEL_ID}`. The packages come from "
                        f"`uv sync --extra slm`; the model weights are a "
                        f"separate one-time download - see the README."
                    )

                ch_col1, ch_col2  = st.columns(2)

                with ch_col1:
                    ## No `index =` here on purpose - the starting value
                    ## comes from session state, which is seeded further
                    ## up. Passing both means the widget's own default is
                    ## ignored, silently.
                    st.selectbox('Pick a chat model'
                                , list(CHAT_MODELS.keys())
                                , key = 'Chat_Model_Selected'
                                , on_change = set_chatmodel
                                , disabled = using_local_engine())
                    #st.write(f"You selected: {st.session_state['Chat_Model_Selected']}")
                    
                with ch_col2:
                    ## Left blank, the key from .env is used. The label is
                    ## visible now - it used to be hidden, which left an
                    ## unexplained text box next to the model picker.
                    st.text_input('Anthropic API key'
                                , key = 'Provided_Anthropic_Key'
                                , type = 'password'
                                , placeholder = 'Leave blank to use your .env file'
                                , on_change = set_User_API_key
                                , disabled = using_local_engine())
                        

                st.markdown('#####')

                    
                    
                Build_Chat_Bot_Button = st.button('Start Chat!')

                if Build_Chat_Bot_Button:
                    ## 🩹 This check used to read
                    ##     if st.session_state["Provided_Google_Key"] is not None
                    ## which was never False: a `st.text_input` with a key
                    ## sets its session-state entry to the empty string, not
                    ## None. So the warning below was unreachable and the app
                    ## went ahead with no key at all.
                    _typed_key = (st.session_state["Provided_Anthropic_Key"] or "").strip()
                    _env_key = st.session_state["Anthropic_Key"]
                    ## The local engine needs no key at all, so it must
                    ## not be held behind one. Somebody who chose Local
                    ## very likely has no key, and that is the point.
                    if using_local_engine() or _typed_key or _env_key:
                        with st.spinner("Hold on Tight walking up the Chat Bot!"):
                            response_container_Chat = st.empty()                                    
                            with response_container_Chat.container():
                                progress_bar_Chat = st.columns([2,5,2])                                
                                with progress_bar_Chat[1]:                                                                            
                                    #my_bar = st.progress(0, text='')  
                                    try:    
                                        ## Decoration fetched from the internet,
                                        ## so not when the promise is that this
                                        ## runs with no connection at all.
                                        if not using_local_engine():
                                            render_animation2()
                                    except:
                                        st.spinner("Im working hard!")
                                st.session_state["Chat_Start_Problem"] = None
                                Build_Chat_Bot_And_Data()
                                response_container_Chat.empty()
                    else:
                        st.warning(
                            'No Anthropic API key found. Put ANTHROPIC_API_KEY '
                            'in your .env file (see .env.example), or paste a '
                            'key into the box above - or switch to **Local** '
                            'above, which runs on this computer and needs no '
                            'key at all.'
                        )

                ## Outside the container above, which is emptied as soon as the
                ## build returns and would take this message with it.
                if st.session_state["Chat_Start_Problem"]:
                    st.error(st.session_state["Chat_Start_Problem"])

            if st.session_state["Chat_Launched"] is not None: 
                 

                st.button('Clear Chat History'
                          , on_click = Clear_Chat_History)

                st.button('Start Over'
                          , help = "End this chat and choose where the assistant "
                                   "runs again."
                          , on_click = Reset_Chat_Session)
                    
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

                    
  
                

                                                



                



    with tab_settings:

        st.markdown('#### Your acceptable ranges')
        st.write(
            "What counts as acceptable is for your jurisdiction to decide. The rules "
            "differ from place to place and they change, so this app does not tell "
            "you. Enter your own here. The colours on the **4. Results** tab and the "
            "chat assistant's answers judge against these straight away."
        )
        st.info(
            "**These boxes start with placeholder numbers, not a standard.** Replace "
            "them with your own jurisdiction's rules before relying on a colour."
        )

        ## One row per property type, because the rules differ by type. Number boxes
        ## rather than an editable table so that the whole path, from a typed number
        ## to a colour on the Results tab, can be tested.
        _setting_widths = [2.2] + [1] * (2 * len(STARTING_RANGES))
        _heading_columns = st.columns(_setting_widths)
        _heading_columns[0].markdown("**Property type**")
        _position = 1
        for _statistic in STARTING_RANGES:
            _short_name = "Level" if _statistic == "level" else RANGE_LABELS[_statistic]
            _heading_columns[_position].markdown(f"**{_short_name} low**")
            _heading_columns[_position + 1].markdown(f"**{_short_name} high**")
            _position += 2

        for _property_type in PROPERTY_TYPES:
            _row_columns = st.columns(_setting_widths)
            _row_columns[0].markdown(_property_type)
            _position = 1
            for _statistic in STARTING_RANGES:
                for _bound in ("low", "high"):
                    ## No `value =` here on purpose. The starting number is seeded
                    ## near the top of this file. Passing both means the widget's
                    ## own default is thrown away, silently. FINDINGS 32.
                    _row_columns[_position].number_input(
                        f"{_property_type} {RANGE_LABELS[_statistic]} {_bound}"
                        , min_value = 0.0
                        , step = 0.01
                        , format = "%.2f"
                        , key = range_key(_property_type, _statistic, _bound)
                        , label_visibility = "collapsed"
                    )
                    _position += 1

        st.caption(
            "Level applies to both the median ratio and the weighted mean ratio. On "
            "the Results tab a figure shows green inside your range and red outside it."
        )

        for _property_type, _ranges in Current_Acceptable_Ranges().items():
            for _statistic, (_low, _high) in _ranges.items():
                if _low > _high:
                    st.warning(
                        f"{_property_type}, {RANGE_LABELS[_statistic]}: the low of "
                        f"{_low:g} is above the high of {_high:g}, so every figure "
                        f"will show as outside the range."
                    )

        st.markdown('#####')
        st.markdown('#### Small groups')
        st.write(
            "A statistic from a handful of sales mostly measures the handful. One "
            "unusual sale can move it a long way. These two numbers decide how the chat "
            "assistant treats a small group, such as a neighbourhood with few sales."
        )
        _floor_column, _warning_column = st.columns(2)
        _floor_column.number_input(
            "Smallest group that gets any statistic"
            , min_value = 1
            , step = 1
            , key = "Minimum_Sales_For_A_Statistic"
        )
        _warning_column.number_input(
            "Warn about any group with fewer sales than this"
            , min_value = 1
            , step = 1
            , key = "Thin_Group_Warning"
            , help = "Groups below this are reported with a warning and kept out of "
                     "any ranking, so that a city with five sales cannot come out "
                     "as the worst in the county."
        )
        st.caption("These settings last until you close or refresh this page.")

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