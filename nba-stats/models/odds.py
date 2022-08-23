from statistics import mean
import pandas as pd
import numpy as np
import os as os
import scipy
import sys

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests
from EDA import *
from get_stats import *

raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/wins_odds_modified_data.csv')
raw_stats['game_date']=pd.to_datetime(raw_stats['game_date'])

point_totals=[]
spreads=[]

for odds in raw_stats['Close']:
    if odds > 100:
        point_totals.append(odds)
    else:
        spreads.append(odds)

point_totals=np.repeat(point_totals,2)
spreads=np.repeat(spreads,2)

raw_stats['spreads']=spreads[0:12717]
raw_stats['point_totals']=point_totals

raw_stats.to_csv('C:/Users/leose/nba/nba-stats/src/data/final_wins_odds_modified_data.csv')
