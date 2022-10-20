
import shap
from statistics import mean
import pandas as pd
import numpy as np
import os

import pickle as pkl
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from get_stats import *
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import streamlit as st

def train_model_on_all():
   
    raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/wins_modified_data.csv')
    labels=raw_stats['Team_Score'].values
    
    raw_stats=raw_stats[['W_L', 'Opponent_Score','Team_pace','Team_efg_pct',
                        'Team_tov_pct','Opponent_efg_pct','Opponent_tov_pct',
                        'Opponent_orb_pct','Opponent_ft_rate','Opponent_off_rtg',
                                       'Team_orb_pct','Team_ft_rate','Team_off_rtg']]

    X_train, X_test, y_train, y_test = train_test_split(
    raw_stats,labels, test_size=0.2, random_state=42)

    model = XGBRegressor()
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    scores=mean_absolute_error(preds,y_test)
    print(scores)
    pkl.dump(model,open('team_finalized_model2.sav','wb'))

    return model

train_model_on_all()

def predict_winner_on_all(team1,team2):

    filename ='team_finalized_model2.sav'
    t1_model = pkl.load(open(filename, 'rb'))
    
    stats=get_game_results(team1,team2)
    actual_scores=stats['Team_Score'].values

    stats=stats[['W_L', 'Opponent_Score','Team_pace','Team_efg_pct',
                        'Team_tov_pct','Opponent_efg_pct','Opponent_tov_pct',
                        'Opponent_orb_pct','Opponent_ft_rate','Opponent_off_rtg',
                                       'Team_orb_pct','Team_ft_rate','Team_off_rtg']]

 
    
   
    t1_predictions=t1_model.predict(stats)

    return 'Game prediction '+str(np.mean(t1_predictions)),'Last five scores '+str(actual_scores)

print(predict_winner_on_all('NOP','LAL'))
print(predict_winner_on_all('LAL','NOP'))

'''
st.title('NBA Prediction App')
with st.form("Predictions"):
    
   team1=st.selectbox('team1',raw_stats['Team_Abbrev'].unique())
   team2=st.selectbox('team2',raw_stats['Team_Abbrev'].unique())

    # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write(predict_winner_on_all(team1,team2))
       st.write(predict_winner_on_all(team2,team1))
       st.write('team1 last 5 games')
       st.table(dashboard(team1))
       st.write('team2 last 5 games')
       st.table(dashboard(team2))
       st.write('team1 key players last head to head games')
       st.table(get_starter_stats_last_five_games(team1))
       st.write('team2 key players last head to head games')
       st.table(get_starter_stats_last_five_games(team2))
       '''