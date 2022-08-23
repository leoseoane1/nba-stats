
"""
Main file for the Nba predictions
#goals
#we want to ultimately give a probability on moneyline, spread, and point totals
#train on a model on w/l, train a model on point differential, and train a model on points scored both sides
#at prediction time,feed it inactive players,home and away and get a result

#questions

"""


import shap
from statistics import mean
import pandas as pd
import numpy as np
import os as os

from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from EDA import *
from get_stats import *
import streamlit as st

def train_model_on_all():
    raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/wins_modified_data.csv')
    labels=raw_stats['off_rtg'].values

    categorical_columns = ['H_A']
    for column in categorical_columns:
        tempdf = pd.get_dummies(raw_stats[column], prefix=column)
        raw_stats = pd.merge(
            left=raw_stats,
            right=tempdf,
            left_index=True,
            right_index=True,
        )
        raw_stats = raw_stats.drop(columns=column)
    
    raw_stats=raw_stats[['Opponent_pace','Opponent_off_rtg','mp','fg','fga','fg_pct','fg3','fg3a','fg3_pct'	,'ft','fta',
                         'ft_pct','orb','drb','trb','ast','stl','blk','tov','pf','pts','plus_minus','ts_pct','efg_pct','fg3a_per_fga_pct',
                         'fta_per_fga_pct','orb_pct','drb_pct','trb_pct',
                         'ast_pct','stl_pct','blk_pct','tov_pct','usg_pct','off_rtg','def_rtg','bpm','minutes'
                                       'Opponent_efg_pct','Opponent_tov_pct','Opponent_orb_pct','Opponent_ft_rate']]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
    raw_stats,labels, test_size=0.2, random_state=42)

    model = XGBRegressor()
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    res=mean_absolute_error(y_test,preds)
    print(res)
    return model
    
def predict_winner_on_all(player,team2):
    model=train_model_on_all()
    player_stats=get_player_stats(player)
    team2_stats=get_team_stats(team2)
    team2_stats=team2_stats.drop(['Opponent_Score','Team_Score'])
    final_stats=
    player_score_pred=model.predict(final_stats)
    

predict_winner_on_all('Brandon Ingram','IND')

'''
st.title('NBA Prediction App')
with st.form("Predictions"):
    
   team1=st.selectbox('team1',raw_stats['Team_Abbrev'].unique())
   team2=st.selectbox('team2',raw_stats['Team_Abbrev'].unique())

    # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write(predict_winner_on_all(team1,team2,10))
       st.write('team1 last 5 games')
       st.table(dashboard(team1))
       st.write('team2 last 5 games')
       st.table(dashboard(team2))
       st.write('team1 key players last 5 games')
       st.table(get_starter_stats_last_five_games(team1))
       st.write('team2 key players last 5 games')
       st.table(get_starter_stats_last_five_games(team2))
'''

