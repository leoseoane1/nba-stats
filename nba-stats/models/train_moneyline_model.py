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
    raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/wins_odds_modified_data.csv')
    raw_stats=raw_stats.drop_duplicates(subset=['game_id'])
    labels=raw_stats['ML'].values

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
    
    raw_stats=raw_stats[['Team_pace','Team_efg_pct','Team_tov_pct','Team_orb_pct','Team_ft_rate','Team_off_rtg','Opponent_pace','Opponent_off_rtg',
                                       'Opponent_efg_pct','Opponent_tov_pct','Opponent_orb_pct','Opponent_ft_rate']]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
    raw_stats,labels, test_size=0.2, random_state=42)

    model = XGBRegressor()
    model.fit(X_train,y_train)
    #preds=model.predict(X_test)
    #res=mean_absolute_error(y_test,preds)
    #print(res)
    return model
    
def predict_winner_on_all(team1,team2):
    model=train_model_on_all()
    team1_stats=get_team_stats_with_opp(team1)
    
    #team2_stats=get_team_stats_with_opp(team2)
    team1_stats=team1_stats[['Team_pace','Team_efg_pct','Team_tov_pct','Team_orb_pct','Team_ft_rate','Team_off_rtg','Opponent_pace','Opponent_off_rtg',
                                       'Opponent_efg_pct','Opponent_tov_pct','Opponent_orb_pct','Opponent_ft_rate']]
    #team2_stats=team2_stats.drop(['Opponent_Score','Team_Score'])
    t1_ml=model.predict(team1_stats)
   # t2_ml=model.predict(team2_stats)
    print(t1_ml)
    #print(t2_ml)
    

predict_winner_on_all('NOP','IND')

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
