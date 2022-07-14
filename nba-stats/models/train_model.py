import shap
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


def train_model_on_all():
    raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/wins_modified_data.csv')
    labels=raw_stats['W_L'].values
    
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
    
    raw_stats=raw_stats.drop(['season','W_L','Team_Abbrev', 'Opponent_Abbrev', 'DKP_per_minute', 'FDP_per_minute', 'SDP_per_minute','Opponent_Score','Team_Score','mp','Inactives','Unnamed: 0.1',
                                       'Opponent_Score','game_id','game_date','player_id','Unnamed: 0','Team_pace','Team_efg_pct','Team_tov_pct','player','W_L',
                                       'Team_orb_pct','Team_ft_rate','Team_off_rtg','Unnamed: 0.1','H_A_A','H_A_H'],axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
    raw_stats,labels, test_size=0.2, random_state=42)

    model = XGBClassifier()
    model.fit(X_train,y_train)
    return model
    




def predict_winner_on_all(team1,team2,num_simulations):
    t1_model=train_model_on_all()
    stats=get_both_team_stats(team1,team2)
   
    
    categorical_columns = ['H_A']
    for column in categorical_columns:
            tempdf = pd.get_dummies(stats[column], prefix=column)
            team1_stats = pd.merge(
                left=stats,
                right=tempdf,
                left_index=True,
                right_index=True,
            )
            stats = stats.drop(columns=column)

    stats=stats.drop(['season','Team_Abbrev', 'Opponent_Abbrev', 'DKP_per_minute', 'FDP_per_minute', 'SDP_per_minute','Opponent_Score','Team_Score','mp','Inactives','Unnamed: 0.1',
                                       'Opponent_Score','game_id','game_date','player_id','Unnamed: 0','Unnamed: 0.1','Team_pace','Team_efg_pct','Team_tov_pct',
                                       'Team_orb_pct','Team_ft_rate','Team_off_rtg','player','W_L'],axis=1)


 
   
    columns=stats.columns
    
    team1_wins=0
    team2_wins=0
    total_games_simulated=0

    for i in range(0,num_simulations):
        t1_predictions=t1_model.predict(stats)
        #t2_predictions=t1_model.predict(team2_stats)[0:3000]

    for pred in t1_predictions:
        total_games_simulated+=1
        if pred == 0:
            team2_wins+=1
        else:
            team1_wins+=1
    
    explainer = shap.TreeExplainer(t1_model)
    X_importance = stats
    shap_values = explainer.shap_values(X_importance)     
    shap.summary_plot(shap_values, X_importance,feature_names=columns,title=team1,show=False)
    plt.title(team1)
    plt.show()
    

    print('Percentage of the time '+team1+' should win',team1_wins/total_games_simulated)
    print('Percentage of the time '+team2+' should win',team2_wins/total_games_simulated)

predict_winner_on_all('LAL','PHX',10)