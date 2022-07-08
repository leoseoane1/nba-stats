# -*- coding: utf-8 -*-
"""
Main file for the Nba predictions
#goals
#we want to ultimately give a probability on moneyline, spread, and totals
#train on a model on w/l, train a model on point differential, and train a model on points scored both sides
#at prediction time,feed it inactive players,home and away and get a result

#questions

"""


import shap
from statistics import mean
import pandas as pd
import numpy as np
import os as os
import scipy
import sys

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split




#load data into pandas dataframes
raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/modified_data.csv')


#temp feature creation to be modified in the build_features section
raw_stats['game_date']=pd.to_datetime(raw_stats['game_date'])


#defining some useful functions for grabbing data for the dashboard

def get_previous_matchup_results(team1,team2):
    all_games=raw_stats.loc[(raw_stats['Team_Abbrev'] == team1) & (raw_stats['Opponent_Abbrev'] == team2)]
    all_games=all_games.drop_duplicates(subset=['game_id'])
    team1wins=0
    team2wins=0

    for game in all_games.itertuples():
        if game.Team_Score > game.Opponent_Score:
            team1wins+=1
        else:
            team2wins+=1

    win_pct=team1wins/team2wins
    return win_pct

def get_game_results(team1,team2):
    all_games=raw_stats.loc[(raw_stats['Team_Abbrev'] == team1) & (raw_stats['Opponent_Abbrev'] == team2)]
    all_games=all_games.drop_duplicates(subset=['game_id'])
    return all_games
    


def get_player_stats_last_five_games(player):
    player_stats=raw_stats.loc[raw_stats['player']==player]
    player_stats=player_stats.sort_values('game_date', ascending=False).drop_duplicates('game_date').head(5)
    return player_stats



def get_team_stats_last_five_games(team):
    team_stats=raw_stats.loc[raw_stats['Team_Abbrev'] == team]
    team_stats=team_stats.sort_values('game_date', ascending=False).drop_duplicates('game_date').head(5)
    final_team_stats=team_stats[['Team_pace','Team_efg_pct','Team_tov_pct','Team_orb_pct','Team_ft_rate','Team_off_rtg']]
    return final_team_stats

def get_player_stats(player):
    player_stats=raw_stats.loc[raw_stats['player']==player]
    final_player_stats=player_stats[['starter', 'mp','fg','fga','fg_pct','fg3','fg3a','fg3_pct'	,'ft','fta','ft_pct','orb','drb','trb','ast','stl','blk','tov','pf','pts','plus_minus','ts_pct','efg_pct','fg3a_per_fga_pct','fta_per_fga_pct','orb_pct','drb_pct','trb_pct','ast_pct','stl_pct','blk_pct','tov_pct','usg_pct','off_rtg','def_rtg','bpm','minutes'
]]
    return final_player_stats


def get_team_stats(team):
    team_stats=raw_stats.loc[raw_stats['Team_Abbrev'] == team]
    final_team_stats=team_stats[['Team_Abbrev','Team_Score','Team_pace','Team_efg_pct','Team_tov_pct','Team_orb_pct','Team_ft_rate','Team_off_rtg']]
    return final_team_stats



def run_all_predictions(team1,team2):
   labels=raw_stats['Team_Score']
   final_df==raw_stats.loc[raw_stats['Team_Abbrev'] == team1]
   final_df=raw_stats.drop(['Team_Score'],axis=1)

   final_df=final_df.drop(columns=['Team_Abbrev','mp','H_A','Inactives','Opponent_Score','game_id','game_date','player','player_id','Opponent_Abbrev', 'DKP_per_minute', 'FDP_per_minute', 'SDP_per_minute'],axis=1)
   columns=final_df.columns

   X_train, X_test, y_train, y_test = train_test_split(
    final_df,labels, test_size=0.2, random_state=42)

   
   model = XGBRegressor()
   model.fit(X_train,y_train)

   t1_predictions=model.predict(X_test)
   y_actual=y_test.values
   correct_count =0
   incorrect_count=0
   for i in range(0,len(t1_predictions)):

       if abs(int(t1_predictions[i]) - int(y_actual[i])) <= 1:
           
           correct_count+=1
       else:
           incorrect_count+=1
   print(correct_count/len(t1_predictions))

   explainer = shap.TreeExplainer(model)
   X_importance = X_test
   shap_values = explainer.shap_values(X_importance)     
   shap.summary_plot(shap_values, X_importance,feature_names=columns)

print(run_all_predictions('IND','NOP'))




