# -*- coding: utf-8 -*-
"""
Main file for the Nba predictions
"""

from audioop import avg
import itertools
from pickletools import TAKEN_FROM_ARGUMENT1
import shap
from statistics import mean
import pandas as pd
import numpy as np
import os as os
import scipy
import sys

import xgboost 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()


#load data into pandas dataframes
raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/modified_data.csv')
team_raptor=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/modern_RAPTOR_by_team.csv')
player_raptor=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/modern_RAPTOR_by_player.csv')

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
    final_team_stats=team_stats[['Team_pace','Team_efg_pct','Team_tov_pct','Team_orb_pct','Team_ft_rate','Team_off_rtg']]
    return final_team_stats


def get_player_raptor(player,year):
    temp_player_raptor=player_raptor.loc[player_raptor['player_name']==player]
    temp_player_raptor=temp_player_raptor[['raptor_box_offense','raptor_box_defense','raptor_box_total','raptor_onoff_offense','raptor_onoff_defense','raptor_onoff_total','raptor_offense','raptor_defense','raptor_total','war_total','war_reg_season','war_playoffs','predator_offense','predator_defense','predator_total','pace_impact','season']]
    final_player_raptor=temp_player_raptor.loc[temp_player_raptor['season']==year]
    return final_player_raptor

 
def get_team_raptor(team,year):
    tm_raptor=team_raptor.loc[team_raptor['team'] ==team]
    tm_raptor=tm_raptor[['raptor_box_offense','raptor_box_defense','raptor_box_total','raptor_onoff_offense','raptor_onoff_defense','raptor_onoff_total','raptor_offense','raptor_defense','raptor_total','war_total','war_reg_season','war_playoffs','predator_offense','predator_defense','predator_total','pace_impact','season']]
    final_tm_raptor=tm_raptor.loc[tm_raptor['season']==year]
    final_tm_raptor=final_tm_raptor.apply(mean)
    final_tm_raptor['team']=team
    return final_tm_raptor

def run_predictions(team1,team2):
   team1_raptor=get_team_raptor(team1,2021).to_frame()
   team1_label=get_game_results(team1,team2)['Team_Score'].values
   team1_stats=get_game_results(team1,team2).drop(['mp','H_A','Inactives','Team_Score','game_id','game_date','player','player_id'],axis=1)
   final_df=pd.merge(team1_stats,team1_raptor.T,left_on='Team_Abbrev', right_on='team')
   final_df=final_df.drop(columns=['Team_Abbrev','Opponent_Abbrev','team'],axis=1)
   columns=final_df.columns
   final_df=final_df.values

   team2_raptor=get_team_raptor(team2,2021).to_frame()
   team2_label=get_game_results(team1,team2)['Opponent_Score'].values
   
   team2_stats=get_game_results(team1,team2).drop(['mp','H_A','Inactives','Opponent_Score','game_id','game_date','player','player_id'],axis=1)
   final_df2=pd.merge(team2_stats,team2_raptor.T,left_on='Opponent_Abbrev', right_on='team')
   final_df2=final_df2.drop(columns=['Team_Abbrev','Opponent_Abbrev','team'],axis=1)
   final_df2=final_df2.values
   X_train, X_test, y_train, y_test = train_test_split(
    final_df,team1_label, test_size=0.2, random_state=42)

   model = XGBRegressor()
   model.fit(X_train,y_train)
   t1_predictions=model.predict(X_test)
   y1_actual=y_test
   print('Actual scores',y_test)
   print('predicted scores',t1_predictions)
   X_train, X_test, y_train, y_test = train_test_split(
    final_df2,team2_label, test_size=0.2, random_state=42)

   model = XGBRegressor()
   model.fit(X_train,y_train)
   t2_predictions=model.predict(X_test)
   print('Actual scores',y_test)
   print('predicted scores',t2_predictions)
   y2_actual=y_test

   acc=[]

   for i in range(0,len(y1_actual)):
       acc.append(abs(t1_predictions[i] / y1_actual[i]))
       acc.append(abs(t2_predictions[i] / y2_actual[i]))

   X_importance = X_test

   # Explain model predictions using shap library:
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X_importance)     
   shap.summary_plot(shap_values, X_importance,feature_names=columns)

   return acc



print(run_predictions('NOP','IND'))

"""
def get_odds(team1,team2):
    pass
"""

def get_head_to_head(team1,team2):
  pass
 

