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

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests



#load data into pandas dataframes
raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/wins_modified_data.csv')

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

#gets all head to head games
def get_game_results(team1,team2):
    all_games=raw_stats.loc[(raw_stats['Team_Abbrev'] == team1) & (raw_stats['Opponent_Abbrev'] == team2)]
    all_games=all_games.drop_duplicates(subset=['game_id'])
    return all_games

    
#gets all team stats of last 3 years
def get_all_stats(team1):
    all_games=raw_stats.loc[(raw_stats['Team_Abbrev'] == team1)]
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

#gets all player stats
def get_player_stats(player):
    player_stats=raw_stats.loc[raw_stats['player']==player]
    final_player_stats=player_stats[['starter', 'mp','fg','fga','fg_pct','fg3','fg3a','fg3_pct'	,'ft','fta','ft_pct','orb','drb','trb','ast','stl','blk','tov','pf','pts','plus_minus','ts_pct','efg_pct','fg3a_per_fga_pct','fta_per_fga_pct','orb_pct','drb_pct','trb_pct','ast_pct','stl_pct','blk_pct','tov_pct','usg_pct','off_rtg','def_rtg','bpm','minutes'
]]
    return final_player_stats

#gets all team aggregated stats
def get_team_stats(team):
    team_stats=raw_stats.loc[raw_stats['Team_Abbrev'] == team]
    final_team_stats=team_stats[['Opponent_Score','Team_Score','Team_pace','Team_efg_pct','Team_tov_pct','Team_orb_pct','Team_ft_rate','Team_off_rtg']]
    return final_team_stats

#get all team aggregated stats including the opponents performance 
def get_team_stats_with_opp(team):
    team_stats=raw_stats.loc[raw_stats['Team_Abbrev'] == team]
    final_team_stats=team_stats[['Opponent_Score','Team_Score','Team_pace','Team_efg_pct','Team_tov_pct','Team_orb_pct','Team_ft_rate','Team_off_rtg','Opponent_pace',
                                 'Opponent_efg_pct','Opponent_tov_pct','Opponent_orb_pct','Opponent_ft_rate']]

    return final_team_stats

def set_wins_column(team1):
   team1_stats=get_team_stats(team1)
   wins=np.zeros(len(team1_stats))
   for i in range(0,len(team1_stats)):
       if team1_stats['Team_Score'].iloc[i] > team1_stats['Opponent_Score'].iloc[i]:
           wins[i]=wins[i]+1
   return wins

 

def train_model_on_team_with_players(team1,inactive_list):
    team1_stats=get_all_stats(team1)
    #team1_stats['W_L']=set_wins_column(team1)
    
    team1_stats=team1_stats[~team1_stats['player'].isin(inactive_list)]
    t1_labels=team1_stats['W_L'].values

    categorical_columns = ['H_A']
    for column in categorical_columns:
        tempdf = pd.get_dummies(team1_stats[column], prefix=column)
        team1_stats = pd.merge(
            left=team1_stats,
            right=tempdf,
            left_index=True,
            right_index=True,
        )
        team1_stats = team1_stats.drop(columns=column)
    
    team1_stats=team1_stats.drop(['season','W_L','Team_Abbrev', 'Opponent_Abbrev', 'DKP_per_minute', 'FDP_per_minute', 'SDP_per_minute','Opponent_Score','Team_Score','mp','Inactives',
                                       'Opponent_Score','game_id','game_date','player_id','Unnamed: 0','Team_pace','Team_efg_pct','Team_tov_pct',
                                       'Team_orb_pct','Team_ft_rate','Team_off_rtg','Opponent_pace','Opponent_off_rtg','player',
                                       'Opponent_efg_pct','Opponent_tov_pct','Opponent_orb_pct','Opponent_ft_rate'],axis=1)
 
    X_train, X_test, y_train, y_test = train_test_split(
    team1_stats,t1_labels, test_size=0.2, random_state=42)

    model = XGBClassifier()
    model.fit(X_train,y_train)
    return model
    

def train_model_on_team(team1):
   team1_stats=get_team_stats_with_opp(team1)
   #team1_stats['W_L']=set_wins_column(team1)

   t1_labels=team1_stats['W_L'].values
   team1_stats=team1_stats.drop(['W_L','Opponent_Score','Team_Score'],axis=1)
   X_train, X_test, y_train, y_test = train_test_split(
    team1_stats,t1_labels, test_size=0.2, random_state=42)

   model = XGBClassifier()
   model.fit(X_train,y_train)
   return model

def predict_winner(team1,team2,num_simulations,with_player_data):
    if with_player_data:
         t1_model=train_model_on_team_with_players(team1,[])
         team2_stats=get_all_stats(team2)
         
         categorical_columns = ['H_A']
         for column in categorical_columns:
            tempdf = pd.get_dummies(team2_stats[column], prefix=column)
            team2_stats = pd.merge(
                left=team2_stats,
                right=tempdf,
                left_index=True,
                right_index=True,
            )
            team2_stats = team2_stats.drop(columns=column)

         team2_stats=team2_stats.drop(['season','Team_Abbrev', 'Opponent_Abbrev', 'DKP_per_minute', 'FDP_per_minute', 'SDP_per_minute','Opponent_Score','Team_Score','mp','Inactives',
                                       'Opponent_Score','game_id','game_date','player_id','Unnamed: 0','Team_pace','Team_efg_pct','Team_tov_pct',
                                       'Team_orb_pct','Team_ft_rate','Team_off_rtg','Opponent_pace','Opponent_off_rtg','player',
                                       'Opponent_efg_pct','Opponent_tov_pct','Opponent_orb_pct','Opponent_ft_rate'],axis=1)

  
    else:
        t1_model=train_model_on_team(team1)
        team2_stats=get_team_stats_with_opp(team2).drop(['Opponent_Score','Team_Score'],axis=1)
   
    columns=team2_stats.columns
    team1_wins=0
    total_games_simulated=0

    for i in range(0,num_simulations):
        t1_predictions=t1_model.predict(team2_stats)
       
        for i in range(0,len(t1_predictions)):
            total_games_simulated+=1
            if int(t1_predictions[i])==1:
                team1_wins+=1

    explainer = shap.TreeExplainer(t1_model)
    X_importance = team2_stats
    shap_values = explainer.shap_values(X_importance)     
    shap.summary_plot(shap_values, X_importance,feature_names=columns,title=team1,show=False)
    plt.title(team1)
    plt.show()
    print('Percentage of the time '+team1+' should win',team1_wins/total_games_simulated)

#predict_winner('NOP','GSW',1,True)
#predict_winner('GSW','NOP',1,True)



