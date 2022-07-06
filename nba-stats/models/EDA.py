# -*- coding: utf-8 -*-
"""
Main file for the Nba predictions
"""

import pandas as pd
import numpy as np
import os as os
import scipy
import sklearn
import sys


#load data into pandas dataframes
raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/modified_data.csv')
#team_raptor=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/modern_RAPTOR_by_team.csv')
#player_raptor=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/modern_RAPTOR_by_player.csv')

#temp feature creation to be modified in the build_features section
raw_stats['game_date']=pd.to_datetime(raw_stats['game_date'])


#defining some useful functions for grabbing data for the dashboard

#get team score,opponent score where team1 and team2 in dataframe
def get_previous_matchup_results(team1,team2):
    all_games=raw_stats.loc[(raw_stats['Team_Abbrev'] == team1) & (raw_stats['Opponent_Abbrev'] == team2)]
    all_games=all_games.drop_duplicates(subset=['game_id'])
    team1wins=0
    team2wins=0
    return all_games
    
print(get_previous_matchup_results('IND','BRK'))
      
"""
def get_player_stats_last_five_games(player):
    pass

def get_team_stats_last_five_games(team):
    pass

def get_player_stats(player,num_seasons):
    pass

def get_team_stats(team,num_seasons):
    pass

def get_player_raptor(player):
    pass

def get_team_raptor(team):
    pass

def run_predictions(team1,team2):
    pass


def get_odds(team1,team2):
    pass


def get_head_to_head(team1,team2):
    dashboard=pd.DataFrame()
    dashboard
    return dashboard
 """

