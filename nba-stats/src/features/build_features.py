import pandas as pd
import os

raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/ASA_All_NBA_Raw_Data.csv')
team_raptor=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/modern_RAPTOR_by_team.csv')
player_raptor=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/modern_RAPTOR_by_player.csv')

raw_stats['game_date']=pd.to_datetime(raw_stats['game_date'])

raw_stats.to_csv('C:/Users/leose/nba/nba-stats/src/data/modified_data.csv')