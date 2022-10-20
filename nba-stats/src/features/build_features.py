from tempfile import tempdir
import pandas as pd
import os
'''
raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/ASA_All_NBA_Raw_Data.csv')
team_raptor=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/modern_RAPTOR_by_team.csv')
player_raptor=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/modern_RAPTOR_by_player.csv')

raw_stats['game_date']=pd.to_datetime(raw_stats['game_date'])

raw_stats.to_csv('C:/Users/leose/nba/nba-stats/src/data/modified_data.csv')


raw_stats=pd.read_csv('C:/Users/leose/nba/nba-stats/src/data/wins_modified_data.csv')

'''
'''
categorical_columns = ['H_A']
for column in categorical_columns:
        tempdf = pd.get_dummies(raw_stats[column], prefix=column)
        raw_stats = pd.merge(
            left=raw_stats,
            right=tempdf,
            left_index=True,
            right_index=True,
        )

raw_stats.to_csv('C:/Users/leose/nba/nba-stats/src/data/wins_modified_data.csv')

'''
raw_stats=pd.read_csv('C:/Users/leose/nba/nba-stats/src/data/wins_odds_modified_data.csv')
raw_stats=raw_stats[raw_stats['Team_Abbrev']=='NOP']
raw_stats.to_excel('C:/Users/leose/nba/nba-stats/src/data/wins_odds_modified_data.xlsx')