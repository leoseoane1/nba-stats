import numpy as np
import pandas as pd
from EDA import set_wins_column,get_all_stats


#load data into pandas dataframes
raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/modified_data.csv')

#temp feature creation to be modified in the build_features section
raw_stats['game_date']=pd.to_datetime(raw_stats['game_date'])
new_df=get_all_stats('NOP')
new_df['W_L']=set_wins_column('NOP')

for team in raw_stats['Team_Abbrev'].unique():
    if team =='NOP':
        pass
    else:
        temp_team=get_all_stats(team)
        temp_team['W_L']=set_wins_column(team)
        new_df=new_df.append(temp_team)


new_df.to_csv('C:/Users/leose/nba/nba-stats/src/data/wins_modified_data.csv')