import numpy as np
import pandas as pd
from get_stats import set_wins_column,get_all_stats,set_player_raptor


#load data into pandas dataframes
raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/modified_data.csv')
team_raptor=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/modern_RAPTOR_by_team.csv')
player_raptor=pd.read_csv(r'C:/Users/leose/nba/nba-stats/data/raw/modern_RAPTOR_by_player.csv')

#temp feature creation to be modified in the build_features section
raw_stats['game_date']=pd.to_datetime(raw_stats['game_date'])
player_raptor['season']=pd.to_datetime(player_raptor['season']).dt.strftime("%Y%m%d").astype(int)
raw_stats['season'] = pd.DatetimeIndex(raw_stats['game_date']).year.astype(int)

"""
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
"""
count=0
#new_df[['raptor_box_offense','raptor_box_defense','war_total','war_reg_season','pace_impact','season']]
temp=set_player_raptor('Brandon Ingram')

for player in raw_stats['player'].unique():
    print(player)
    if player =='Brandon Ingram':
        pass
    else:
        temp_player=set_player_raptor(player)
        print(temp_player)
        temp=temp.append(temp_player)
    count+=1
    if count > 2:
        break
print(temp)
