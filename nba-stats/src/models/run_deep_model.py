  
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from deep_player_prediction import *
from get_stats import *

def main():
    #with torch.no_grad():
    
        # Retrieve item
        player_stats=get_player_stats('Brandon Ingram')
        player_stats=player_stats[['fg','fga','fg_pct','fg3','fg3a','fg3_pct','ft','fta',
                         'ft_pct','orb','drb','trb','ast','stl','blk','tov','pf','plus_minus','ts_pct','efg_pct','fg3a_per_fga_pct',
                         'fta_per_fga_pct','orb_pct','drb_pct','trb_pct',
                         'ast_pct','stl_pct','blk_pct','tov_pct','usg_pct','off_rtg','def_rtg','bpm','minutes']]

        #opp_stats=get_team_stats('IND')
        #opp_stats=opp_stats[['Team_pace','Team_efg_pct','Team_tov_pct','Team_orb_pct','Team_ft_rate','Team_off_rtg']]
        #final_stats=pd.concat([player_stats,opp_stats])
        final_stats = StandardScaler().fit_transform(player_stats)
        final_stats = torch.from_numpy(final_stats).float()
        final_stats=torch.flatten(final_stats)
        #print(final_stats)
        
        # Loading the saved model
        save_path = './saved_model.pth'
        mlp = MLP()
        mlp.load_state_dict(torch.load(save_path))
        mlp.eval()
       # Generate prediction
        prediction = mlp(final_stats)

        print(prediction)
    

main()