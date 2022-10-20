import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from get_stats import *

class PlayerDataset(torch.utils.data.Dataset):
  '''
  Prepare the dataset for regression
  '''

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)
      

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]
      

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(34, 64),
      nn.ReLU(),
      nn.Linear(64,128),
      nn.ReLU(),
      nn.Linear(128,64),
      nn.ReLU(),
      nn.Linear(64,32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

  
def train():
      # Set fixed random number seed
      torch.manual_seed(42)
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print(device)
      # Load Player dataset
 
      raw_stats=pd.read_csv(r'C:/Users/leose/nba/nba-stats/src/data/wins_modified_data.csv')

      raw_stats=raw_stats[['fg','fga','fg_pct','fg3','fg3a','fg3_pct','ft','fta',
                             'ft_pct','orb','drb','trb','ast','stl','blk','tov','pf','plus_minus','ts_pct','efg_pct','fg3a_per_fga_pct',
                             'fta_per_fga_pct','orb_pct','drb_pct','trb_pct','pts',
                             'ast_pct','stl_pct','blk_pct','tov_pct','usg_pct','off_rtg','def_rtg','bpm','minutes' ]]
      labels=raw_stats['pts']
      raw_stats=raw_stats.drop(['pts'],axis=1)

      X_train, X_test, y_train, y_test = train_test_split(raw_stats.to_numpy(dtype=float),labels.to_numpy(dtype=float), test_size=0.2, random_state=42)



      # Prepare Player dataset
      train_dataset = PlayerDataset(X_train, y_train,scale_data=False)
      test_dataset = PlayerDataset(X_test, y_test,scale_data=False)
      

      trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
      testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)


      # Initialize the MLP
      mlp = MLP()
  
      # Define the loss function and optimizer
      loss_function = nn.L1Loss()
      optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
  
      # Run the training loop
      for epoch in range(0, 5): # 5 epochs at maximum
    
        # Print epoch
        print(f'Starting epoch {epoch+1}')
    
        # Set current loss value
        current_loss = 0.0
    
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
      
          # Get and prepare inputs
          inputs, targets = data
          inputs, targets = inputs.float(), targets.float()
          targets = targets.reshape((targets.shape[0], 1))
      
          # Zero the gradients
          optimizer.zero_grad()
      
          # Perform forward pass
          outputs = mlp(inputs)
      
          # Compute loss
          loss = loss_function(outputs, targets)
      
          # Perform backward pass
          loss.backward()
      
          # Perform optimization
          optimizer.step()
      
          # Print statistics
          current_loss += loss.item()
          if i % 1000 == 0:
              print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
              current_loss = 0.0

        valid_loss = 0.0
        min_valid_loss = np.inf
        mlp.eval()     # Optional when not using Model Specific layer

        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            # Forward Pass
            outputs = mlp(inputs)
            # Find the Loss
            loss = loss_function(outputs,targets)
            # Calculate Loss
            valid_loss += loss.item()

            if min_valid_loss > valid_loss:
                print('Saving The Model')
                min_valid_loss = valid_loss
         
                # Saving State Dict
                torch.save(mlp.state_dict(), 'saved_model.pth')

        print(str(epoch) +' Training Loss: '+str(loss / len(trainloader))+' Validation Loss: '+ str(valid_loss / len(testloader)))

      # Process is complete.
      print('Training process has finished.')

if __name__ == '__main__':
  mlp=MLP()
  mlp.load_state_dict(torch.load('saved_model.pth'))
  mlp.eval()

  val_data=get_player_stats_last_five_games('Brandon Ingram')
  y_val=val_data['pts']
  X_val=val_data[['fg','fga','fg_pct','fg3','fg3a','fg3_pct','ft','fta',
                             'ft_pct','orb','drb','trb','ast','stl','blk','tov','pf','plus_minus','ts_pct','efg_pct','fg3a_per_fga_pct',
                             'fta_per_fga_pct','orb_pct','drb_pct','trb_pct',
                             'ast_pct','stl_pct','blk_pct','tov_pct','usg_pct','off_rtg','def_rtg','bpm','minutes' ]]

  val_dataset=PlayerDataset(X_val.to_numpy(dtype=float),y_val.to_numpy(dtype=float),scale_data=False)
  val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=True, num_workers=1)

  outputs=[]

  for i, data in enumerate(val_loader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))
        # Forward Pass
        outputs.append(mlp(inputs))
   

  print(outputs)
  print(y_val)

