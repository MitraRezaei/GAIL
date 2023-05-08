"""
Step 1: Read and pre-process experts' trajectories
"""
import pandas as pd
import torch
import numpy as np


def read_file():
    # -------------------------------------1----------------------------------------------
    # load the pedestrian data from the CSV file
    pedestrian_data = pd.read_csv('../data/vadere_csv.csv').replace([np.inf, -np.inf], 1000).astype(float)

    # extract the relevant features (e.g., omega, velocity)
    # states = pedestrian_data.drop(['w', 'v'], axis=1).to_numpy()
    states = pedestrian_data[['id','x_world', 'y_world', 'dist_bin_0', 'dist_bin_1', 'dist_bin_2',
                              'speed_bin_0', 'speed_bin_1', 'speed_bin_2',
                              'obs_dist_bin_0', 'obs_dist_bin_1', 'obs_dist_bin_2'
                              ]].to_numpy()
    actions = pedestrian_data[['w', 'v']].to_numpy()

    # split the data into training and testing sets
    train_ratio = 0.8
    train_size = int(train_ratio * len(pedestrian_data))
    indices = np.random.permutation(len(pedestrian_data))

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_states = states[train_indices]
    train_actions = actions[train_indices]
    test_states = states[test_indices]
    test_actions = actions[test_indices]

    # convert the training data to tensors
    train_states = torch.tensor(train_states, dtype=torch.float32)
    train_actions = torch.tensor(train_actions, dtype=torch.float32)

    # convert the testing data to tensors
    test_states = torch.tensor(test_states, dtype=torch.float32)
    test_actions = torch.tensor(test_actions, dtype=torch.float32)

    return train_states, train_actions, test_states, test_actions, train_states.shape[1], train_actions.shape[1]


# print(read_file())



