import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
1. Collect expert data: Gather a dataset of expert demonstrations, including the inputs (features) and outputs (actions)
for each time step.
2. Preprocess data: Clean, normalize, and preprocess the collected data to make it suitable for training a machine 
learning model.
3. Split data into training and testing sets: Divide the preprocessed data into a training set and a testing set to 
evaluate the performance of the model.
4. Train the model: Train a traditional machine learning model, such as a decision tree, random forest, or support 
vector machine, on the training set using the inputs as features and the outputs as targets.
5. Evaluate the model: Evaluate the performance of the model on the testing set and compare it to the expert data to 
determine the accuracy of the model.
6. Use the model: Use the trained model to generate actions for new input situations, in effect cloning the expert's 
behavior
"""

# Load data from an Excel file and select specific columns
df = pd.read_excel("./data/vadere.xlsx", sheet_name='Sheet2')
num_epochs = 100

# Gather a dataset for training
def generate_dataset(df, batch_size=32):
    observations = []
    actions = []
    train_loaders = []
    val_loaders = []
    expert_info = []

    # Group data by column1 and select distinct values of column2
    grouped_df = df.groupby('id')['id'].unique()

    num_experts = len(grouped_df)
    selected_columns_observations = ["x_world", "y_world", "angOrien_x", "angOrien_y"]
    selected_columns_actions = ["Ang_acc_x", "Ang_acc_y"]

    state_dim = len(selected_columns_observations)
    action_dim = len(selected_columns_actions)

    # Use a for loop to create the numpy arrays of observations and actions
    for group_value in grouped_df:
        expert_info.append("Expert {}".format(group_value))

        # Select all data for the current group value
        group_data = df[df['id'].isin(group_value)]

        # Select specific columns from the data
        group_data_observations = group_data[selected_columns_observations]
        group_data_actions = group_data[selected_columns_actions]

        # Append the selected data to the list
        observations.append(np.array(group_data_observations))
        actions.append(np.array(group_data_actions))

    for i in range(num_experts):
        expert_observations = observations[i]
        expert_actions = actions[i]

        # Split data to train and test
        train_ratio = 0.8
        train_size = int(expert_observations.shape[0] * train_ratio)
        train_observations = expert_observations[:train_size]
        train_actions = expert_actions[:train_size]

        val_observations = expert_observations[train_size:]
        val_actions = expert_actions[train_size:]

        # Convert the numpy arrays to PyTorch tensors
        train_observations = torch.from_numpy(train_observations).float()
        train_actions = torch.from_numpy(train_actions).float()

        val_observations = torch.from_numpy(val_observations).float()
        val_actions = torch.from_numpy(val_actions).float()

        # Create a TensorDataset from the tensors
        dataset_train = TensorDataset(train_observations, train_actions)

        dataset_val = TensorDataset(val_observations, val_actions)

        # Create a DataLoader from the dataset
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

        # Append the DataLoader to the list of train_loaders
        train_loaders.append(train_loader)

        val_loaders.append(val_loader)

    return train_loaders, val_loaders, num_experts, state_dim, action_dim, expert_info


# Define the network model
class BehavioralCloningNet(nn.Module):
    def __init__(self):
        super(BehavioralCloningNet, self).__init__()
        self.fc1 = nn.Linear(in_features=obs_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=act_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# Train the model
def train_model(models, number_experts, train_loaders):
    learning_rate = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    # Create loss function and optimization function for each model
    criterions = [nn.BCELoss() for i in range(number_experts)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.1) for model in models]
    # train_losses = [[][] for _ in range(number_experts)]

    train_losses = [[0 for j in range(num_epochs)] for i in range(number_experts)]

    # Train each model on the data from its corresponding expert
    for expert_idx in range(number_experts):
        for epoch in range(num_epochs):
            total_loss = 0.0
            for i, data in enumerate(train_loaders[expert_idx], 0):
                inputs, actions = data
                optimizers[expert_idx].zero_grad()
                outputs = models[expert_idx](inputs)
                loss = criterions[expert_idx](outputs, actions)
                loss.backward()
                optimizers[expert_idx].step()
                total_loss += loss.item()
            # train_losses[expert_idx] = total_loss
            train_losses[expert_idx][epoch] = total_loss

    # Save each trained model
    for expert_idx in range(number_experts):
        torch.save({'model_state_dict': models[expert_idx].state_dict(),
                    'optimizer_state_dict': optimizers[expert_idx].state_dict()},
                   f"./policy_model/policy_expert_{expert_idx}.pt")
    return train_losses


# Validate the model
def validate_model(models, number_experts, val_loaders):
    val_losses = [[] for _ in range(number_experts)]
    with torch.no_grad():
        # Evaluate the model on the validation data
        for expert_idx in range(number_experts):
            for i, data in enumerate(val_loaders[expert_idx], 0):
                inputs, labels = data
                outputs = models[expert_idx](inputs)
                loss = nn.MSELoss()(outputs, labels)
                val_losses[expert_idx] = val_losses[expert_idx] + [loss.item()]
            print(expert_idx, inputs, outputs)
    return val_losses


def plot_loss(num_experts, expert_info, losses):
    # Plot the convergence curve for each expert
    # The X-axis represents the epoch, and the Y-axis represents the loss.
    # The legend indicates which curve corresponds to which expert.

    # for expert_idx in range(num_experts):
    #     plt.plot(losses[expert_idx])
    for expert_idx in range(num_experts):
        plt.plot(range(num_epochs), losses[expert_idx], label='Expert ' + str(expert_idx))
        # for lr in range(len(learning_rate)):
        #     plt.plot(losses[expert_idx][lr])
    # plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend(expert_info)
    # plt.show()
    # Save the plot image
    plt.savefig('./image/bc/loss_history.png')


# -----------------------------Call-------------------------------------------

train_loaders, val_loaders, number_experts, obs_size, act_size, expert_info = generate_dataset(df)
models = [BehavioralCloningNet() for i in range(number_experts)]
train_losses = train_model(models, number_experts, train_loaders)
val_losses = validate_model(models, number_experts, val_loaders)
plot_loss(number_experts, expert_info, train_losses)
