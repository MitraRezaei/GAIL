import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Load observation and action datasets
# Collect expert demonstrations

# Load data from an Excel file and select specific columns
df = pd.read_excel("./data/vadere.xlsx", sheet_name='Sheet2')

expert_observations_column = ['x_world', 'y_world']
expert_observations = df[expert_observations_column]
expert_observations_np = expert_observations.values
expert_observations = torch.tensor(expert_observations_np, dtype=torch.float32)

expert_actions_column = ['angle']
expert_actions = df[expert_actions_column]
expert_actions_np = expert_actions.values
expert_actions = torch.tensor(expert_actions_np, dtype=torch.float32)

obs_dim = expert_observations.shape[1]
act_dim = expert_actions.shape[1]


# Define the neural network architecture
class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.fc1 = nn.Linear(in_features=obs_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Initialize the policy model
policy_model = PolicyModel()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(policy_model.parameters(), lr=0.001)

num_epochs = 100
# Train the model

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, obs in enumerate(expert_observations):
        optimizer.zero_grad()
        outputs = policy_model(obs)
        loss = criterion(outputs, expert_actions[i])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, running_loss/len(expert_observations)))


# --------------------------------------------------------------------
# Load validation dataset
val_observations = np.array([
[10.36781935, 17.97051302],
[10.36754156, 17.85295801],
[10.36725428, 17.72751366],
[10.36696353, 17.59562109]
])
val_actions = np.array([[0], [-1], [-1], [-1]])

val_observations = torch.tensor(val_observations, dtype=torch.float32)
val_actions = torch.tensor(val_actions, dtype=torch.float32)

# Evaluate the model on validation dataset
with torch.no_grad():
    policy_model.eval()
    total_loss = 0.0
    for i, val_obs in enumerate(val_observations):
        val_outputs = policy_model(val_obs)
        val_loss = criterion(val_outputs, val_actions[i])
        total_loss += val_loss.item()
    avg_loss = total_loss / len(val_observations)
    # print('Validation Loss: {:.4f}'.format(avg_loss))
    print('Output is:', val_outputs)

# Set the model back to training mode
policy_model.train()


# Save the trained policy model
torch.save(policy_model.state_dict(), 'policy_model.pth')
