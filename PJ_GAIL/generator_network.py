"""
Step 2: Define the policy/generator network architecture
    The generative model is responsible for generating synthetic pedestrian behavior that imitates the expert data.
    This model can take many forms, such as a neural network or a probabilistic model. The model should take as input
    the pedestrian state and output a distribution over actions.
"""

# import torch
# import torch.nn as nn
# import config_file
#
#
# class Generator(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size, num_layers=2):
#         super(Generator, self).__init__()
#         self.state_size = state_size
#         self.hidden_size = hidden_size
#         self.action_size = action_size
#         self.num_layers = num_layers
#
#         self.lstm = nn.LSTM(state_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, action_size)
#         # self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#
#     def forward(self, states, noise):
#         states = torch.cat([states, noise], dim=-1)
#         batch_size = states.size(0)
#         h0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
#         c0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
#         if torch.cuda.is_available():
#             h0 = h0.cuda()
#             c0 = c0.cuda()
#         # reshape state
#         states = states.view(batch_size, -1, self.state_size)
#
#         # pass states through LSTM layer
#         out, _ = self.lstm(states, (h0, c0))
#
#         # pass output through fully connected layer and activation function
#         actions = self.fc(out[:, -1, :])
#         # actions = self.relu(actions)
#         actions = self.tanh(actions)
#
#         return actions

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, num_layers=2):
        super(Generator, self).__init__()

        self.state_size = state_size
        self.noise_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # # Define LSTM layers
        # self.lstm = nn.LSTM(input_size=state_size + noise_size,
        #                     hidden_size=hidden_size,
        #                     num_layers=2,
        #                     batch_first=True)
        #
        # # Define output layer
        # self.fc = nn.Linear(hidden_size, state_size)

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    # def forward(self, states, noises):
    #     batch_size = states.size(0)
    #     # Concatenate states and noises
    #     inputs = torch.cat((states, noises), dim=1).unsqueeze(0)
    #
    #     # Pass inputs through LSTM layers
    #     # Reset the hidden state at the beginning of each batch
    #     h0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size)
    #     c0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size)
    #     out, _ = self.lstm(inputs, (h0, c0))
    #
    #     # Get output from last LSTM layer
    #     out = out[:, -1, :]
    #
    #     # Pass output through output layer
    #     generated_states = self.fc(out)
    #
    #     return generated_states

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        outputs = self.fc3(x)

        return outputs

