"""
Step 3: Define a discriminator network
    The discriminator is responsible for distinguishing between the expert data and the synthetic data generated by
    the generative model. This model can also take many forms, such as a neural network. The model should take as input
    the state and action and output a probability that the data came from the expert data
"""
import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=(128, 128), activation='tanh'):
        super(Discriminator, self).__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_size + action_size
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.logic = nn.Linear(last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        x = torch.sigmoid(self.logic(x))

        return x


