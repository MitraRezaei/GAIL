import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import recorder


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.Sigmoid()(self.fc3(x))
        return x


class Generator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = nn.ReLU()(self.fc1(state))
        x = nn.ReLU()(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

data = pd.read_csv('data.csv').values.astype(np.float32)
states = data[:, :-1]
actions = data[:, -1:]

def discriminator_loss(expert_output, generator_output):
    expert_loss = torch.mean(torch.log(expert_output))
    generator_loss = torch.mean(torch.log(1 - generator_output))
    return -(expert_loss + generator_loss)

def generator_loss(generator_output):
    return -torch.mean(torch.log(generator_output))


state_dim = states.shape[1]
action_dim = actions.shape[1]

discriminator = Discriminator(state_dim, action_dim)
generator = Generator(state_dim, action_dim)

discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)

num_epochs = 1000
batch_size = 128
num_batches = int(states.shape[0] / batch_size)

for epoch in range(num_epochs):
    for batch in range(num_batches):
        # Train discriminator
        discriminator_optimizer.zero_grad()

        expert_states = torch.tensor(states[batch * batch_size:(batch + 1) * batch_size])
        expert_actions = torch.tensor(actions[batch * batch_size:(batch + 1) * batch_size])
        expert_outputs = discriminator(expert_states, expert_actions)

        generator_states = torch.tensor(states[batch * batch_size:(batch + 1) * batch_size])
        generator_actions = generator(generator_states).detach()
        generator_outputs = discriminator(generator_states, generator_actions)

        d_loss = discriminator_loss(expert_outputs, generator_outputs)
        d_loss.backward()
        discriminator_optimizer.step()

        # Train generator
        generator_optimizer.zero_grad()

        generator_states = torch.tensor

        g_loss = generator_loss(generator_outputs)
        g_loss.backward()
        generator_optimizer.step()

# Print loss every 100 epochs
if epoch % 100 == 0:
    print(f"Epoch {epoch}, Discriminator loss: {d_loss}, Generator loss: {g_loss}")
