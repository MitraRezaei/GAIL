import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)


# Define the generator network
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


def train(generator, discriminator, expert_states, expert_actions, batch_size, epochs, lr_G, lr_D):
    # Define the loss function and optimizers
    criterion = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D)

    expert_states = torch.from_numpy(expert_states).float()
    expert_actions = torch.from_numpy(expert_actions).float()

    d_loss_expert = [[] for _ in range(epochs)]
    g_loss_expert = [[] for _ in range(epochs)]

    for epoch in range(epochs):
        for i in range(0, expert_states.shape[0], batch_size):
            if (expert_states.shape[0] - (i + batch_size)) <= 0:
                batch_size = expert_states.shape[0] - i
            # Generate fake actions
            generated_actions = generator(expert_states[i:i + batch_size])

            # Discriminator loss with expert actions
            d_expert_logits = discriminator(expert_states[i:i + batch_size], expert_actions[i:i + batch_size])
            d_expert_loss = criterion(d_expert_logits, torch.ones((batch_size, expert_actions.shape[1])))

            # Discriminator loss with generated actions
            d_generated_logits = discriminator(expert_states[i:i + batch_size], generated_actions)
            d_generated_loss = criterion(d_generated_logits, torch.zeros((batch_size, expert_actions.shape[1])))
            g_loss = -torch.mean(d_generated_logits)

            # calculate overall loss for discriminator
            d_loss = (d_expert_loss + d_generated_loss) / 2

            # update generator and discriminator
            g_loss.backward(retain_graph=True)
            optimizer_G.step()

            d_loss = d_loss.clone().detach().requires_grad_(True)
            d_loss.backward(retain_graph=True)
            optimizer_D.step()
        d_loss_expert[epoch] = d_loss.item()
        g_loss_expert[epoch] = g_loss.item()

    # return generator and discriminator after training
    return d_loss_expert, g_loss_expert


def evaluate_generator(generator, val_observations, val_actions):
    # Generate actions for new states using the generator model

    expert_observations = torch.tensor(val_observations).float()
    generated_actions = generator(expert_observations)

    # Compare the generated actions with the expert actions
    expert_actions = torch.tensor(val_actions).float()
    similarity_scores = nn.functional.cosine_similarity(generated_actions, expert_actions)

    # Print the mean and standard deviation of the similarity scores
    print("Mean similarity score: {:.3f}".format(similarity_scores.mean().item()))
    print("Standard deviation of similarity scores: {:.3f}".format(similarity_scores.std().item()))


def plot_loss(num_experts, expert_info, epoch, losses, disc):
    # Plot the convergence curve for each expert
    # The X-axis represents the epoch, and the Y-axis represents the loss.
    # The legend indicates which curve corresponds to which expert.

    # for expert_idx in range(num_experts):
    #     plt.plot(losses[expert_idx])
    for expert_idx in range(num_experts):
        plt.plot(range(epoch), losses[expert_idx])

    # plt.plot(losses)
    plt.xlabel("epoch")
    plt.legend(expert_info)

    if disc:
        plt.ylabel("discriminator_loss")
        plt.savefig('./image/gail/discriminator/d_loss.png')
    else:
        plt.ylabel("generator_loss")
        plt.savefig('./image/gail/generator/g_loss.png')


# -----------------------------Call-------------------------------------------

df = pd.read_excel("./data/vadere.xlsx", sheet_name='Sheet2')
observations = np.array([])
actions = np.array([])
expert_info = []

# Group data by column1 and select distinct values of column2
grouped_df = df.groupby('id')['id'].unique()
num_experts = len(grouped_df)
num_experts_list = []
for i in range(num_experts):
    num_experts_list.append(i)

selected_columns_observations = ["x_world", "y_world", "angOrien_x", "angOrien_y"]
selected_columns_actions = ["Ang_acc_x", "Ang_acc_y"]
dim_state = len(selected_columns_observations)
dim_action = len(selected_columns_actions)

epoch = 100
lr_G = 0.0001
lr_D = 0.0001
batch_size = 32

d_losses = [[0 for _ in range(epoch)] for _ in range(num_experts)]
g_losses = [[0 for _ in range(epoch)] for _ in range(num_experts)]

# Define the generator and discriminator models
generator = Generator(dim_state, dim_action)
discriminator = Discriminator(dim_state, dim_action)

for group_value, num_expert in zip(grouped_df, num_experts_list):
    expert_info.append("Expert {}".format(group_value))
    # Select all data for the current group value
    group_data = df[df['id'].isin(group_value)]

    # Select specific columns from the data
    group_data_observations = group_data[selected_columns_observations]
    group_data_actions = group_data[selected_columns_actions]

    # Append the selected data to the list
    observations = np.array(group_data_observations)
    actions = np.array(group_data_actions)

    train_ratio = 0.8
    train_size = int(observations.shape[0] * train_ratio)

    train_observations = observations[:train_size]
    train_actions = actions[:train_size]

    val_observations = observations[train_size:]
    val_actions = actions[train_size:]

    # Train the generator and discriminator models
    d_loss_expert, g_loss_expert = train(generator, discriminator, train_observations, train_actions, batch_size,
                                         epoch, lr_G, lr_D)

    # d_losses[int(group_value)].append(d_loss_expert)
    # g_losses[int(group_value)].append(g_loss_expert)

    d_losses[num_expert] = d_loss_expert
    g_losses[num_expert] = g_loss_expert

    # Evaluate the generator model
    # evaluate_generator(generator, val_observations, val_actions)

disc = True
plot_loss(num_experts, expert_info, epoch, d_losses, disc)

disc = False
plot_loss(num_experts, expert_info, epoch, g_losses, disc)
