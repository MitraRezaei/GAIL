import gym
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# Define the generator network
class Generator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_GAIL(generator, discriminator, expert_states, expert_actions, device,
               g_optimizer, d_optimizer, expert_returns, gamma):
    env = gym.make('CartPole-v1')

    for episode in range(num_episodes):
        # Generate a trajectory
        states = []
        actions = []
        rewards = []
        state = env.reset()
        for t in range(env._max_episode_steps):
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            action = generator(state)
            action = torch.squeeze(action).cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(torch.tensor(action, device=device, dtype=torch.float32))
            rewards.append(reward)
            if done:
                break
            state = next_state

        T = len(rewards)
        g_loss = 0
        d_loss = 0
        returns = torch.tensor([0.0] * T, device=device)
        running_reward = 0
        for t in range(T - 1, -1, -1):
            running_reward = rewards[t] + gamma * running_reward
            returns[t] = running_reward

        expert_actions = expert_actions.to(device)
        expert_states = expert_states.to(device)
        expert_returns = expert_returns.to(device)

        # Update the generator
        for _ in range(5):
            generator.zero_grad()
            generated_actions = generator(states)
            disc_gen_output = discriminator(states, generated_actions)
            g_loss = -torch.mean(torch.log(disc_gen_output))
            g_loss.backward()
            g_optimizer.step()

        # Update the discriminator
        for _ in range(5):
            discriminator.zero_grad()
            disc_expert_output = discriminator(expert_states, expert_actions)
            disc_gen_output = discriminator(states, generated_actions.detach())
            d_loss = -torch.mean(torch.log(disc_expert_output) + torch.log(1 - disc_gen_output))
            d_loss.backward()
            d_optimizer.step()

        if episode % 100 == 0:
            print('Episode: {}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'.format(
                episode, g_loss.item(), d_loss.item()))


def evaluate_policy(policy, env):
    total_rewards = 0
    for i in range(num_episodes):
        state = env.reset()
        episode_rewards = 0
        done = False
        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_rewards += reward
            state = next_state
        total_rewards += episode_rewards
    avg_rewards = total_rewards / num_episodes
    return avg_rewards


# Evaluate the policy after training

num_episodes = 100
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = 2

generator = Generator(state_dim, action_dim)
avg_rewards = evaluate_policy(generator, env)
print("Average rewards over {} episodes: {}".format(num_episodes, avg_rewards))

# def train(generator, discriminator, expert_states, expert_actions, num_epochs, batch_size, lr, criterion):
#     # set optimizers for both networks
#     optim_G = torch.optim.Adam(generator.parameters(), lr=lr)
#     optim_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
#
#     for epoch in range(num_epochs):
#         for i in range(0, expert_states.shape[0], batch_size):
#             # get the current batch of expert states and actions
#             expert_batch_states = expert_states[i:i + batch_size]
#             expert_batch_actions = expert_actions[i:i + batch_size]
#
#             # convert numpy to tensor
#             expert_batch_states = torch.from_numpy(expert_batch_states).float()
#             expert_batch_actions = torch.from_numpy(expert_batch_actions).float()
#
#             # Generate actions using the generator
#             generated_actions = generator(expert_batch_states)
#
#             # Calculate loss for the discriminator
#             d_real = discriminator(expert_batch_states, expert_batch_actions)
#             d_fake = discriminator(expert_batch_states, generated_actions)
#             ones = torch.ones(batch_size)
#             zeros = torch.zeros(batch_size)
#             d_real_loss = criterion(d_real, ones)
#             d_fake_loss = criterion(d_fake, zeros)
#             d_loss = d_real_loss + d_fake_loss
#
#             # update the parameters of the discriminator
#             optim_D.zero_grad()
#             d_loss.backward(retain_graph=True)
#             optim_D.step()
#
#             # Calculate loss for the generator
#             g_fake = discriminator(expert_batch_states, generated_actions)
#             g_loss = criterion(g_fake, ones)
#
#             # update the parameters of the generator
#             optim_G.zero_grad()
#             g_loss.backward()
#             optim_G.step()
#
#             # Print the loss values for both networks every 100 iterations
#             if (i + 1) % 100 == 0:
#                 print("Epoch [{}/{}], Step [{}/{}], d_real_loss: {:.4f}, d_fake_loss: {:.4f}, g_loss: {:.4f}"
#                       .format(epoch + 1, num_epochs, i + 1, expert_states.shape[0] // batch_size, d_real_loss.item(),
#                               d_fake_loss.item(), g_loss.item()))


# def evaluate_generator(generator, expert_actions, num_eval_samples=100, batch_size=32):
#     generator.eval()
#     with torch.no_grad():
#         generated_actions = []
#         for _ in range(num_eval_samples):
#             noise = torch.randn(batch_size, generator.latent_dim)
#             generated_actions.append(generator(noise).detach().numpy())
#
#         generated_actions = np.concatenate(generated_actions, axis=0)
#         expert_actions = expert_actions.numpy()
#
#         precision = precision_score(expert_actions, generated_actions)
#         recall = recall_score(expert_actions, generated_actions)
#         f1 = f1_score(expert_actions, generated_actions)
#
#         print("Precision: {:.4f} Recall: {:.4f} F1-Score: {:.4f}".format(precision, recall, f1))
# --------------------------------------------------------------------------


# Load the data from the Excel file
# df = pd.read_excel("./data/vadere.xlsx", sheet_name='Sheet2')
#
# # Extract the states and actions from the dataframe
# selected_columns_observations = ["x_world", "y_world", "angOrien_x", "angOrien_y"]
# selected_columns_actions = ["Ang_acc_x", "Ang_acc_y"]
#
# states = df[selected_columns_observations].values
# actions = df[selected_columns_actions].values
#
# # data and labels are the input dataset and its corresponding labels
# train_ratio = 0.8
# train_size = int(df.shape[0] * train_ratio)
#
# train_states = states[:train_size]
# train_actions = actions[:train_size]
#
# test_states = states[train_size:]
# test_actions = actions[train_size:]
#
# # Define the generator and discriminator networks
# generator = Generator()
# discriminator = Discriminator()
#
# # Train the GAIL model
# train(generator, discriminator, train_states, train_actions, num_epochs=100)
#
# # Evaluate the generator on the test data
# evaluate_generator(generator, test_states, test_actions)