"""
Step 4: Train the policy/generator network and discriminator network.
    We aim to minimize the discriminator's ability to differentiate between real and fake data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from generator_network import Generator
from discriminator_network import Discriminator
import numpy as np
import config_file
from read_expert_trajectory import read_file
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt


class GAIL(nn.Module):
    def __init__(self, train_states, train_actions, state_dim, action_dim):
        super(GAIL, self).__init__()
        self.train_states = train_states
        self.train_actions = train_actions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the generator and discriminator models
        self.generator = Generator(state_dim, action_dim, config_file.hidden_size)
        self.discriminator = Discriminator(state_dim, action_dim)

        # Define the optimizers
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=config_file.learning_rate)
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=config_file.learning_rate)

        self.list_loss_discriminator = np.array([])
        self.list_loss_generator = np.array([])

    def loss_discriminator(self, expert_output, generator_output):
        # expert_loss = torch.mean(torch.log(expert_output))
        expert_loss = torch.log(expert_output)
        generator_loss = torch.log(1 - generator_output)
        return -torch.mean(expert_loss + generator_loss)

    def loss_generator(self, generator_output):
        return -torch.mean(torch.log(generator_output))

    def train_discriminator(self, expert_states, generated_actions, expert_actions):
        # Set the discriminator to train mode
        self.discriminator.train()
        self.optimizer_discriminator.zero_grad()

        # Get discriminator outputs for real and generated data
        expert_output = self.discriminator(expert_states, expert_actions)
        generator_output = self.discriminator(expert_states, generated_actions.detach())

        # Compute discriminator loss
        d_loss = self.loss_discriminator(expert_output, generator_output)

        # Optimize discriminator
        d_loss.backward()
        self.optimizer_discriminator.step()

        return self.discriminator, d_loss

    def train_generator(self, expert_states):
        # Set the generator to train mode
        self.generator.train()
        self.optimizer_generator.zero_grad()

        # Generate fake actions from the states
        fake_actions = self.generator(expert_states)

        # Train generator
        fake_output = self.discriminator(expert_states, fake_actions)
        g_loss = self.loss_generator(fake_output)

        # Optimize generator
        g_loss.backward()
        self.optimizer_generator.step()

        return self.generator, fake_actions, g_loss

    def train_gail(self, states, actions):
        # Define hyperParameters and other settings
        num_epochs = config_file.num_epochs
        batch_size = config_file.batch_size

        num_samples = states.size(0)
        num_batches = int(np.ceil(num_samples / batch_size))

        total_d_loss_epoch = 0
        total_g_loss_epoch = 0

        # Loop over epochs
        for epoch in range(num_epochs):
            for i, j in zip(range(0, num_samples, num_batches), range(batch_size)):
                # Get batch of data
                start_idx = i  # batch_size ->num_batches
                end_idx = min((j + 1) * num_batches, num_samples)  # batch_size ->num_batches

                states_batch = states[start_idx:end_idx]
                expert_actions_batch = actions[start_idx:end_idx]
                # Train generator
                generator, fake_actions, loss_generator = self.train_generator(states_batch)
                total_g_loss_epoch += loss_generator.detach().numpy()

                # Train discriminator
                discriminator, loss_discriminator = self.train_discriminator(states_batch, fake_actions,
                                                                             expert_actions_batch)
                total_d_loss_epoch += loss_discriminator.detach().numpy()

            self.list_loss_generator = np.append(self.list_loss_generator, total_g_loss_epoch)
            self.list_loss_discriminator = np.append(self.list_loss_discriminator, total_d_loss_epoch)
            # Print loss
            print(f"Epoch {epoch + 1}: D_loss={loss_discriminator.item():.4f}, G_loss={loss_generator.item():.4f}")

            # Select states and actions using shuffled index array
            index = np.arange(num_samples)
            np.random.shuffle(index)
            states = states[index]
            actions = actions[index]

        torch.save(discriminator.state_dict(), "../model/discriminator.pth")
        torch.save(generator.state_dict(), "../model/generator.pth")

    def evaluate_gail(self, states, actions):
        discriminator = self.discriminator
        generator = self.generator
        discriminator.load_state_dict(torch.load("../model/discriminator.pth"))
        generator.load_state_dict(torch.load("../model/generator.pth"))

        generator_actions = generator(states)

        test_expert_scores = discriminator(states, actions)
        test_generator_scores = discriminator(states, generator_actions)

        test_expert_reward = torch.mean(test_expert_scores).item()
        test_generator_reward = torch.mean(test_generator_scores).item()

        # Compute the Wasserstein distance between distribution of real actions and distribution of generated actions
        expert_actions_np = actions.detach().cpu().numpy()
        fake_actions_np = generator_actions.detach().cpu().numpy()
        wasserstein_dist = wasserstein_distance(expert_actions_np.reshape(-1), fake_actions_np.reshape(-1))

        df = pd.DataFrame((states.tolist(), generator_actions.tolist(), actions.tolist()))
        df = df.transpose()
        df.to_csv("../output/vadere_generated.csv", header=['states', 'expert_actions', 'agent_actions'])

        print(f"Mean reward of the expert policy: {test_expert_reward:.4f}, "
              f"Mean reward of the agent policy: {test_generator_reward:.4f}, "
              f"Wasserstein distance between distribution of real actions and distribution of generated actions: "
              f"{wasserstein_dist:.4f}")

        # return test_expert_reward, test_generator_reward, wasserstein_dist

    def plot_loss(self):
        fig, ax = plt.subplots()

        # Plot the generator and discriminator loss values as two separate lines on the same plot
        x_axis = np.arange(1, config_file.num_epochs+1)
        print(self.list_loss_discriminator)
        ax.plot(x_axis, self.list_loss_generator, label='Generator loss')
        ax.plot(x_axis, self.list_loss_discriminator, label='Discriminator loss')
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss value')
        ax.set_title('GAIL training loss')
        plt.savefig('../output/GAIL_training_loss.png')
        # plt.show()
        # plt.close()


train_states, train_actions, test_states, test_actions, state_size, action_size = read_file()
gail = GAIL(train_states, train_actions, state_size, action_size)
gail.train_gail(train_states, train_actions)
gail.plot_loss()
gail.evaluate_gail(test_states, test_actions)

