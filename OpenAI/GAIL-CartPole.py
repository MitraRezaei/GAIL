import gymnasium
import random
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

env_name = 'CartPole-v1'
env = gymnasium.make(env_name)


class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n

        def get_action(self, state):
            pole_angle = state[2]
            action = 0 if pole_angle < 0 else 1
            return action


agent = Agent(env)
n_episodes = 1000
expert_data = {'observations': [], 'actions': []}

for _ in range(n_episodes):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        if type(obs) == tuple:
            expert_data['observations'].append(obs[0])
        else:
            expert_data['observations'].append(obs)
        expert_data['actions'].append(action)
        obs, reward, done, _, _ = env.step(action)
expert_data['observations'] = np.array(expert_data['observations'])
expert_data['actions'] = np.array(expert_data['actions'])
np.savez('expert_data.npz', expert_data)
# --------------------------------------------------------------------

load_file = np.load('expert_data.npz')
observations = expert_data['observations']
actions = expert_data['actions']
expert_dataset = list(zip(observations, actions))


def create_generator(input_shape, output_shape):
    model = Sequential([
        Dense(32, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(output_shape, activation='softmax')
    ])
    return model


def create_discriminator(input_shape):
    model = Sequential([
        Dense(32, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


expert_dataset_size = len(expert_dataset)
num_epochs = 100
batch_size = 32
learning_rate_generator = 0.001
learning_rate_discriminator = 0.001
optimizer_actor = Adam(lr=learning_rate_generator)
optimizer_discriminator = Adam(lr=learning_rate_discriminator)


input_shape = env.observation_space.shape[0]
output_shape = env.action_space.n
generator = create_generator(input_shape, output_shape)
discriminator = create_discriminator(input_shape)
generator.compile(optimizer=optimizer_actor, loss='categorical_crossentropy')
discriminator.compile(optimizer=optimizer_discriminator, loss=BinaryCrossentropy(from_logits=True))


# Train GAIL
for epoch in range(num_epochs):
    # Sample batch of expert and GAIL trajectories
    expert_batch = [expert_dataset[i] for i in np.random.randint(expert_dataset_size, size=batch_size)]
    gail_batch = [(observation, generator.predict(np.array([observation]))[0]) for observation, _ in expert_batch]
    gail_batch = [(observation, action) for observation, action in gail_batch]

    # Train discriminator
    x = np.concatenate([np.array(expert_batch)[:, 0], np.array(gail_batch)[:, 0]])
    y = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    discriminator.train_on_batch(x, y)

    # Train actor
    x = np.array([observation for observation, _ in gail_batch])
    y = np.ones((batch_size, 1))
    generator.train_on_batch(x, y)

    # Evaluate performance
    if epoch % 10 == 0:
        success_rate = 0
        for _ in range(100):
            observation = env.reset()
            done = False
            while not done:
                action = generator.predict(np.array([observation]))[0]
                observation, _, done, info, _ = env.step(np.argmax(action))
                if done:
                    success_rate += 1
                    break
        success_rate /= 100.0
        print("Epoch: {}, Success rate: {}".format(epoch, success_rate))