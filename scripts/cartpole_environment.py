import gym
import numpy as np
import matplotlib.pyplot as plt

class MyEnvironment:
    def __init__(self, environment_name: str, action = None, observation = None):

        self.environment_name = environment_name
        self.environment = gym.make(self.environment_name)
        self.action = action
        self.observation = observation
        if 'CartPole' in self.environment_name:
            self.n_bins = 20
            self.bins = [np.linspace(-4.8, 4.8, self.n_bins), np.linspace(-4, 4, self.n_bins), 
                np.linspace(-0.418, 0.418, self.n_bins), np.linspace(-4, 4, self.n_bins)]

    # Returns the name of the environment
    def get_environment_name(self) -> str: 
        return self.environment.unwrapped.spec.id 

    def get_action_space(self):
        return self.environment.action_space

    def get_action_space_length(self):
        if 'CartPole' in self.environment_name:
            return 2
        return self.environment.action_space.n

    def get_observation_space(self):
        return self.environment.observation_space

    def get_observation_space_length(self):
        if 'CartPole' in self.environment_name:
            return [self.n_bins+1] * len(self.environment.observation_space.high)
        return [self.environment.horizon + 1]

    # Discretize the observation for CartPole.
    def set_observation(self, observation) -> None:
        self.observation = observation
        if 'CartPole' in self.environment_name:
            observation_index = []
            for i in range(len(self.get_observation_space().high)):
                observation_index.append(np.digitize(self.observation[i], self.bins[i]) - 1)  
            self.observation = tuple(observation_index)

    def display_environment(self):
        plt.imshow(self.environment.render(mode = 'rgb_array'))

    def reset(self):
        return self.environment.reset()

    def step(self):
        return self.environment.step(self.action)

    def render(self):
        return self.environment.render()

    def close(self):
        self.environment.close()
