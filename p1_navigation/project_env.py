#!/bin/python

from unityagents import UnityEnvironment

class ProjectEnv():
    def __init__(self, env_file_name):
        self.env = UnityEnvironment(file_name=env_file_name)
        self.brain_name = self.env.brain_names[0]

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        
        self.action_size = self.env.brains[self.brain_name].vector_action_space_size
        self.state_size = len(env_info.vector_observations[0])

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode)[self.brain_name]
        next_state = env_info.vector_observations[0]

        return next_state

    def step(self, action):        
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        return next_state, reward, done, env_info

    def close(self):
        self.env.close()