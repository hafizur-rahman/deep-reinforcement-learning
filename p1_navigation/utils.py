#!/bin/python

from collections import deque
import numpy as np
import torch

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size, eps_start=1.0, eps_end=0.1, eps_decay=0.995):
    episode_rewards = []
    rewards_window = deque(maxlen=100)
    eps = eps_start

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)

            agent.step(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        rewards_window.append(episode_reward)

        eps = max(eps_end, eps*eps_decay)
        average_score = np.mean(rewards_window)

        print('\rEpisode {} \tAverage score: {: .2f}'.format(episode, average_score), end="")
        
        if episode % 100 == 0:
            print('\rEpisode {} \tAverage score: {: .2f}'.format(episode, average_score))
            
        if average_score >= 13:      # check if environment is solved
            print('\nEnvironment solved in {: d} episodes!\tAverage Score: {: .2f}'.format(episode - 100, average_score))
                      
            torch.save(agent.qnetwork_local.state_dict(), '{}.pth'.format(agent.update_type))
            break          

    return episode_rewards
