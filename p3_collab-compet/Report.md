[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

[image2]: ./images/actor_network.png "Actor Network"

[image3]: ./images/critic_network.png "Critic Network"

[image4]: ./images/scores.png "Rewards"

# Deep Reinforcement Learning - Collaboration and Competition

Author: Hafizur Rahman

## Project Overview
For this project, agents need to be trained in [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## Learning Algorithm
Multi-Agent Deep Deterministic Policy Gradient method was used to solve the project.

Initially the observations were flattened and added to replay buffer and plain DDPG was tried. However, it did not seem to converge at all.

As mentioned in benchmark implementation, it was noted that each agent receives its own, local observation. Thus, so the code was modified to simultaneously train both agents through self-play. This MADDPG has 2 DDPG agents and experience is collected from both agents and added to a shared replay buffer.

### Noise
Noise was added to the actions to promotoe exploration.

## Model Architecture
Actor-Critic model with separate target network is used to improvde training stability.

### Actor Network
Actor (policy) network maps states -> actions.

* The model has 3 fully connected layers
* The first layer takes in the state passes it through 256 nodes with `relu` activation
* The second layer take the output from first layer and passes through 128 nodes with `relu` activation
* The third layer takes the output from the previous layer and outputs the action size with `tanh` activation
* Adam optimizer

```
Input nodes (8x3=24 states ) 
  -> Fully Connected Layer (256 units, Relu activation) 
      -> Fully Connected Layer (128 units, Relu activation) 
         -> Ouput nodes (2 units/actions, tanh activation)
```

### Critic Network
Critic (value) network maps (state, action) pairs -> Q-values.

The model has 3 fully connected layers
* The first layer takes the concatenated states and actions passes through 256 nodes with `relu` activation
* We then pass this to second layer which forwards through 128 nodes with `relu` activation
* The third layer takes the output from the previous layer and outputs 1
* Adam optimizer

```
Input nodes ( [ 8x3=24 states + 2 actions ] x 2 Agents = 52  ) 
  -> Fully Connected Layer (256 units, Relu activation)
      -> Fully Connected Layer (128 units, Relu activation) 
        -> Ouput node (1 unit, no activation)
```

### Hyperparameters

| Param  | Value | Description |
| ------------- | ------------- | ------------- |
| BUFFER_SIZE  | 100000 | Replay Buffer Size |
| BATCH_SIZE | 250 | Mini batch size |
| GAMMA | 0.99 | Discount factor |
| TAU | 1e-3| Soft update of target network param |
| LR_ACTOR | 1e-4 | Actor learning rate |
| LR_CRITIC | 1e-3 | Critic learning rate |
| WEIGHT_DECAY | 0 | L2 Weight Decay |

## Plot of Rewards
![Scores][image4]


## Ideas for Future Work

Stabilizing the model was really challenging. Further exploration on the following may improve training performance.

* Implement Prioritized Experience Replay
* Skipping learning for several steps and/or learn multiple times in a single episode
* Implement algorithms like PPO or D4PG
* Change number of layers or neuronsW
