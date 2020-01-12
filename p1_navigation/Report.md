[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: ./DDQN.jpg "Dueling DQN"

# Project 1: Navigation
### Author: Hafizur Rahman

For this project, an agent was trained to navigate (and collect bananas!) in a large, square world.

![Trained Agent][image1]

## Learning Algorithm
The project implemented various [Deep Q-Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) algorithm, a value-based Reinforcement Learning algorithm which surpassed human-level performance in Atari games.

* [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Double Deep Q-Network](https://arxiv.org/abs/1509.06461)
* [Dueling Q-Network](https://arxiv.org/abs/1511.06581)
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

![Dueling DQN][image2]

### Deep Q-Network
Initially I tried to solve using a vanilla DQN with Experience Replay. However, it was not converging at all. Once Fixed Q-Targets were introduced, it solved in less than 400 episodes.

### Double Deep Q-Network
Initially Double Deep Q-Network was not showing any improvement. However, hyperparameter tuning for `learning rate` and `eps` helped to improve the performance.

### Dueling Network
Though I tried Dueling Network, it didn't show signification improvement. The hidden layers might need tuning to get the desired behavior.

### Prioritized Experience Replay
Even though it's supposed to improve the performance, the initial tests showed that both DQN and DDQN results worsen when using using prioritized replay. Further efforts on parameter tuning might improve the outcome.

### Hyperparameters

| Hyperparameter                      | Value     |
|-------------------------------------|:---------:|
| Replay buffer size	              | 10000     |
| Batch size	                      | 64        |
| gamma (discount factor)	          | 0.99      |
| tau                                 | 1e-2      |
| Learning rate	                      | 4.8e-4    |
| update interval	                  | 4         |
| Number of episodes	              | 500       |
| Max number of timesteps per episode |	300       |
| Epsilon start	                      | 0.1       |
| Epsilon minimum	                  | 0.01      |
| Epsilon decay	                      | 0.985     |

## Plot of Rewards
| Network         | # Episode to solve |
|-----------------|:------------------:|
| dqn-wo-dueling  |  85                |
| dqn-w-dueling   |  78                |
| ddqn-wo-dueling |  55                |
| ddqn-w-dueling  |  77                |

The best performance was achieved by `DDQN` without `Dueling` network, scored 13+ in 55 episodes.

Hyperparameter tuning for `learning rate` and `eps` helped to improve the performance.

![alt text](scores.png)

## Ideas for Future Work
* [Prioritized experience replay](https://arxiv.org/abs/1511.05952) in Deep Q-Networks (DQN) achieved human-level performance across many Atari games, so it was tried to explore. However, it didn't give expected performance. Probably it needs to make more effort on parameter tuning.
* [Rainbow](https://arxiv.org/abs/1710.02298) approach is also worthy of exploration to combine improvements in Deep Reinforcement Learning
* Other improvements like [multi-step bootstrap targets](https://arxiv.org/abs/1602.01783), [Distributional DQN](https://arxiv.org/abs/1707.06887), [Noisy DQN](https://arxiv.org/abs/1706.10295) might be benefical as well
