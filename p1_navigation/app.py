#!/bin/python
import argparse
import matplotlib.pyplot as plt

from project_env import ProjectEnv
from dqn_agent import *
from utils import mini_batch_train

update_types = {
    'dqn-wo-dueling': {'use_double_dqn': False, 'use_dueling': False},
    'dqn-w-dueling': {'use_double_dqn': False, 'use_dueling': True},
    'ddqn-wo-dueling': {'use_double_dqn': True, 'use_dueling': False},
    'ddqn-w-dueling': {'use_double_dqn': True, 'use_dueling': True}
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unity-env", type=str, required=True,
                        help="Location of Unity env")
    parser.add_argument("--agent-type", type=str, required=True,
                        choices=['dqn-wo-dueling', 'dqn-w-dueling',
                                 'ddqn-wo-dueling', 'ddqn-w-dueling'],
                        help="Type of Agent")
    parser.add_argument("--action", type=str, default="train",
                        choices=["train", "run"],
                        help="Train an agent or Run a trained agent?")

    args = parser.parse_args()

    env = ProjectEnv(args.unity_env)

    if args.action == "train":
        train(env, args.agent_type)
    elif args.action == "run":
        run(env, args.agent_type)


def train(env, agent_type):
    n_episodes = 500
    max_t = 300
    eps_start = 0.1
    eps_end = 0.01
    eps_decay = 0.985

    conf = update_types[agent_type]

    agent = DQNAgent(agent_type, env.state_size, env.action_size, conf['use_double_dqn'],
                     conf['use_dueling'], use_prioritized_replay=False)

    print('\nRunning DQNAgent ({})'.format(agent_type))
    scores, scores_mean = mini_batch_train(env, agent, max_episodes=n_episodes, max_steps=max_t, batch_size=64,
                                           eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)

    plot_rewards(agent_type, scores, scores_mean)


def plot_rewards(agent_type, scores, scores_mean, save_fig=True):
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores_mean)), scores_mean)
    plt.title(agent_type)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(('Score', 'Mean'), fontsize='large')

    if save_fig:
        plt.savefig('scores-{}.png'.format(agent_type), bbox_inches='tight')

    plt.show()


def run(env, agent_type, run_count=5):
    conf = update_types[agent_type]

    agent = DQNAgent(agent_type, env.state_size, env.action_size, conf['use_double_dqn'],
                     conf['use_dueling'], use_prioritized_replay=False)
    agent.qnetwork_local.load_state_dict(
        torch.load('{}.pth'.format(agent_type)))

    for i in range(0, 5):
        state = env.reset()
        score = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, env_info = env.step(
                action)
            score += reward
            state = next_state

            if done:
                break

        print("Score using {} model (run {}): {}".format(agent_type, i+1, score))


if __name__ == "__main__":
    main()
