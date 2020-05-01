import gym
import argparse
from agents.continuous_environments import Environment
from agents.a2c.a2c import A2C
import numpy as np


class Agent(object):
    """The world's simplest agents!"""
    def __init__(self, action_space):
        self.action_space = action_space

    # You should modify this function
    def act(self, observation, reward, done):
        return self.action_space.sample()


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', nargs='?', default='Berzerk-v0', help='Select the environment to run')
    args = parser.parse_args()

    env_name = 'CartPole-v0'
    num_consecutive_frames = 5

    # env = gym.make(args.env_id)
    env = Environment(gym.make(env_name), num_consecutive_frames)
    env.reset()
    state_dim = env.get_state_size()
    action_dim = gym.make(env_name).action_space.n

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'random-agents-results'

    # env.seed(0)
    # agents = Agent(env.action_space)

    agent = A2C(action_dim, state_dim, num_consecutive_frames)

    episode_count = 100

    old_state, i, average = env.reset(), 0, 0
    score = 0
    scores = list()
    while True:
        # env.render()
        a = agent.policy_action(old_state)
        old_state, r, done, _ = env.step(a)
        score += r
        if done:
            scores.append(score)
            if i > 50:
                scores.pop(0)
            average = np.mean(scores)
            i += 1
            print(str(i + 1) + ', ' + str(score) + ', ' + str(int(average)))
            score = 0
            env.reset()

    env.env.close()

    # for i in range(episode_count):
    #     reward = 0
    #     done = False
    #     score = 0
    #     special_data = {}
    #     special_data['ale.lives'] = 3
    #     ob = env.reset()
    #     while not done:
    #         action = agent.act(ob, reward, done)
    #         ob, reward, done, x = env.step(action)
    #         score += reward
    #         env.render()
    #
    #     agent.act(ob, reward, done)
    #
    #     # agent_num_levels = agents.num_levels
    #     # agent_total_steps = agents.total_steps
    #     # agent_elapsed_time = agents.elapsed_time
    #     # Close the env and write monitor result info to disk
    #     # print ("Your score: %d" % score)
    #     print(str(i + 1) + ', ' + str(score))
    #     env.close()


if __name__ == '__main__':
    main()