import gym
import argparse
import numpy as np
from a2c.a2c import A2C
from a2c_continuous.a2c import A2C_Continuous
# from gym import wrappers
from agents.monitor import Monitor


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

    record = False
    # env_name = 'CartPole-v0'
    # env_name = 'Pendulum-v0'
    env_name = 'MountainCarContinuous-v0'

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'random-agents-results'

    # env.seed(0)
    # agents = Agent(env.action_space)

    env = gym.make(env_name)
    # if record:
    #     env = Monitor(env, 'temp/experiment_1',max_segments=100, max_steps=20,
    #                   video_callable=lambda episode_id: episode_id% 10 == 0, force=True)
    scores, i, average, max_score, num_steps = [], 0, 0, float('-inf'), 0

    state_size = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    action_dim = env.action_space.shape[0]
    # agent = A2C(state_size=state_size, action_size=action_dim)
    agent = A2C_Continuous(state_size=state_size, action_size=action_dim)
    while True:
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if i > 200:
                env.render()

            num_steps += 1
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            # reward = reward if not done or score == 499 else -100
            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                # score = score if score == 500.0 else score + 100
                max_score = max(max_score, score)
                scores.append(score)
                if i > 50:
                    scores.pop(0)
                average = np.mean(scores)
                i += 1
                # every episode, plot the play time
                print('%s, %s, %s, %s, %s' % (i, num_steps, score, int(average), int(max_score)))
                num_steps = 0
                # print(str(i) + ', ' + str(score) + ', ' + str(int(average)) + ', ' + str(int(max_score)))

    env.close()


if __name__ == '__main__':
    main()
