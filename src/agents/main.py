import gym
import argparse
import numpy as np
from a2c.a2c import A2C
from a2c_continuous.a2c import A2C_Continuous
# from gym import wrappers
from src.agents.monitor import Monitor


class Ensemble(object):
    def __init__(self, ensemble_size=3):
        self.ensemble_size = ensemble_size
        self.models = ensemble_size * [A2C(state_size=1, action_size=1)]  # TODO change to RewardModel
        # TODO are these linked to the current "reward model" instead of freshly created?

    def get_variance(self, seg1, seg2):
        """Given 2 sequences, feed them into the ensemble and get the variance"""
        seg1_rewards = [sum([model.predict(oa_pair) for oa_pair in seg1]) for model in self.models]
        seg2_rewards = [sum([model.predict(oa_pair) for oa_pair in seg2]) for model in self.models]


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', nargs='?', default='Berzerk-v0', help='Select the environment to run')
    args = parser.parse_args()

    record = True
    # env_name = 'CartPole-v0'
    env_name = 'Pendulum-v0'
    # env_name = 'MountainCarContinuous-v0'
    # env_name = 'LunarLanderContinuous-v2'

    cont_for_env = {'CartPole-v0': False, 'MountainCarContinuous-v0': True, 'Pendulum-v0': True,
                    'LunarLander-v2': False, 'LunarLanderContinuous-v2': True}
    steps_for_env = {'CartPole-v0': 25, 'MountainCarContinuous-v0': 200, 'Pendulum-v0': 50,
                     'LunarLander-v2': 50, 'LunarLanderContinuous-v2': 50}
    continuous = cont_for_env[env_name]

    env = gym.make(env_name)
    if record:
        env = Monitor(env, 'recordings/'+env_name, max_segments=100, max_steps=steps_for_env[env_name],
                      video_callable=lambda episode_id: episode_id % 10 == 0, force=True)
    scores, i, average, max_score, num_steps = [], 0, 0, float('-inf'), 0
    state_size = env.observation_space.shape[0]

    if continuous:
        action_dim = env.action_space.shape[0]
        agent = A2C_Continuous(state_size=state_size, action_size=action_dim)
    else:
        action_dim = env.action_space.n
        agent = A2C(state_size=state_size, action_size=action_dim)

    while True:
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if i > 100 and i % 20 == 0:
                env.render()

            num_steps += 1
            action = agent.get_action(state)
            if continuous:
                action = action.reshape((action.shape[1],))
            if record:
                next_state, reward, done, info = env.step(state, action)
            else:
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
