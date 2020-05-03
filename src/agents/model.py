import gym
import argparse
import numpy as np
from src.agents.a2c.a2c import A2C
from src.agents.a2c_continuous.a2c import A2C_Continuous
from src.agents.reward_model.reward_model import RewardModel
from src.agents.monitor import Monitor
from backend.web import last_feedback_time
import json
import sched, time


class TrainingSystem:
    def __init__(self, env_name, record=False, use_reward_model=False):
        self.env_name = env_name
        self.reward_model = RewardModel
        self.use_reward_model = use_reward_model
        self.record = record

    def init_ai(self):
        cont_for_env = {'CartPole-v0': False, 'MountainCarContinuous-v0': True, 'Pendulum-v0': True,
                        'LunarLander-v2': False}
        steps_for_env = {'CartPole-v0': 25, 'MountainCarContinuous-v0': 200, 'Pendulum-v0': 25, 'LunarLander-v2': 50}
        self.continuous = cont_for_env[self.env_name]
        self.env = env = gym.make(self.env_name)

        if self.record:
            self.env = Monitor(env, 'recordings/' + self.env_name, max_segments=100, max_steps=steps_for_env[
                self.env_name],
                               video_callable=lambda episode_id: episode_id % 10 == 0, force=True)
        self.scores, self.i, self.average, self.max_score, self.num_steps = [], 0, 0, float('-inf'), 0
        self.state_size = env.observation_space.shape[0]

        if self.continuous:
            action_dim = env.action_space.shape[0]
            self.agent = A2C_Continuous(state_size=self.state_size, action_size=action_dim)
        else:
            action_dim = env.action_space.n
            self.agent = A2C(state_size=self.state_size, action_size=action_dim)

    def predict_reward(self, state, action):
        return self.reward_model.get_reward(state, action)

    def train_reward_model(self, pref_db):
        self.reward_model.train_model(pref_db)

    def training_loop(self, s, sc):
        # last_feedback_time
        print("training loop")
        s.enter(2, 1, self.training_loop, (sc,))

    def pretrain_model(self):
        # pref_db = self.pull_pref_db()
        # # pretrain
        # if pref_db is not None:
        #     self.train_reward_model(pref_db)
        print("begin pretrain")

        # pretrain
        # if no one is giving feedback, dont overfit
        # if we are getting feedback, train every once in a while

    def pull_pref_db(self):
        with open("preferences/" + self.env_name + "/pref_db.json", 'r') as f:
            pref_db = json.load(f)
            return pref_db if len(pref_db) > 0 else None

    def play(self):
        self.init_ai()
        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('--env_id', nargs='?', default='Berzerk-v0', help='Select the environment to run')
        args = parser.parse_args()

        while True:
            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            while not done:
                if self.i > 100 and self.i % 20 == 0:
                    self.env.render()

                self.num_steps += 1
                action = self.agent.get_action(state)
                if self.continuous:
                    action = action.reshape((action.shape[1],))
                if self.record:
                    next_state, reward, done, info = self.env.step(state, action)
                else:
                    next_state, reward, done, info = self.env.step(action)

                if self.use_reward_model:
                    reward = self.predict_reward(state, action)

                next_state = np.reshape(next_state, [1, self.state_size])
                self.agent.train_model(state, action, reward, next_state, done)

                score += reward
                state = next_state

                if done:
                    self.max_score = max(self.max_score, score)
                    self.scores.append(score)
                    if self.i > 50:
                        self.scores.pop(0)
                    self.average = np.mean(self.scores)
                    self.i += 1
                    # every episode, plot the play time
                    print('%s, %s, %s, %s, %s' % (self.i, self.num_steps, score, int(self.average), int(self.max_score)))
                    self.num_steps = 0

        self.env.close()