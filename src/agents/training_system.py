import gym
import numpy as np
from src.agents.a2c.a2c import A2C
from src.agents.a2c_continuous.a2c import A2C_Continuous
from src.agents.reward_model.reward_model import Ensemble
from src.agents.monitor import Monitor
import json
import time
import os
import matplotlib.pyplot as plt


class TrainingSystem:
    def __init__(self, env_name, last_feedback_time, record=True, use_reward_model=False, load_model=False):
        self.env_name = env_name
        self.use_reward_model = use_reward_model
        self.record = record
        self. last_pull_time = None
        self.load_model = load_model
        self.last_feedback_time = last_feedback_time
        self.critic_loss = []
        self.actor_loss = []
        self.reward_model_loss = []

    def __init_ai(self):
        cont_for_env = {'CartPole-v0': False, 'MountainCarContinuous-v0': True, 'Pendulum-v0': True,
                        'LunarLander-v2': False, 'LunarLanderContinuous-v2': True}
        steps_for_env = {'CartPole-v0': 25, 'MountainCarContinuous-v0': 200, 'Pendulum-v0': 25, 'LunarLander-v2': 50,
                         'LunarLanderContinuous-v2': 50}
        self.continuous = cont_for_env[self.env_name]
        self.env = env = gym.make(self.env_name)
        if self.record:
            path = os.path.dirname(os.path.abspath(__file__)) + '/recordings/' + self.env_name
            self.env = Monitor(env, path, max_segments=100, max_steps=steps_for_env[self.env_name],
                               video_callable=lambda episode_id: episode_id % 10 == 0, force=True)
        self.scores, self.i, self.average, self.max_score, self.num_steps = [], 0, 0, float('-inf'), 0
        self.state_size = env.observation_space.shape[0]

        if self.continuous:
            action_dim = env.action_space.shape[0]
            action_high = env.action_space.high
            self.agent = A2C_Continuous(env=self.env_name, state_size=self.state_size, action_size=action_dim,
                                        action_high=action_high, load_model=self.load_model)
        else:
            action_dim = env.action_space.n
            self.agent = A2C(env=self.env_name, state_size=self.state_size, action_size=action_dim,
                             load_model=self.load_model)

        if self.continuous:
            self.ensemble = Ensemble(self.state_size, action_dim, steps_for_env[self.env_name], self.env_name,
                                     load_model=self.load_model)
        else:
            self.ensemble = Ensemble(self.state_size, 1, steps_for_env[self.env_name], self.env_name,
                                     load_model=self.load_model)
        self.reward_model = self.ensemble.model

    def predict_reward(self, state, action):
        return self.reward_model.get_reward(state, action)

    def train_reward_model(self):
        print("Training reward model...")
        pref_db = self.pull_pref_db()
        if pref_db is not None:
            history = self.reward_model.train_model(pref_db)
            self.reward_model_loss.extend(history.history['loss'])
            self.reward_model.save_model(self.env_name)
            self.save_reward_model_graph()
            print("Finished training reward model")
        else:
            print("Preferences db empty")

    def save_agent_graph(self):
        plt.tight_layout()
        plt.plot(self.actor_loss)
        plt.plot(self.critic_loss)
        plt.title('Model Loss\nFinal Actor Loss: ' + str(round(self.actor_loss[-1], 4)) +
                  ' | Final Critic Loss: ' + str(round(self.critic_loss[-1], 4)))
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.legend(['actor', 'critic'], loc='upper left')
        plt.savefig("agents/save_model/"+self.env_name+"/agents_graphs.png")
        plt.clf()

    def save_reward_model_graph(self):
        plt.tight_layout()
        plt.plot(self.reward_model_loss)
        plt.title('Model Loss\nFinal Reward Model Loss: ' + str(round(self.reward_model_loss[-1], 4)))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['reward model'], loc='upper left')
        plt.savefig("agents/save_model/"+self.env_name+"/reward_model_graph.png")
        plt.clf()

    def pull_pref_db(self):
        self.last_pull_time = time.time()
        with open("preferences/" + self.env_name + "/pref_db.json", 'r') as f:
            pref_db = json.load(f)
            return pref_db if len(pref_db) > 0 else None

    def play(self):
        self.__init_ai()

        if self.use_reward_model:
            self.train_reward_model()

        timesteps = 0
        while True:

            # check if db has refreshed to retrain reward model
            if self.use_reward_model and self.last_feedback_time.get_last_feeback_time(self.env_name) \
                    > self.last_pull_time + 90:
                print("got some new feedback, retraining the model...")
                self.train_reward_model()

            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            if self.i != 0 and self.i % 50 == 0:
                self.agent.save_model(self.env_name)
                self.save_agent_graph()

            while not done:
                if not self.record:
                    self.env.render()

                self.num_steps += 1
                timesteps += 1

                action = self.agent.get_action(state)
                if self.continuous:
                    action = action.reshape((action.shape[1],))
                if self.record:
                    next_state, reward, done, info = self.env.step(state, action)
                else:
                    next_state, reward, done, info = self.env.step(action)

                if self.use_reward_model:
                    if self.continuous:
                        reward = self.predict_reward(state, np.array([action]))
                    else:
                        reward = self.predict_reward(state, np.array([[action]]))

                    if np.isnan(reward):
                        reward = 0

                next_state = np.reshape(next_state, [1, self.state_size])
                result = self.agent.train_model(state, action, reward, next_state, done)
                if result[0] == 'Trained':
                    self.critic_loss.extend(result[1].history['loss'])
                    self.actor_loss.extend(result[2].history['loss'])

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
                    print('%s %s, %s, %s, %s, %s %s' % (self.env_name, self.i, self.num_steps, round(score, 2),
                                                        round(self.average, 2), round(self.max_score, 1), timesteps))
                    self.num_steps = 0

        self.env.close()