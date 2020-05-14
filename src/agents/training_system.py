import gym
import numpy as np
from src.agents.a2c.a2c import A2C
from src.agents.a2c_continuous.a2c import A2C_Continuous
from src.agents.reward_model.reward_model import Ensemble
from src.agents.monitor import Monitor
import json
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TrainingSystem:
    """
    In charge of running the entire training process. The play() function is the "main" that starts up this process.
    This class holds the reward model and agent model for a specific environment. It runs simulations in the env,
    using the agent to decide what moves to take, while the reward model gives it a corresponding reward. The agent
    uses this reward to train on each step, while the reward model trains periodically throughout the process.
    """

    def __init__(self, env_name, record=True, use_reward_model=False, load_model=False):
        self.env_name = env_name
        self.use_reward_model = use_reward_model
        self.record = record
        self.load_model = load_model
        self.reward_model_scaler = None

    def __init_ai(self):
        """
        Set up the rest of the initialization. This is called at the beginning of play() and is a separate method
        so that all AI components are created and used by the same thread (due to limitations with Tensorflow).
        """
        cont_for_env = {'CartPole-v1': False, 'MountainCarContinuous-v0': True, 'Pendulum-v0': True,
                        'LunarLander-v2': False, 'LunarLanderContinuous-v2': True}
        steps_for_env = {'CartPole-v1': 50, 'MountainCarContinuous-v0': 200, 'Pendulum-v0': 25, 'LunarLander-v2': 50,
                         'LunarLanderContinuous-v2': 50}
        self.continuous = cont_for_env[self.env_name]
        self.env = env = gym.make(self.env_name)

        self.observation_examples = np.array([[np.clip(x, -100, 100) for x in self.env.observation_space.sample()]
                                              for _ in range(10000)])
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(self.observation_examples)
        self.reward_model_scaler = StandardScaler()

        if self.record:
            path = os.path.dirname(os.path.abspath(__file__)) + '/recordings/' + self.env_name
            self.env = Monitor(env, path, max_segments=30, max_steps=steps_for_env[self.env_name],
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
        """
        Returns a scaled reward for an action given a state. The reward is scales with self.reward_model_scaler, which
        is updated every time the reward model is retrained.
        """
        return self.reward_model_scaler.transform(np.array(self.reward_model.get_reward(state, action)).reshape(-1, 1))[0][0]

    def train_reward_model(self):
        """
        Trains the reward model. New user feedback was probably given, so we need to update the reward model to better
        reflect the human reward function. We first pull the pref_db for this env and then use this as training data to
        update the model. Finally, we recreate our scalar for the rewards, preserving 0 mean and normal st dev for the
        rewards.
        """
        print('Training reward model for %s...' % self.env_name)
        pref_db = self.pull_pref_db()
        if pref_db is not None:
            self.reward_model.train_model(pref_db)
            self.reward_model.save_model()
            self.save_reward_model_graph()

            if self.continuous:
                sampled_actions = [np.array(self.agent.get_action(self.scaler.transform([state])[0]))
                                   for state in self.observation_examples]
            else:
                sampled_actions = [np.array([self.agent.get_action(
                    np.reshape(self.scaler.transform([state])[0], [1, self.state_size]))])
                   for state in self.observation_examples]

            sampled_rewards = [self.reward_model.get_reward([state], action) for state, action in
                               zip(self.observation_examples, sampled_actions)]
            scaler = StandardScaler()
            self.reward_model_scaler = scaler.fit(np.array(sampled_rewards).reshape(-1, 1))
            print('Finished training reward model for %s' % self.env_name)
        else:
            print('Preferences db empty for %s' % self.env_name)

    def save_agent_graph(self):
        """
        Creates a svg graph of the loss history for the agent. Useful for reporting results and seeing the performance
        of the model over time.
        """
        plt.tight_layout()
        actor_loss = self.agent.actor_loss_history[1:]
        critic_loss = self.agent.critic_loss_history[1:]
        if len(actor_loss) > 0 and len(critic_loss) > 0:
            plt.plot(actor_loss)
            plt.plot(critic_loss)
            plt.title('Model Loss\nFinal Actor Loss: ' + str(round(actor_loss[-1], 4)) +
                      ' | Final Critic Loss: ' + str(round(critic_loss[-1], 4)))
            plt.ylabel('loss')
            plt.xlabel('step')
            plt.legend(['actor', 'critic'], loc='upper left')
            plt.savefig('agents/save_model/'+self.env_name+'/agents_graphs.svg')
            plt.clf()

    def save_reward_model_graph(self):
        """
        Creates a svg graph of the loss history for the reward model. Useful for reporting results and seeing the
        performance of the model over time.
        """
        plt.tight_layout()
        reward_model_loss = self.reward_model.reward_model_training_history
        if len(reward_model_loss) > 0:
            plt.plot(reward_model_loss)
            plt.title('Model Loss\nFinal Reward Model Loss: ' + str(round(reward_model_loss[-1], 4)))
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['reward model'], loc='upper left')
            plt.savefig('agents/save_model/'+self.env_name+'/reward_model_graph.svg')
            plt.clf()

    def pull_pref_db(self):
        """
        Pulls the corresponding pref_db for this environment from the saved json file. If it fails (the file is in the
        middle of being written to), try again.
        """
        passed = False
        while not passed:
            with open('preferences/' + self.env_name + '/pref_db.json', 'r') as f:
                try:
                    pref_db = json.load(f)
                    passed = True
                    return pref_db if len(pref_db) > 0 else None
                except Exception as e:
                    pass

    def play(self):
        """
        This is the "main" that starts up the simulation. Runs in a loop forever, with each loop simulating an entire
        run of the environment. Inside this loop, there is another loop for each frame of the simulation where the
        agent takes an action based on its environment. The reward model gives a reward for this action based on the
        observation, which the agent then uses to train on and optimize this reward. Periodically, the reward model is
        trained on any new user feedback to better represent the humans' internal reward function.
        """
        self.__init_ai()

        if self.use_reward_model:
            self.train_reward_model()

        timesteps = 0
        while True:

            done = False
            score = 0
            state = self.env.reset()
            state = self.scaler.transform([state])[0]
            state = np.reshape(state, [1, self.state_size])

            if self.i != 0 and self.i % 50 == 0:
                self.agent.save_model()
                self.save_agent_graph()

            if self.use_reward_model:
                if self.i != 0 and self.i % 50 == 0:
                    self.train_reward_model()

            while not done:
                if not self.record:
                    self.env.render()

                self.num_steps += 1
                timesteps += 1

                action = self.agent.get_action(state)
                got_nan = False
                if self.continuous:
                    if np.isnan(action).any():
                        print("GOT NAN FOR ACTION")
                        action[np.isnan(action)] = 0
                    action = action.reshape((action.shape[1],))
                else:
                    action = action if not np.isnan(action) else 0
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
                        got_nan = True
                        # raise Exception("GOT NAN FOR REWARD")
                        # print("GOT NAN FOR REWARD")
                        reward = 0

                next_state = self.scaler.transform([next_state])[0]
                next_state = np.reshape(next_state, [1, self.state_size])
                if not got_nan:
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
                    print('%s %s, %s, %s, %s, %s %s' % (self.env_name, self.i, self.num_steps, round(score, 2),
                                                        round(self.average, 2), round(self.max_score, 1), timesteps))
                    self.num_steps = 0

        self.env.close()