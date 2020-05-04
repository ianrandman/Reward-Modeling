from backend import web
from trajectory_builder import TrajectoryBuilder
import threading, multiprocessing
from agents.training_system import TrainingSystem
import time


def main():
    env_lst = ['CartPole-v0', 'MountainCarContinuous-v0', 'Pendulum-v0', 'LunarLanderContinuous-v2']
    trajectory_builder = TrajectoryBuilder()

    # start up the flask backend api
    last_feedback_time = web.LastFeedbackTime()
    app = web.get_webapp(trajectory_builder, env_lst, last_feedback_time)
    threading.Thread(target=app.run, args=['0.0.0.0', 5000]).start()

    training_system = TrainingSystem(env_lst[1], last_feedback_time, record=True, use_reward_model=True)
    multiprocessing.Process(target=training_system.play()).start()


if __name__ == "__main__":
    main()
