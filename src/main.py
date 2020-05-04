from backend import web
from trajectory_builder import TrajectoryBuilder
import threading, multiprocessing
from agents.training_system import TrainingSystem


def main():
    env_lst = ['CartPole-v0', 'MountainCarContinuous-v0', 'Pendulum-v0', 'LunarLanderContinuous-v2']

    training_system = TrainingSystem(env_lst[2], record=True, use_reward_model=True)
    trajectory_builder = TrajectoryBuilder(training_system)

    # start up the flask backend api
    app = web.get_webapp(trajectory_builder, env_lst)
    threading.Thread(target=app.run, args=['0.0.0.0', 5000]).start()

    multiprocessing.Process(target=training_system.play()).start()


if __name__ == "__main__":
    main()
