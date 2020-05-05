from backend import web
from trajectory_builder import TrajectoryBuilder
import threading, multiprocessing
from agents.training_system import TrainingSystem


def main():
    env_lst = ['CartPole-v1', 'MountainCarContinuous-v0', 'LunarLanderContinuous-v2']  # 'Pendulum-v0'
    trajectory_builder = TrajectoryBuilder()

    # start up the flask backend api
    app = web.get_webapp(trajectory_builder, env_lst)
    threading.Thread(target=app.run, args=['0.0.0.0', 5000]).start()

    for env in env_lst:
        training_system = TrainingSystem(env, record=True, use_reward_model=True, load_model=False)
        multiprocessing.Process(target=training_system.play).start()


if __name__ == "__main__":
    main()
