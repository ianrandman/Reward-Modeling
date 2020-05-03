from backend import web
from trajectory_builder import TrajectoryBuilder
import threading


def main():
    env_lst = ['CartPole-v0', 'MountainCarContinuous-v0', 'Pendulum-v0', 'LunarLander-v2']

    # create the trajectory builder
    trajectory_builder = TrajectoryBuilder()

    # start up the flask backend api
    app = web.get_webapp(trajectory_builder, env_lst)
    threading.Thread(target=app.run, args=['0.0.0.0', 5000]).start()
    print('wassup')

    # start up a2c reward model system


if __name__ == "__main__":
    main()
