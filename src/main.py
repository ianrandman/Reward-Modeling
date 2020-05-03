from backend import web
from trajectory_builder import TrajectoryBuilder
import threading


def main():
    env_lst = ['CartPole-v0', 'MountainCarContinuous-v0', 'Pendulum-v0', 'LunarLander-v2']

    # create the trajectory builder
    trajectory_builder = TrajectoryBuilder()

    # start up the flask backend api
    app = web.get_webapp(trajectory_builder, env_lst)
    threading.Thread(target=app.run(host='0.0.0.0', port=5000)).start()

    # start up a2c reward model system


if __name__ == "__main__":
    main()
