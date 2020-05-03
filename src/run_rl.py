from backend import web
from trajectory_builder import TrajectoryBuilder
import threading, multiprocessing
from agents.model import TrainingSystem


def main():
    env_lst = ['CartPole-v0', 'MountainCarContinuous-v0', 'Pendulum-v0', 'LunarLander-v2']
    training_system = TrainingSystem(env_lst[1], record=False)
    training_system.play()


if __name__ == "__main__":
    main()