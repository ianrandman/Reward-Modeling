from backend import web
from trajectory_builder import TrajectoryBuilder
import threading
import multiprocessing
from agents.training_system import TrainingSystem


def main():
    """
    Main entry point to the program. Facilitates creating objects and starting all threads for the program.
    The Flask backend runs on a separate thread from the TrainingSystems. There is a separate TrainingSystem
    thread for each environment being run and evaluated at the same time.
    """

    env_lst = ['LunarLanderContinuous-v2', 'CartPole-v1']
    trajectory_builder = TrajectoryBuilder()

    # start up the flask backend api
    app = web.get_webapp(trajectory_builder, env_lst)
    threading.Thread(target=app.run, args=['0.0.0.0', 5000]).start()

    # start up a thread for each environment
    for env in env_lst:
        training_system = TrainingSystem(env, record=True, use_reward_model=False, load_model=True)
        multiprocessing.Process(target=training_system.play).start()


if __name__ == "__main__":
    main()
