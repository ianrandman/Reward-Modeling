from backend import web
from trajectory_builder import TrajectoryBuilder
import threading


def main():
    # create the trajectory builder
    trajectory_builder = TrajectoryBuilder()

    # start up the flask backend api
    app = web.get_webapp(trajectory_builder)
    threading.Thread(target=app.run(host='0.0.0.0', port=5000)).start()

    # start up A2C reward model system


if __name__ == "__main__":
    main()
