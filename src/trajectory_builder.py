from os import path, listdir
from os.path import isfile, join
import json
import random
from base64 import b64encode
from flask import jsonify


class TrajectoryBuilder:
    def __init__(self):
        self.counter = 0

    def get_binary(self, filepath):
        """
        Given a string filepath, return the binary of the file
        """
        with open(filepath, "rb") as f:
            return b64encode(f.read()).decode('utf-8')

    def get_variance(self, seg1, seg2):
        return random.randint(0, 100)

    def get_metadata(self):
        """
        Returns a map with some metadata about the model
        """
        return {}

    def get_pair(self, env, sample_size=20):
        """
        :return: the binary of two video files for the user to compare
        """
        dir_path = path.dirname(path.abspath(__file__)) + "/agents/recordings/"+env+"/"
        filenames = [f for f in listdir(dir_path) if f[-4:] == 'json' and isfile(join(dir_path, f))]
        filenames.sort(key=path.basename)  # sort list of names
        filenames = filenames[:-1]

        max_variance = -1
        mvpair = []

        for _ in range(sample_size):
            pair = random.sample(filenames, 2)
            # open and read the contents of both json files
            with open(dir_path+pair[0]) as f1:
                data1 = json.load(f1)
                with open(dir_path + pair[1]) as f2:
                    data2 = json.load(f2)
                    variance = self.get_variance(data1["pairs"], data2["pairs"])
                    if variance > max_variance:
                        max_variance = variance
                        mvpair = [(pair[0], data1), (pair[1], data2)]

        # get the mp4 video with the same title as the json file
        vids = [x[0][:-4]+"mp4" for x in mvpair]
        # convert to the binary of the mp4 files
        env_vids = [self.get_binary(dir_path+vids[0]), self.get_binary(dir_path+vids[1])]
        # return {"t1": mvpair[0], "t2": mvpair[1]}

        return {"seq1": {"sopairs": mvpair[0][1]["pairs"], "vid": env_vids[0]},
                "seq2": {"sopairs": mvpair[1][1]["pairs"], "vid": env_vids[1]},
                "metadata": self.get_metadata()}


if __name__ == '__main__':
    trajectorybuilder = TrajectoryBuilder()
    print(trajectorybuilder.get_pair())
