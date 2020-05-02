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

    def get_pair(self, sample_size=20):
        """
        :return: the binary of two video files for the user to compare
        """
        dir_path = path.dirname(path.abspath(__file__)) + "/agents/temp/experiment_1/"
        filenames = [f for f in listdir(dir_path) if f[-4:] == 'json' and isfile(join(dir_path, f))]

        max_variance = -1
        most_variable_pair = []

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
                        most_variable_pair = pair

        print("most_variable_pair: "+ str(most_variable_pair))
        # get the mp4 video with the same title as the json file
        most_variable_pair = [x[:-4]+"mp4" for x in most_variable_pair]
        # convert to the binary of the mp4 files
        most_variable_pair = [self.get_binary(dir_path+most_variable_pair[0]), self.get_binary(dir_path+most_variable_pair[1])]
        print("returning: "+str({"t1": most_variable_pair[0], "t2": most_variable_pair[1]}))
        return {"t1": most_variable_pair[0], "t2": most_variable_pair[1]}


if __name__ == '__main__':
    trajectorybuilder = TrajectoryBuilder()
    print(trajectorybuilder.get_pair())
