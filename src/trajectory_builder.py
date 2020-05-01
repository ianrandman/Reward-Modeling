
class TrajectoryBuilder:
    def __init__(self):
        self.counter = 0

    def get_pair(self):
        self.counter += 1
        return '{"data": "T'+str(self.counter)+'"}'
