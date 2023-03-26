import numpy as np


class PartitionRectangle:

    def __init__(self, x1, x2, y1, y2, partition_id):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.partition_id = partition_id

        self.a = abs(x2 - x1)
        self.b = abs(y2 - y1)

        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x1, y2])
        self.p3 = np.array([x2, y1])
        self.p4 = np.array([x2, y2])
    #
    # def get_area(self):
    #     return abs(self.x2 - self.x1) * abs(self.y2 - self.y1)

    def get_area(self):
        return self.a * self.b

    def get_side_ratio(self):
        if self.b == 0:
            return 0
        else:
            return self.a / self.b
