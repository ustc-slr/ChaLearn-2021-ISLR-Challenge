import sys
import copy
sys.path.insert(0, '')
sys.path.extend(['../'])
import numpy as np
from graph import tools


num_node = 65 # 11 + 21 + 21 + 12
self_link = [(i, i) for i in range(num_node)]


body_link = [(0, 0),
             (1, 0), (2, 0), (3, 1), (4, 2),
             (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8)]

hand_link = [(0, 0),
             (1, 0),  (2, 1),   (3, 2),   (4, 3),  # index left_hand +18
             (5, 0),  (6, 5),   (7, 6),   (8, 7),
             (9, 0),  (10, 9),  (11, 10), (12, 11),
             (13, 0), (14, 13), (15, 14), (16, 15),
             (17, 0), (18, 17), (19, 18), (20, 19)]
hand_link1 = [(i+11, j+11) for (i, j) in hand_link]
hand_link2 = copy.deepcopy(hand_link)
hand_link2 = [(i+32, j+32) for (i, j) in hand_link2]

face_link = [(0, 0),
             (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9), (11, 10)
             ]
face_link = [(i+53, j+53) for (i, j) in face_link]

inward = body_link + hand_link1 + hand_link2 + face_link

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
