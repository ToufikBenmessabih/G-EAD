import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='inhard',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'inhard':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), 
                             (4, 5), (5,6), (6, 7), (7, 8), 
                             (4, 9), (9, 10), (10, 11), (11, 12),
                             (4, 13), (13, 14), 
                             (0, 15), (15, 16), (16, 17),
                             (0, 18), (18, 19), (19, 20)]
            self.edge = self_link + neighbor_link
            self.center = 4 # Spine3
        elif layout == 'inhard_21':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (0, 4), (0, 7),
                             (1, 2), (2, 3), 
                             (4, 5), (5,6),
                             (7, 8), (8, 9), (9, 10), 
                             (10, 11),(10, 13),(10, 17), 
                             (11, 12),
                             (13, 14), (14, 15), (15, 16),
                             (17, 18), (18, 19), (19, 20)]
            self.edge = self_link + neighbor_link
            self.center = 10 # Spine3
        elif layout == 'inhard_72':
            self.num_node = 72
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), 
                             (4, 5), (5,6), (6, 7), (7, 8), 
                             (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
                             (8, 14), (14, 15), (15, 16), (16, 17), (17, 18),
                             (8, 19), (19, 20), (20, 21), (21, 22), (22, 23),
                             (8, 24), (24, 25), (25, 26), (26, 27), (27, 28),
                             (8, 29), (29, 30), (30, 31), (31, 32),
                             (4, 33), (33, 34), (34, 35), (35, 36),
                             (36, 37), (37, 38), (38, 39), (39, 40), (40, 41),
                             (36, 42), (42, 43), (43, 44), (44, 45), (45, 46),
                             (36, 47), (47, 48), (48, 49), (49, 50), (50, 51),
                             (36, 52), (52, 53), (53, 54), (54, 55), (55, 56),
                             (36, 57), (57, 58), (58, 59), (59, 60),
                             (4, 61), (61, 62), (62, 63),
                             (0, 64), (64, 65), (65, 66), (66, 67),
                             (0, 68), (68, 69), (69, 70), (70, 71)]
            self.edge = self_link + neighbor_link
            self.center = 4 # Spine3
        elif layout == 'ha4m': #-----------------------------------------------------------
            self.num_node = 32
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 26), 
                             (2, 4), (4, 5), (5,6), (6, 7), (7, 8), (7,10), (8, 9),
                             (2, 11), (11, 12), (12, 13), (13, 14), (14, 17), (14, 15), (15, 16),
                             (0, 18), (18, 19), (19, 20), (20, 21),
                             (0, 22), (22, 23), (23, 24), (24, 25)]
            self.edge = self_link + neighbor_link
            self.center = 0 # central hip
        elif layout == 'IKEA': #---------------------------------------------------------------------------------
            self.num_node = 33
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0,1), (0, 4),
                             (1,2), (2,3), (3,7),
                             (4, 5), (5, 6), (6, 8),
                             (9, 10),
                             (11, 12), (11, 13), (11, 23),
                             (13, 15), (15, 17), (15, 19), (15, 21),
                             (12, 14), (12, 24), 
                             (14, 16), (16, 18), (16, 20), (16, 22),
                             (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
                             (24, 26), (26, 28), (28, 30), (28, 32)
                             ]
            self.edge = self_link + neighbor_link
            self.center = 23 # left_hip
        elif layout == 'IKEA_up': #---------------------------------------------------------------------------------
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0,1), (0, 4),
                             (1,2), (2,3), (3,7),
                             (4, 5), (5, 6), (6, 8),
                             (9, 10),
                             (11, 12), (11, 13), (11, 23),
                             (13, 15), (15, 17), (15, 19), (15, 21),
                             (12, 14), (12, 24), 
                             (14, 16), (16, 18), (16, 20), (16, 22),
                             (23, 24)
                             ]
            self.edge = self_link + neighbor_link
            self.center = 23 # left_hip
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD