import numpy as np
from utils import euclidean_distance as euclid

class Node():
    """
    Nodes will be used to define the simplest relational environment (atomic, state-based)
    Each node represents an atomic state, describing the current state, prior state, and child states

    Attributes
    __________
    parent - the complete parent node, to unwravel the state-space for TSP
    state - the current state as a numpy array (city number, X, Y)
    children - the list of possible children (city numbers only)
    """

    def __init__(self, parent, state: dict, children: list):
        """
        initialize the node
        :param parent: parent node, None if initial
        :param state: numpy array [number, X, Y] of current node position
        :param children: list of possible children (city numbers only)
        """
        self.parent = parent
        self.children = children
        self.state = state
        if self.parent is None:
            self.path_cost = 0.
            self.depth = 0
        else:
            self.path_cost = parent.path_cost + \
                 euclid(np.array([self.parent.state, self.state]), loop=False)
            self.depth = 1 + parent.depth

    def unwrap_parents(self):
        """
        unwravel parent nodes to return a complete path
        :return: numpy array of path/action sequence, each row containing (number, X, Y)
        """
        path = self.state
        tmp = self.parent
        while not tmp is None:
            path = np.vstack((tmp.state,path))
            tmp = tmp.parent
        return path

    def get_children(self): return self.children
    def get_coords(self): return np.array([self.state['x'], self.state['y']])