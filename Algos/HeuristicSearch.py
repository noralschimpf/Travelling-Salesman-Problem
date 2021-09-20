import numpy as np
from utils import isEnd, DistFromLines
from Structures.node import Node
def isEmpty(frontier: list):
    if len(frontier) == 0: return True
    else: return False

def GreedyBestFirstSearch(data: np.array, restrictions: dict, status: dict, goaltest = isEmpty):
    # Hard-coded endpoint for Project 3
    prj3End = data[0]
    possible_children = data[:,0].astype(int)

    # Initialize space - hard-coded start at city 1
    node = Node(parent=None, state=data[0], children=possible_children)
    possible_children = possible_children[possible_children != data[0]]

    frontier = [node]
    route = node.state
    while 1:
        if goaltest(frontier):
            return route[1:]

        # Pop from queue the node closest to the current route
        nodedists = [DistFromLines(x.state, node.unwrap_parents()) for x in frontier]
        dist_next_pt = min(nodedists); idx_next_pt = [x for x in range(len(frontier)) if nodedists[x] == dist_next_pt][0]
        node = frontier.pop(idx_next_pt)
        route = np.vstack((route, node.state))

        # assign the frontier as the remaining (unused) nodes
        possible_children = possible_children[possible_children != node.state[0]]
        frontier = [Node(parent=node, state=data[x-1], children=possible_children[possible_children != x]) for x in possible_children]