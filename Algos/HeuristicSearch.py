import numpy as np
from utils import isEnd, DistFromLines, route_animate, FPS
from Structures.node import Node
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from functools import partial

def isEmpty(frontier: list):
    if len(frontier) == 0: return True
    else: return False


def GreedyNodeUCS(data: np.array, restrictions: dict, status: dict, goaltest=isEmpty, animate = False):
    """
    Variant of Breadth-First Search. Algorithm in
    Russell, Norvig: "AI: A Modern Approach (3rd edition)" (p.84)
    returns solution with sequence of cheapest actions
    :param data: numpy array of available nodes
    :param restrictions: dictionary of child nodes associated with each node
    :param status: dictionary tracking the possible failure conditions of the algorithm
    :param goaltest: function to check if goal state is achieved
    :return: solution path on success, status dictionary on failure
    """
    # Initialize space - hard-coded start at city 1
    rmn_cities = data[1:, 0].astype(int)
    node = Node(parent=None, state=data[0], children=rmn_cities)
    frontier = [node]
    frames = []

    while 1:
        if goaltest(frontier):
            if animate:
                fig, ax = plt.subplots()
                ax.scatter(data[:, 1], data[:, 2], color='r')
                [ax.annotate(str(int(data[i, 0])), (data[i, 1] + 0.3, data[i, 2] + 0.2)) for i in
                 range(len(data))]
                title = '{} (Dim: {})'.format(GreedyNodeUCS.__name__, len(data))
                ax.set_title(title)
                line = ax.plot(data[0, 1], data[0, 2])[0]
                animation = FuncAnimation(fig, func=route_animate, fargs=(line,True), frames=frames)
                Writer = writers['ffmpeg']
                writer = Writer(fps=FPS, metadata={'artist': 'Me'}, bitrate=1800)
                animation.save('Figures/{}/concorde{}.mp4'.format(GreedyNodeUCS.__name__, len(data)), writer)
                fig.clf();
                ax.cla();
                plt.close()
            return np.vstack((node.unwrap_parents(),node.unwrap_parents()[0]))

        # select the node with the smallest path cost
        target_cost = min([x.path_cost for x in frontier])
        argmin = [i for i,x in enumerate(frontier) if x.path_cost == target_cost][0]
        node = frontier.pop(argmin)

        # Generate a view of the current route for animations
        if animate: frames.append(node.unwrap_parents().reshape(-1,3))

        # Re-generate the frontier with only unused cities
        rmn_cities = rmn_cities[rmn_cities != node.state[0]]
        frontier = [Node(parent=node, state=data[x-1], children=rmn_cities) for x in node.get_children()]

def GreedyLineSearch_LASTSEG(data: np.array, restrictions: dict, status: dict, goaltest = isEmpty, animate = False):
    # Set search space restrictions - each node may only appear once
    possible_children = data[1:, 0].astype(int)

    # Initialize space - hard-coded start at city 1
    node = Node(parent=None, state=data[0], children=possible_children)
    possible_children = possible_children[possible_children != data[0]]
    frontier = [node]
    frames = [node.state.reshape(-1, 3)]
    route = node.state.reshape((1, 3))
    while 1:
        if goaltest(frontier):
            if animate:
                fig, ax = plt.subplots()
                ax.scatter(data[:, 1], data[:, 2], color='r')
                [ax.annotate(str(int(data[i, 0])), (data[i, 1] + 0.3, data[i, 2] + 0.2)) for i in
                 range(len(data))]
                title = '{} (Dim: {})'.format(GreedyLineSearch_LASTSEG.__name__, len(data))
                ax.set_title(title)
                line = ax.plot(data[0, 1], data[0, 2])[0]
                animation = FuncAnimation(fig, func=route_animate, fargs=(line,True), frames=frames)
                Writer = writers['ffmpeg']
                writer = Writer(fps=FPS, metadata={'artist': 'Me'}, bitrate=1800)
                animation.save('Figures/{}/concorde{}.mp4'.format(GreedyLineSearch_LASTSEG.__name__, len(data)),
                               writer)
                fig.clf();
                ax.cla();
                plt.close()
            return route

        # Pop from queue the node closest to the last segment
        nodedists = [DistFromLines(x.state, route[-2:])[0] for x in frontier]
        dist_next_pt = min(nodedists)
        idx_next_pt = [x for x in range(len(frontier)) if nodedists[x] == dist_next_pt][0]
        node = frontier.pop(idx_next_pt)

        # Insert node to restructure the last route segment
        if len(route) < 2: route = np.insert(route,1,node.state,0)
        else: route = np.insert(route,-1,node.state,0)
        if animate: frames.append(route.reshape(-1, 3))

        # assign the frontier as the remaining (unused) nodes
        possible_children = possible_children[possible_children != node.state[0]]
        frontier = [Node(parent=node, state=data[x-1], children=possible_children[possible_children != x])
                    for x in possible_children]


def GreedyLineSearch_MIN(data: np.array, restrictions: dict, status: dict, goaltest=isEmpty, animate = False):
    # Set search space restrictions - each node may only appear once
    possible_children = data[1:, 0].astype(int)
    node = Node(parent=None, state=data[0], children=possible_children)
    possible_children = possible_children[possible_children != data[0]]
    frontier = [node]
    frames = [node.state.reshape(-1,3)]
    route = node.state.reshape((1, 3))
    while 1:
        if goaltest(frontier):
            if animate:
                fig, ax = plt.subplots()
                ax.scatter(data[:, 1], data[:, 2], color='r')
                line = ax.plot(data[0, 1], data[0, 2])[0]
                [ax.annotate(str(int(data[i, 0])), (data[i, 1] + 0.3, data[i, 2] + 0.2)) for i in
                 range(len(data))]
                title = '{} (Dim: {})'.format(GreedyLineSearch_MIN.__name__, len(data))
                ax.set_title(title)
                animation = FuncAnimation(fig, func=route_animate, fargs=(line,True), frames=frames)
                Writer = writers['ffmpeg']
                writer = Writer(fps=FPS, metadata={'artist': 'Me'}, bitrate=1800)
                animation.save('Figures/{}/concorde{}.mp4'.format(GreedyLineSearch_MIN.__name__, len(data)), writer)
                fig.clf();
                ax.cla();
                plt.close()
            return route

        # Pop from queue the node closest to the current route
        nodedists = [DistFromLines(x.state, route) for x in frontier]
        idxdists = [x[1] for x in nodedists]; nodedists = [x[0] for x in nodedists]
        dist_next_pt = min(nodedists)
        idx_next_pt = [x for x in range(len(frontier)) if nodedists[x] == dist_next_pt][0]
        node = frontier.pop(idx_next_pt)

        # Insert the node appropriately
        # if only the start/end exists so far, the node MUST be placed between them (maintain closure)
        if len(route) == 2 and (route[0] == route[1]).all():
            route = np.insert(route, 1, node.state, 0)
        else: route = np.insert(route, idxdists[idx_next_pt], node.state, 0)
        if animate: frames.append(route)

        # assign the frontier as the remaining (unused) nodes
        possible_children = possible_children[possible_children != node.state[0]]
        frontier = [Node(parent=node, state=data[x - 1], children=possible_children[possible_children != x]) for
                    x in possible_children]