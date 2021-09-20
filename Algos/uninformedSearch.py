import numpy as np
from utils import isEnd
from Structures.node import Node

def BreadthFirstSearch(data: np.array, restrictions: dict, status: dict, goaltest=isEnd):
    """
    Breadth-First Search based on algorithm in
    Russell, Norvig: "AI: A Modern Approach (3rd edition)" (p.82)
    returns shallowest solution
    :param data: numpy array of available nodes
    :param restrictions: dictionary of child nodes associated with each node
    :param status: dictionary tracking the possible failure conditions of the algorithm
    :param goaltest: function to check if goal state is achieved
    :return: solution path on success, status dictionary on failure
    """
    # Hard-coded endpoint for Project 2
    prj2End = data[-1]
    # Initialize space - hard-coded start at city 1
    node = Node(parent=None, state=data[0], children=restrictions[str(int(data[0,0]))])
    if goaltest(node.state, prj2End): return node.unwrap_parents()
    frontier = [node]
    explored = []
    while status['limit'] == -1 or frontier[0].depth < status['limit']:
        if frontier is None: return None
        # choose the shallowest node in the frontier
        # (shallowest node always first, children appended to end)
        node = frontier.pop(0)
        explored.append(node.state[0])
        print('Depth: {}'.format(node.depth), end='\r')
        for childName in node.get_children():
            chNode = Node(parent=node, state=data[childName-1], children=restrictions[str(childName)])
            if not ((chNode.state[0] in explored) or (chNode in frontier)):
                if goaltest(chNode.state, prj2End):
                    return chNode.unwrap_parents()
                frontier.append(chNode)

def UniformCostSearch(data: np.array, restrictions: dict, status: dict, goaltest=isEnd):
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
    # Hard-coded endpoint for Project 2
    prj2End = data[-1]
    # Initialize space - hard-coded start at city 1
    node = Node(parent=None, state=data[0], children=restrictions[str(int(data[0,0]))])
    if goaltest(node.state, prj2End): return node.unwrap_parents()
    frontier = [node]
    explored = []
    while status['limit'] == -1 or frontier[0].depth < status['limit']:
        if frontier is None: return None
        # select the node with the smallest path cost
        target_cost = min([x.path_cost for x in frontier])
        argmin = [i for i,x in enumerate(frontier) if x.path_cost == target_cost][0]
        node = frontier.pop(argmin)
        explored.append(node.state[0])
        print('Depth: {}'.format(node.depth), end='\r')
        for childName in node.get_children():
            chNode = Node(parent=node, state=data[childName-1], children=restrictions[str(childName)])
            if not ((chNode.state[0] in explored) or (chNode in frontier)):
                if goaltest(chNode.state, prj2End):
                    return chNode.unwrap_parents()
                frontier.append(chNode)


def DepthFirstSearch_Graph(data: np.array, restrictions: dict, status:dict, goaltest=isEnd):
    """
    Depth-First Search based on algorithm in
    Russell, Norvig: "AI: A Modern Approach (3rd edition)" (p.82)
    returns shallowest solution
    :param data: numpy array of available nodes
    :param restrictions: dictionary of child nodes associated with each node
    :param status: dictionary tracking the possible failure conditions of the algorithm
    :param goaltest: function to check if goal state is achieved
    :return: solution path on success, status dictionary on failure
    """
    # Hard-coded endpoint for Project 2
    prj2End = data[-1]
    # Initialize space - hard-coded start at city 1
    node = Node(parent=None, state=data[0], children=restrictions[str(int(data[0, 0]))])
    if goaltest(node.state, prj2End): return node.unwrap_parents()
    frontier = [node]
    explored = []
    while 1:
        if frontier is None: return None
        # choose the deepest node in the frontier
        # (deepest node always last, children appended to end)
        node = frontier.pop(-1)
        explored.append(node.state[0])
        print('Depth: {}'.format(node.depth), end='\r')
        if status['limit'] == -1 or node.depth < status['limit']:
            for childName in node.get_children():
                chNode = Node(parent=node, state=data[childName - 1], children=restrictions[str(childName)])
                if not ((chNode.state[0] in explored) or (chNode in frontier)):
                    if goaltest(chNode.state, prj2End): return chNode.unwrap_parents()
                    frontier.append(chNode)
    return soln

def DepthFirstSearch_Tree(node: Node, data: np.array, restrictions: dict, status: dict, goaltest=isEnd):
    """
    Depth-First Search based on tree algorithm in
    Russell, Norvig: "AI: A Modern Approach (3rd edition)" (p.88)
    :param data: numpy array of available nodes
    :param restrictions: dictionary of child nodes associated with each node
    :param status: dictionary tracking the possible failure conditions of the algorithm
    :param goaltest: function to check if goal state is achieved
    :return: solution path on success, status dictionary on failure
    """
    prj2End = data[-1]
    if goaltest(node.state, prj2End): return node.unwrap_parents()
    elif status['limit'] == 0:
        status['cutoff'] = True; return status
    else:
        status['cutoff'] = False
        for childName in node.get_children():
            status_ndepth = status.copy(); status_ndepth['limit'] = status_ndepth['limit'] - 1
            chNode = Node(parent=node, state=data[childName-1],children=restrictions[str(childName)])
            result = DepthFirstSearch_Tree(chNode, data, restrictions, status_ndepth, goaltest=isEnd)
            if isinstance(result,dict):
                if result['cutoff'] == True: status['cutoff'] = True
            else: return result
        if status['cutoff']: return status
        else: status['failure'] = True; return status


def IterativeDepthSearch(node: Node, data: np.array, restrictions: dict, status: dict, goaltest=isEnd):
    """
    Iterative Depth Search based on algorithm in
    Russell, Norvig. "AI: A Modern Approach (3rd edition)" (p.89)
    :param node: initial node / starting position
    :param data: numpy array of all nodes and positions
    :param restrictions: dictionary of child nodes associated with each node
    :param status: dictionary tracking the possible failure conditions of the algorithm
    :param goaltest: function to check if goal state is achieved
    :return: solution path on success, status dictionary on failure
    """
    depth = 0
    while status['limit'] == -1 or depth <= status['limit']:
        status_tmp = status.copy(); status_tmp['limit'] = depth
        soln = DepthFirstSearch_Tree(node, data, restrictions, status_tmp)
        if isinstance(soln, dict):
            if soln['cutoff']: depth += 1
            else: return soln
        else:
            return soln