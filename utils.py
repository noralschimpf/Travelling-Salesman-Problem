import numpy as np, matplotlib.pyplot as plt, os
from matplotlib.animation import FuncAnimation, writers
import re
from numba import jit
from functools import partial
import warnings

FPS = 5

def sortByComplexity(files: list):
    """
    Sorts .tsp files by the number of cities indicated in the filename
    i.e. "Random4.tsp, Random5.tsp, ... Random11.tsp, Random12.tsp\
    :param files: list of .tsp filenames
    :return: sorted list of filenames
    """

    integer_subs = []
    for file in files:
        mo = re.match('[^0-9]*', file)
        integer_subs.append(int(file[mo.end():file.index('.')]))
    argsorted = np.argsort(integer_subs)
    f_sorted = [files[x] for x in argsorted]
    return f_sorted

def euclidean_distance(soln: np.array, loop: bool = True):
    """
    Computes the total Euclidean distance of a proposed route
    :param soln: numpy array of the proposed route, with columns [city number, X coordinate, Y coordinate]
    :return: Sum of Euclidean distances for each segment
    """
    dist = 0.
    length = soln.shape[0]
    if not loop: length = length - 1
    for i in range(length):
        path_dist = 0.
        if i == soln.shape[0] - 1 and loop: path_dist = np.sqrt((soln[0,1] - soln[i,1])**2 + (soln[0,2] - soln[i,2])**2)
        else: path_dist = np.sqrt((soln[i+1,1] - soln[i,1])**2 + (soln[i+1,2] - soln[i,2])**2)
        dist += path_dist
    return dist

@jit(nopython=True)
def euclid_partial(soln: np.array):
    """
    Computes squared Euclidean distance, to save time in vrute force approach
    :param soln: numpy array of the proposed route, with columns [city number, X coordinate, Y coordinate]
    :return: Sum of Euclidean distances for each segment
    """
    dist = 0.
    for i in range(soln.shape[0]):
        path_dist = 0.
        if i == soln.shape[0] - 1: path_dist = (soln[0,1] - soln[i,1])**2 + (soln[0,2] - soln[i,2])**2
        else: path_dist = (soln[i+1,1] - soln[i,1])**2 + (soln[i+1,2] - soln[i,2])**2
        dist += path_dist
    return dist

def shortest_dist(point: np.ndarray, segment: np.ndarray):
    """
    Calculates the shortest distance between a point an given line segment
    Calculation based on
    https://math.stackexchange.com/questions/2248617/shortest-distance-between-a-point-and-a-line-segment
    :param point: shape (1,3) containing (city, x, y)
    :param segment: shape (2,3) containing (city, x, y)
    :return: the length of the line perpendicular to the segment (if it intersects the segment)
              OR the distance to the nearest endpoint of the segment
              also return a bias, shifting the placement of the node
    """
    # Calculating the ratio of the dot product between the point and segment v. the segment length-squared identifies
    #   the perpendicular (theoretically shortest path) intersects
    with warnings.catch_warnings(record=True) as w:
        seglen = (segment[1,1]-segment[0,1])**2 + (segment[1,2]-segment[0,2])**2
        if seglen == 0: return (euclidean_distance(np.vstack((point,segment[0])),loop=False), 0)
        t = -1*(
                (
                 ((segment[0,1]-point[1]) * (segment[1,1]-segment[0,1])) +
                 ((segment[0,2]-point[2])*(segment[1,2]-segment[0,2]))
                ) / seglen)
        if not w is None:
            if len(w) == 1 and issubclass(w[0].category, RuntimeWarning):
                print("invalid value {} calculating for pt {} to segment {}-{}".format(t, point[0], segment[0,0], segment[1,0]))

    if t >= 0 and t <= 1:
        # if the theoretically shortest holds, it is calculated by the cross-product (s2-s1)x(s1-p)
        #   divided by the magnitude of the segment
        d = np.abs(
                   (
                    ((segment[1, 1] - segment[0, 1])*(segment[0, 2] - point[2])) -
                    ((segment[1, 2] - segment[0,2])*(segment[0, 1] - point[1]))
                   )
                  ) / euclidean_distance(segment, loop=False)
        return (d, 1)
    else:
        # if the theoretically shortest length does not hold, return the nearest endpoint of the segment
        p_dists = [euclidean_distance(np.vstack((point,x)),loop=False) for x in segment]
        mindist = min(p_dists); bias = 0 if mindist == p_dists[0] else 2
        return (min(p_dists), bias)

def DistFromLines(point: np.array, route: np.array):
    """
    Calculates distance of a point from each line segment in the current route
    d = |(x2-x1)(y1-y0) - (x1-x0)(y2-y1)| / sqrt((x2-x1)^2 + (y2-y1)^2)
    x0, y0: coordinates of point
    x/y1, x/y2: coordinates of line segment start and end
    eqn. from: https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    :param point: state of point (city number, x, y)
    :param route: current route as matrix of rows (city number, x, y)
    :return: distance of last segment (initial draft)
    """

    dists = [] # Distances from each line segment
    biases = [] # Distances from each point on the route
    if route.shape[0] < 2:
        dists.append(euclidean_distance(np.vstack((point,route)),loop=False))
        return (dists[0], 0)
    else:
        for i in range(route.shape[0]):
            if i < len(route)-1:
                dists.append(shortest_dist(point, route[i:i+2]))

    biases = [x[1] for x in dists]; dists = [x[0] for x in dists]
    mindist = min(dists)
    idxmin_seg = [i for i in range(len(dists)) if dists[i] == mindist][0]
    idxmin_pt = idxmin_seg + biases[idxmin_seg]
    # if ptdists[idxmin_pt] > ptdists[idxmin_pt+1]: idxmin_pt += 1
    # if idxmin_pt == len(route)+1: idxmin_pt -= 1
    if idxmin_pt == 0: idxmin_pt += 1
    return (mindist, idxmin_pt)
    # return (dists[-1], -1)


def isEnd(state, endstate):
    """
    Goal-State Check
    :param state: state of current node
    :return: boolean of whether the goal is met
    """
    if state[0] == endstate[0]: return True
    else: return False


def route_animate(soln_frame, route, loop):
    if loop: soln_frame = np.vstack((soln_frame, soln_frame[0]))
    route.set_xdata(soln_frame[:,1])
    route.set_ydata(soln_frame[:,2])
    # route.set_data(x,y)



def report(data: dict, soln: dict, metrics: dict,loop=True, scalems=False, animate=False):
    """
    Generates report figure of proposed route
    :param data: dictionary containing the cities, problem name, and any miscellaneous information
    :param soln: dictionary containing the route, algorithm used, and miscellaneous information
    :param metrics: dictionary containing the runtime and memory usage of the solution
    :return:
    """
    scale = 1000 if scalems else 1
    cost = euclidean_distance(soln['soln'],loop=loop)
    fig, ax = plt.subplots(1,1)
    nda_dat = data['data']; nda_soln = soln['soln']
    if loop: nda_soln = np.concatenate((nda_soln, nda_soln[0].reshape(1,-1)))
    ax.scatter(nda_dat[:,1], nda_dat[:,2], color='r')
    [ax.annotate(str(int(nda_dat[i,0])), (nda_dat[i,1]+0.3, nda_dat[i,2]+0.2)) for i in range(len(nda_dat))]
    ax.plot(nda_soln[:,1], nda_soln[:,2], color='b')
    fig.suptitle('TSP {} (Dim: {})\ncost: {:.3f}   time:{:.3f} {}s   mem: {:.3f} KB'.format(soln['name'], data['DIMENSION'],cost, metrics['time']*scale, 'm' if scalems else '', metrics['memory']))
    if not os.path.isdir('Figures/{}'.format(soln['name'])): os.makedirs('Figures/{}'.format(soln['name']))
    fig.savefig('Figures/{}/{}.png'.format(soln['name'], data['NAME']), dpi=300)
    fig.clf(); ax.cla(); plt.close()
