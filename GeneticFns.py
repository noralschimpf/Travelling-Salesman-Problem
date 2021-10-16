import numpy as np
from utils import euclidean_distance, MAXTHREAD
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor

def fit_euclid(nda_data: np.array):
    """
    Fitness function using euclidean norm
    :param nda_data: solution route, not looped
    :return: total euclidean distance
    """
    return euclidean_distance(nda_data, loop=True)

@jit(nopython=True, cache=True, parallel=True)
def fit_intersect(nda_data: np.array):
    """
    Fitness function using number of times route segments overlap
    :param nda_data: solution route, not looped
    :return: total number of intersections
    """

    # initialize
    ints = 0
    nda_data = np.vstack((nda_data, nda_data[0].reshape(-1,3)))

    # for each segment
    #     ints += number of ints with previous segs
    for i in prange(len(nda_data) - 1):
        # with ProcessPoolExecutor(max_workers=MAXTHREAD) as ex:
        for j in range(i):
            ints += is_intersect(nda_data[i:i+2], nda_data[j:j+2])

    return ints

@jit(nopython=True, cache=True)
def is_intersect(seg1: np.array, seg2: np.array):
    """
    returns 1 if segments intersect, 0 else
    See below for algorithm source
    https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    :param seg1:
    :param seg2:
    :return:
    """
    # if seg1_x's between seg2_x's and seg2
    if ccw(seg1[0],seg2[0],seg2[1]) != ccw(seg1[1],seg2[0],seg2[1]) and ccw(seg1[0],seg1[1],seg2[0]) != ccw(seg1[0],seg1[1],seg2[1]):
        return 1
    else: return 0

@jit(nopython=True, cache=True)
def ccw(A,B,C):
    """
    See is_intersect stackoverflow link
    """
    return (C[2]-A[2]) * (B[2]-A[1]) > (B[2]-A[2]) * (C[1]-A[1])

# @jit(nopython=True)
# def cross_ordered(parent1, parent2):
#     """
#     Copied from TDS article. FOR DEBUG ONLY
#     https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
#     :param parent1:
#     :param parent2:
#     :return:
#     """
#     child = []
#     childP1 = []
#     childP2 = []
#
#     geneA = int(np.random.random() * len(parent1))
#     geneB = int(np.random.random() * len(parent1))
#
#     startGene = min(geneA, geneB)
#     endGene = max(geneA, geneB)
#
#     for i in range(startGene, endGene):
#         childP1.append(parent1[i])
#
#     childP2 = [item for item in parent2 if item not in childP1]
#
#     child = childP1 + childP2
#     return child

@jit(nopython=True, cache=True)
def cross_p_point(pair1, pair2, p=1):
    """
    Select p indexes at random to join pair 1 and pair2
    To insure all cities remain in the route, crossover inserts remaining cities by the ordering of pair2, not the
    direct values in pair2
    :param pair1: city list order
    :param pair2: city list order
    :param p: number of cross points
    :return: single list
    """
    crosspoints = np.random.choice(np.arange(len(pair1)), size=p, replace=False)
    crosspoints.sort()
    crosspoints = np.array([int(len(pair1)/2)])
    combo = pair1
    for i in range(len(crosspoints) - 1):
        if i % 2 == 0: combo[crosspoints[i]:crosspoints[i+1]] = -1
    if len(crosspoints) == 1:
        combo[crosspoints[0]:] = -1
    missing_vals = set(pair2) - set(pair1)
    idx_misvals = [i for i in range(len(pair2)) if pair2[i] in missing_vals]
    for i in range(len(pair1)):
        if combo[i] == -1:
            combo[i] = pair2[idx_misvals[0]]
            idx_misvals.remove(idx_misvals[0])
    return combo

@jit(nopython=True)
def cross_prob(pair1, pair2, p1=.1):
    """
    For each city, choose by probability assigned to each pair
    Uses same ordering solution as cross_p_point to insure all cities remain in-use
    :param pair1: list of cities
    :param pair2: list of cities
    :param p1: probability for choosing pair1
    :return:
    """
    output = np.array([pair1[i] if np.random.uniform(0,1) <= p1 else -1 for i in range(len(pair1))])
    missing_vals = setdif(pair2, output)
    idx_misvals = [i for i in range(len(pair2)) if pair2[i] in missing_vals]
    for i in range(len(output)):
        if output[i] == -1:
            output[i] = pair2[idx_misvals[0]]
            idx_misvals.remove(idx_misvals[0])
    return output

@jit(nopython=True)
def setdif(a, b):
    result = []
    for i in range(len(a)):
        if not a[i] in b: result.append(a[i])
    return result

@jit(nopython=True)
def mut_neighbor_swap(city_order):
    swap_positions = np.random.choice(range(len(city_order)), size=2, replace=False)
    mutations = [True if np.random.uniform(0,1) <= .01 else False for i in range(len(city_order))]
    for i in range(len(city_order)-1):
        if mutations[i]:
            tmp = city_order[i]
            city_order[i] = city_order[i+1]
            city_order[i+1] = tmp
    return city_order

def mutate(individual, mutationRate=.01):
    """
    Copied from TDS article. FOR DEBUG ONLY
    https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
    :param individual:
    :param mutationRate:
    :return:
    """
    for swapped in range(len(individual)):
        if (np.random.random() < mutationRate):
            swapWith = int(np.random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual