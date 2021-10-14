import numpy as np
from utils import euclidean_distance
from sklearn.preprocessing import MinMaxScaler

def geneticAlgo(data: np.array, restrictions: dict, status: dict, params: dict):
    '''
    Modular implementation of a genetic algorithm
    :param data: complete list of cities
    :param restrictions: path restrictions between cities / nodes
    :param status: tracks the status of the genetic algorithm. Includes DEBUG and ANIMATE flags
    :param params: dictionary of variable parameters including:
    {k: population size,g: generations, f_fit: fitness function, f_cross: crossover function, f_mut: mutation function}
    :return:
    '''
    # Generate k random samples for the population
    tmp = data[:,0].astype(int)
    idx = np.arange(len(tmp))
    population = np.vstack([np.random.shuffle(tmp) for x in range(params['k'])])

    for g in range(params['g']):
        # Assign a probability of selection for each result using a fitness function
        #   Score each function with a heuristic (0 best, increasing for worse)
        fit = np.array([params['f_func'](data[x-1]) for x in tmp])
        # If last generation, return the best-fit individual
        if g == params['g']: return data[population[np.where(fit == min(fit))]]
        fit = fit / np.sum(fit)
        #   Assign probabilities by mapping scores to a linear scale -1 - 0, then invert
        scl = MinMaxScaler(feature_range=(-1,0))
        fit = -1.*scl.fit_transform(fit)


        # Select and "breed" pairs
        #   Select pairs according to their probability
        pairs = [np.random.choice(idx,p=fit) for x in range(params['k']*2)]
        pair1, pair2 = population[pairs[:params['k']]],population[pairs[params['k']:]]
        #   Merge pairs via crossover function
        population = np.array([params['f_cross'](pair1[i], pair2[i]) for i in range(params['k'])])
        # Mutate pairs
        population = np.array([params['f_mut'](population[i]) for i in range(params['k'])])

def fit_euclid(nda_data: np.array):
    """
    Fitness function using euclidean norm
    :param nda_data: solution route, not looped
    :return: total euclidean distance
    """
    return euclidean_distance(nda_data, loop=True)


def fit_intersect(nda_data: np.array):
    """
    Fitness function using number of times route segments overlap
    :param nda_data: solution route, not looped
    :return: total number of intersections
    """

    # initialize
    ints = 0
    nda_data = np.vstack((nda_data, nda_data[0]))

    # for each segment
    #     ints += number of ints with previous segs
    for i in range(len(nda_data)) - 1:
        for j in range(i):
            ints += is_intersect(nda_data[i:i+2], nda_data[j:j+2])

    return ints


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

def ccw(A,B,C):
    """
    See is_intersect stackoverflow link
    """
    return (C[2]-A[2]) * (B[2]-A[1]) > (B[2]-A[2]) * (C[1]-A[1])