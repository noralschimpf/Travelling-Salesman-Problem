import numpy as np, matplotlib.pyplot as plt, os
import re

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

def euclidean_distance(soln: np.array):
    """
    Computes the total Euclidean distance of a proposed route
    :param soln: numpy array of the proposed route, with columns [city number, X coordinate, Y coordinate]
    :return: Sum of Euclidean distances for each segment
    """
    dist = 0.
    for i in range(soln.shape[0]):
        path_dist = 0.
        if i == soln.shape[0] - 1: path_dist = np.sqrt((soln[0,1] - soln[i,1])**2 + (soln[0,2] - soln[i,2])**2)
        else: path_dist = np.sqrt((soln[i+1,1] - soln[i,1])**2 + (soln[i+1,2] - soln[i,2])**2)
        dist += path_dist
    return dist



def report(data: dict, soln: dict, metrics: dict):
    """
    Generates report figure of proposed route
    :param data: dictionary containing the cities, problem name, and any miscellaneous information
    :param soln: dictionary containing the route, algorithm used, and miscellaneous information
    :param metrics: dictionary containing the runtime and memory usage of the solution
    :return:
    """
    cost = euclidean_distance(soln['soln'])
    fig, ax = plt.subplots(1,1)
    nda_dat = data['data']; nda_soln = soln['soln']
    nda_soln = np.concatenate((nda_soln, nda_soln[0].reshape(1,-1)))
    ax.scatter(nda_dat[:,1], nda_dat[:,2], color='r')
    ax.plot(nda_soln[:,1], nda_soln[:,2], color='b')
    fig.suptitle('TSP {} (Dim: {})\ncost: {:.3f}   time:{:.3f} s   mem: {:.3f} KB'.format(soln['name'], data['DIMENSION'],cost, metrics['time'], metrics['memory']))
    if not os.path.isdir('Figures/{}'.format(soln['name'])): os.makedirs('Figures/{}'.format(soln['name']))
    fig.savefig('Figures/{}/{}.png'.format(soln['name'], data['NAME']), dpi=300)