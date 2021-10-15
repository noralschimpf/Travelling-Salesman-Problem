import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import euclidean_distance, route_animate, FPS
import tqdm, os
from matplotlib import pyplot as plt
import time
from matplotlib.animation import FuncAnimation, writers
from numba import jit


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
    idx = np.arange(params['k'])
    population = []
    for x in range(params['k']):
        np.random.shuffle(tmp)
        population.append(list(tmp))
    population = np.array(population)
    best_fits = np.zeros(params['g'])

    for g in range(params['g']):
        # Assign a probability of selection for each result using a fitness function
        #   Score each function with a heuristic (0 best, increasing for worse)
        fit = np.array([params['f_fit'](data[x-1]) for x in population])
        best_fits[g] = min(fit)

        fit_prob_inverse = fit / np.sum(fit)
        #   Assign probabilities by mapping scores to a linear scale -1 - 0, then invert
        fit = -1*(fit - max(fit))
        fit = fit / np.sum(fit)


        # Select and "breed" pairs
        #   Select pairs according to their probability
        pairs = [np.random.choice(idx, p=fit) for x in range(params['k']*2)]
        pair1, pair2 = population[pairs[:params['k']]],population[pairs[params['k']:]]
        #   Merge pairs via crossover function
        population = np.array([params['f_cross'](pair1[i], pair2[i]) for i in range(params['k'])])
        # Mutate pairs
        population = np.array([params['f_mut'](population[i]) for i in range(params['k'])])
    return {'soln': data[population[np.where(fit == min(fit))] - 1][0],
            'training': best_fits}



def GA_Simulate(data: dict, restrictions: dict, status: dict, params: dict):
    """
    Simulation environment for executing a Genetic Algorithm. Tracks multiple runs and statistics, Develops improvement curves, and generates animations / plots
    :param data: complete list of cities
    :param restrictions: path restrictions between cities / nodes
    :param status: tracks the status of the genetic algorithm. Includes DEBUG and ANIMATE flags
    :param params: dictionary of variable parameters including:
    {k: population size,g: generations, f_fit: fitness function, f_cross: crossover function, f_mut: mutation function,
    n: number of runs, animate: whether to build animations}
    :return:
    """
    nda_data = data['data']
    fit_curves = np.zeros((params['n'], params['g']))
    soln_fits = np.zeros(params['n'])
    best_fit_idx = -1; best_route = None
    GA_Name = 'Genetic fit-{} cross-{}'.format(params['f_fit'].__name__, params['f_cross'].__name__)
    frames = []

    sttime = time.time()
    for i in tqdm.trange(params['n']):
        soln_dict = geneticAlgo(data['data'], restrictions, status, params)

        # Add solution to memory
        fit_curves[i] = soln_dict['training']

        # add frame to animation using returned solution route
        frames.append(np.vstack((soln_dict['soln'], soln_dict['soln'][0])))

        # track best solution, plot if last iteration
        soln_fits[i] = euclidean_distance(soln_dict['soln'], loop=True)
        if soln_fits[i] < soln_fits[best_fit_idx] or best_fit_idx == -1:
            best_fit_idx = i
            best_route = np.vstack((soln_dict['soln'], soln_dict['soln'][0]))
        if i == params['n']-1:
            fig_last, ax_last = plt.subplots(1,1)
            ax_last.scatter(nda_data[:,1], nda_data[:,2])
            [ax_last.annotate(str(int(nda_data[i, 0])), (nda_data[i, 1] + 0.3, nda_data[i, 2] + 0.2)) for i in range(len(nda_data))]
            ax_last.plot(best_route[:, 1], best_route[:, 2], color='b')
            fig_last.suptitle('TSP {} (Dim: {})\ncost: {:.3f}'.format(
                GA_Name, data['DIMENSION'], soln_fits[best_fit_idx]))
            if not os.path.isdir('Figures/{}'.format(GA_Name)): os.makedirs('Figures/{}'.format(GA_Name))
            fig_last.savefig('Figures/{}/{}.png'.format(GA_Name, data['NAME']), dpi=300)
            fig_last.clf(); ax_last.cla(); plt.close()

    edtime = time.time()

    # Generate stats plot
    fig_stats, ax_stats = plt.subplots(1,1)
    ax_stats.hist(soln_fits, 50)
    ax_stats.set_xlabel("Fitness Score")
    ax_stats.set_ylabel("Count")
    fig_stats.suptitle("TSP {} (Dim: {})\nMean: {:.3f}    std:{:.3f}    time:{:.3f}s".format(
        GA_Name, data['DIMENSION'], soln_fits.mean(), soln_fits.std(), edtime - sttime))
    fig_stats.savefig('Figures/{}/stats.png'.format(GA_Name), dpi=300)
    fig_stats.clf(); ax_stats.cla(); plt.close()

    # Plot all training curves
    fig_train, ax_train = plt.subplots(1,1)
    for i in range(len(fit_curves)):
        ax_train.plot(fit_curves[i], color='b', alpha=0.3)
    ax_train.set_xlabel("Generation")
    ax_train.set_ylabel("Best Fitness Score")
    fig_train.suptitle("TSP {} (Dim: {})\nPopulation: {}".format(GA_Name, data['DIMENSION'], len(fit_curves)))
    fig_train.savefig("Figures/{}/train.png".format(GA_Name), dpi=300)
    fig_train.clf(); ax_train.cla(); plt.close()

    #Generate Animation
    if params['animate']:
        fig, ax = plt.subplots(1,1)
        ax.scatter(nda_data[:, 1], nda_data[:, 2], color='r')
        [ax.annotate(str(int(nda_data[i, 0])), (nda_data[i, 1] + 0.3, nda_data[i, 2] + 0.2)) for i in
         range(len(nda_data))]
        ax.set_title(GA_Name)
        line = ax.plot(nda_data[0, 1], nda_data[0, 2])[0]
        animation = FuncAnimation(fig, func=route_animate, fargs=(line, True), frames=frames)
        Writer = writers['ffmpeg']
        writer = Writer(fps=FPS, metadata={'artist': 'Me'}, bitrate=1800)
        animation.save('Figures/{}/concorde{}.mp4'.format(GA_Name, len(nda_data)), writer)
        fig.clf(); ax.cla(); plt.close()