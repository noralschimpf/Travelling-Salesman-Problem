import numpy as np
import tqdm
import itertools
from utils import euclid_partial as euclid

def RandomApproach(data: np.array):
    soln = np.zeros_like(data)

    # Select initial point at random
    start = np.random.randint(0,data.shape[0])
    soln[0] = data[start]; data = np.delete(data, start, axis=0)

    for iter in range(data.shape[0]):
        row = np.random.randint(0,data.shape[0])
        soln[iter+1] = data[row]; data = np.delete(data, row, axis=0)

    return soln

def BruteForce(data: np.array):
    soln = np.zeros_like(data)

    # generate a matrix of all permutations of points
    nda_perm = np.array(list(itertools.permutations(data[:,0])))

    # find the minimum cost of all permutations
    costs = np.zeros((nda_perm.shape[0]))
    for i in range(nda_perm.shape[0]):
        tmp_soln = np.zeros_like(soln)
        tmp_soln[:,0] = nda_perm[i]
        tmp_soln[:,1:] = [data[np.where(data == tmp_soln[x,0])[0],1:] for x in range(tmp_soln.shape[0])]
        costs[i] = euclid(tmp_soln)
        del tmp_soln

    soln_idx = np.argmin(costs)
    soln[:,0] = nda_perm[soln_idx]
    soln[:,1:] = [data[np.where(data == soln[x,0])[0],1:] for x in range(soln.shape[0])]
    return soln

def CleanerBruteForce(data: np.array):
    soln = np.zeros_like(data)

    # generate a matrix of all permutations of points
    gen_perm = itertools.permutations(data[:, 0])
    gen_destruct = itertools.permutations(data[:, 0])
    gen_length = sum(1 for ignore in gen_destruct)
    del gen_destruct

    # find the minimum cost of all permutations
    costs = np.full((gen_length), fill_value=np.nan)
    for i, permutation in tqdm.tqdm(enumerate(gen_perm), total=gen_length):
        tmp_soln = np.zeros_like(soln)
        tmp_soln[:, 0] = permutation
        tmp_soln[:, 1:] = [data[np.where(data == tmp_soln[x, 0])[0], 1:] for x in range(tmp_soln.shape[0])]
        costs[i] = euclid(tmp_soln)
        if np.nanmin(costs) == costs[i]: soln = tmp_soln
    return soln

def OptimizedBruteForce(data: np.array):
    soln = np.zeros_like(data)

    # generate a matrix of all permutations of points
    gen_perm = itertools.permutations(data[:, 0].astype(int))
    gen_destruct = itertools.permutations(data[:, 0].astype(int))
    perm_length = sum(1 for ignore in gen_destruct)
    del gen_destruct

    # find the minimum cost of all permutations
    min_cost = np.nan
    for i, permutation in tqdm.tqdm(enumerate(gen_perm), total=perm_length):
        tmp_soln = data[[x-1 for x in permutation]]
        cost = euclid(tmp_soln)
        if cost < min_cost or np.isnan(min_cost): min_cost = cost; soln = tmp_soln
    return soln
