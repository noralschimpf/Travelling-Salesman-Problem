import os, time, tracemalloc
import Dataloader as dl
from Algos import randApproach as alg_simple
from utils import report, sortByComplexity

datadir = os.path.join(os.path.abspath('.'), 'Data')
DEBUG = False

def main(algo):
    files = os.listdir(datadir)
    files = sortByComplexity(files)
    for f in range(len(files)):
        if DEBUG: files[f] = 'Random4.tsp'
        dict_data = dl.load_tsp(os.path.join(datadir,files[f]))
        print(dict_data['NAME'])
        traceflag = f <= 4
        if traceflag: tracemalloc.start()
        sttime = time.time()
        if traceflag: stcur, stpeak = tracemalloc.get_traced_memory()
        opt_soln = algo(dict_data['data'])
        if traceflag:
            edcur, edpeak = tracemalloc.get_traced_memory()
        else: edpeak = -1000
        edtime = time.time()
        tracemalloc.stop()


        metrics = {'time': edtime - sttime, 'memory': edpeak/1000}
        soln = {'name': algo.__name__, 'soln': opt_soln}
        report(dict_data, soln, metrics)








if __name__ == '__main__':
    main(alg_simple.OptimizedBruteForce)