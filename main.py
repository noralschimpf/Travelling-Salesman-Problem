import os, time, tracemalloc
import Dataloader as dl
from Structures.node import Node
from Algos import randApproach as alg_simple
from Algos import uninformedSearch as alg_uninformed
from utils import report, sortByComplexity

# datadir = os.path.join(os.path.abspath('.'), 'Data', 'General')
datadir = os.path.join(os.path.abspath('.'), 'Data', 'Prj2Data')
restrictions = None

Requires_initNode = ['DepthFirstSearch_Tree', 'IterativeDepthSearch']

if 'Prj2Data' in datadir:
    restrictions = dl.prj2_restrictions
    loop = False
    prj2 = True

DEBUG = False

def main(algo):
    # load relevant TSP data
    files = os.listdir(datadir)
    # files = sortByComplexity(files)
    for f in range(len(files)):
        if DEBUG and 'General' in datadir: files[f] = 'Random4.tsp'
        dict_data = dl.load_tsp(os.path.join(datadir,files[f]))
        print(dict_data['NAME'])
        traceflag = f <= 4 or prj2
        if traceflag: tracemalloc.start()
        sttime = time.time()
        if traceflag: stcur, stpeak = tracemalloc.get_traced_memory()
        if algo.__name__ in Requires_initNode:
            node_init = Node(parent=None, state=dict_data['data'][0], children=restrictions['1'])
            opt_soln = algo(node=node_init, data=dict_data['data'], restrictions=restrictions, status={'cutoff': False, 'failure': False, 'limit': -1})
        else: opt_soln = algo(dict_data['data'], restrictions, {'cutoff': False, 'failure': False, 'limit': -1})
        if traceflag:
            edcur, edpeak = tracemalloc.get_traced_memory()
        else: edpeak = -1000
        edtime = time.time()
        tracemalloc.stop()

        if isinstance(opt_soln, dict):
            print("{alg} failed:\t\t\t{dict}".format(alg=algo.__name__, dict=opt_soln))
            return

        metrics = {'time': edtime - sttime, 'memory': edpeak/1000}
        soln = {'name': algo.__name__, 'soln': opt_soln}
        report(dict_data, soln, metrics, loop=loop, scalems=prj2)








if __name__ == '__main__':
    # Test all search algorithms from alg_uninformed (project 2 algorithsm)
    lab2_algos = [x for name, x in alg_uninformed.__dict__.items() if callable(x) and 'Search' in name]
    for alg in lab2_algos: main(alg)