""""
benchmark.py - scripts for benchmarking the performance of the QDNN
Evan Peters 2018
"""

import matplotlib.pyplot as plt
from pyquil.api import QVMConnection
import numpy as np

from main import qRBM
#TODO: 'main' is a shitty module name...


# for statistics and distributions
from collections import Counter
from scipy import stats

def n_coin_toss(n):
    return [1 if x>=.5 else 0 for x in np.random.rand(n)]

def encode(lst):
    # a simply encoding scheme to
    # send 1->[-1,-1,1,1] and 0 -> [1,1,-1,-1]
    out = []
    for v in lst:
        if v == 1:
            out.append([-1,-1,-1,1])
        elif v==0:
            out.append([1,-1,1,-1])
    return out

#FIXME: whats up with the minimizer? Why choose this one?

if  __name__ == "__main__":
    # API for the QVM
    QVM= QVMConnection()

    # TESTING
    # Globals to keep track of
    HIDDEN_INDICES = [0]
    VISIBLE_INDICES = [1,2,3,4]
    DATA = np.asarray(encode(n_coin_toss(10)))
    N_EPOCHS = 4

    # Plotting objects
    epochs_lst = range(N_EPOCHS)
    fig = plt.figure()


    # Create a RBM instance and train it on the dataset
    qrbm_inst = qRBM(QVM, VISIBLE_INDICES, HIDDEN_INDICES, debug=True)
    #Jf, Af, Bf = qrbm_inst.train(DATA, n_epochs=N_EPOCHS)
    Jf_batched, Af_batched, Bf_batched = qrbm_inst.train_batched(DATA, n_epochs=N_EPOCHS)

    # Recover RBM's probability distribution
    p1 = qrbm_inst.visibles_distribution([-1,-1,1,1], Jf_batched, Af_batched, Bf_batched)
    p0 = qrbm_inst.visibles_distribution([1,1,-1,-1], Jf_batched, Af_batched, Bf_batched)
    print("UNBATCHED RESULTS AFTER %i EPOCHS")
    print(qrbm_inst.transform(DATA, Jf))
    print("BATCHED RESULTS AFTER %i EPOCHS")
    print(qrbm_inst.transform(DATA, Jf_batched))
    print("BATCHED DISTRIBUTION ON [-1,-1,1,1]")
    print(p1)
    print("BATCHED DISTRIBUTION ON [1,1,-1,-1]")
    print(p0)

    # KL-divergence ranges
    stats.entropy(pk=[p1, p0], qk=[.5,.5])
