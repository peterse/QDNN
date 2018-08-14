import matplotlib.pyplot as plt
import numpy as np
from pyquil.api import QVMConnection
from main import qRBM

from benchmark import encode, n_coin_toss
from main import get_permus
from collections import Counter
from scipy import stats


print("SCRATCH IMPORT")
# API for the QVM
QVM= QVMConnection()

# TESTING
# Globals to keep track of
HIDDEN_INDICES = [0]
VISIBLE_INDICES = [1,2,3,4]
DATA = np.asarray(encode(n_coin_toss(10)))
N_EPOCHS = 40


qrbm_inst = qRBM(QVM, VISIBLE_INDICES, HIDDEN_INDICES, debug=True)
Jf, Af, Bf = qrbm_inst.train(DATA, n_epochs=N_EPOCHS, eta=1)

permus = get_permus(4, mode="neg")
for perm in permus:
    print(perm)
    print(qrbm_inst.visibles_distribution(perm, Jf, Af, Bf))
