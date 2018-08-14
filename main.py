
from pyquil.paulis import sX, sZ
from pyquil.gates import CNOT, RX
from pyquil.api import QVMConnection
from pyquil.quil import Program

from pyquil.paulis import PauliTerm, PauliSum
from grove.pyqaoa.qaoa import QAOA
from grove.pyvqe.vqe import VQE

# Optimizer for QAOA
from scipy.optimize import fmin_bfgs, minimize

import numpy as np

# for calculating partition function
import itertools

#TODO: initialization for qubit placeholders if we expect to run on QPU
# TODO: J, B as parameters of qRBM class

# u is the set v U h


def sigmoid(x):
    """
    args:
        x - np array type or scalar
    """
    return 1. / (1. + np.exp(-x))

def make_Hco(J, visible_indices, hidden_indices):
    """
    initialize a coupling Hamiltonian for a given coupling matrix J
    IMPORTANT: J IS INDEXED BY POSITIONAL INDEX, BUT Z GATES ARE
    IDENTIFIED BY QUBIT INDEX
    args:
        J - array type with dimension |u| x |u| (|u| = n_vis + n_hid)
    return: H_co with |H_co| = 2^|u|
    """
    out = []
    for p_i, q_i in enumerate(visible_indices):
        for p_j, q_j in enumerate(hidden_indices):
            #FIXME: Why the -1 in michael's code
            #FIXME: change the layout of the J coupling matrix?
            out.append(PauliSum([PauliTerm("Z", q_i, -1.0*J[p_i][p_j]) * PauliTerm("Z", q_j, 1.0)]))
    return out

def make_Hbias(A, B, visible_indices, hidden_indices):
    """
    initialize a bias Hamiltonian for bias array B. This array will describe the set u, with 0's at the location of visible units if the biases are for hidde terms only.
    IMPORTANT: B IS INDEXED BY POSITIONAL INDEX, BUT Z GATE IS
    IDENTIFIED BY QUBIT INDEX
    args:
        A - array type with dim 1 x |v| of visible biases
        B - array type with dim 1 x |h| of hidden biases
    return: H_bias with |H_bias| = 2^|u|
    """

    #FIXME: Do I add programs of gates or make programs of added gates????
    out = []
    # Bias weights must be ordered according to qubit indices
    for p_i, q_i in enumerate(visible_indices):
        #FIXME: What is up with this notation; can I replace it with sZ, etc?
        out.append(PauliSum([PauliTerm("Z", q_i, -1.0*A[p_i])]))

    for p_i, q_i in enumerate(hidden_indices):
        out.append(PauliSum([PauliTerm("Z", q_i, -1.0*B[p_i])]))

    return out

def make_Hdata_full(data_arr, visible_indices):
    """
    initialize a data Hamiltonian using the full dataset at once; to be used in
    a cost hamiltonian undergoing two total loops
    args:
        data_arr - array type with dim n_vis x N, where N = number of datapoints
    return: H_data
    """
    out = []
    for i in range(len(data_arr)):
        out += make_Hdata_partial(data_arr, i, visible_indices)
    return out

def make_Hdata_partial(data_arr, index, visible_indices):
    """
    initialize a data Hamiltonian corresponding to a single data point
    args:
        < see 'make_Hdata_full' >
    """
    out = []
    data_vec = data_arr[index]
    # CAREFUL: the data fed into visible units corresponds to the positional
    #   index of the visible qubits
    for p_j, q_j in enumerate(visible_indices):
        switch = -1.*(-1.)**data_vec[p_j]
        out.append(PauliSum([PauliTerm("Z", q_j, switch)]))
    return out


def make_Hmix(visible_indices, hidden_indices):
    """
    initialize a mixer hamiltonian for the purposes of qaoa
    args:
        *see method 'make_Hco'
    """

    out = []
    for i in visible_indices + hidden_indices:
        #FIXME: What is up with this notation; can I replace it with sZ, etc?
        out.append(PauliSum([PauliTerm("X", i, 1.0)]))

    return out

def get_permus(n, mode="bin"):
    """
    return a list of all permutations of an n-length bitstring, with each
    bitstring in listform
    kwargs:
        bin - lists of {0,1}
        neg - lists of {-1,1}
    """
    configs =["".join(seq) for seq in itertools.product("01", repeat=n)]
    configs = [[int(c) for c in s] for s in configs]

    # replace all 0's with -1's
    if mode == "neg":
        for k, lst in enumerate(configs):
            tmp = []
            for i in lst:
                if i == 0:
                    tmp.append(-1)
                else:
                    tmp.append(i)
            configs[k] = tmp
    return configs


class qRBM:

    def __init__(self, qvm, visible_indices, hidden_indices, debug=False):
        """
        create an RBM with the specified number of visible and hidden units params:
            qvm - a valid pyquil QVM connection
            n_vis - Number of visible units in RBM (int)
    		n_hid - Number of hidden units in RBM (int)
    		debug - print optional statements and checkpoints
		"""

        self.visible_indices = visible_indices
        self.hidden_indices = hidden_indices

        self.n_vis = len(visible_indices)
        self.n_hid = len(hidden_indices)

        # just some shortcuts for iterating
        self.all_indices = visible_indices + hidden_indices
        self.n_qubits = len(self.all_indices)
        self.debug = debug

        # hardcoded stats params for QAOA
        self.beta_temp = 2
        self.state_angle_init = np.arctan(np.e**(-self.beta_temp/2.0)) * 2.0
        self.n_measure = None # no. measurements for expectation; None=analy
        self.n_qaoa_steps = 1
        self.qaoa_minimizer = fmin_bfgs
        self.seed = 9932

        # spin up a VQE object
        self.vqe_inst = VQE(minimizer=minimize, minimizer_kwargs={'method': 'nelder-mead'})

        self.QVM = qvm


    def get_expectation(self, state, op):
        """
        get the expectation value of <state|op|state>

        args:
            state - some quantum state of objtype=?????
            op - an operator of obtype=???? with dim = dim(state)
        """
        return self.vqe_inst.expectation(state, op, self.n_measure, self.QVM)


    def get_QAOA(self, H_cost):
        """
        wrapped QAOA object, where stats and QVM params are taken care of by
        the parent QVM object

        args:
            H_mix - mixer hamiltonian of dim ??
            H_cost - cost hamiltonian of dim ??

        return:
            nu_arr, gamma_arr - arrays of nu,gamma vars len = n_qaoa_steps
        """

        # prepare the mixing hamiltonian
        H_mix = make_Hmix(self.visible_indices, self.hidden_indices)

        # Doing entanglement of nearest neighbors over QUBIT INDICES
        state_prep = Program()
        for i in range(self.n_qubits-1):
            tmp = Program()
            tmp.inst(RX(self.state_angle_init, i), CNOT(i, i+1))
            state_prep += tmp

        qaoa_inst = QAOA(
                    self.QVM,
                    qubits=self.all_indices,
                    steps=self.n_qaoa_steps,
                    ref_ham=H_mix,
                    cost_ham=H_cost,
                    driver_ref=state_prep,
                    store_basis=True,
                    minimizer=self.qaoa_minimizer,
                    minimizer_kwargs={'maxiter':50},
                    vqe_options={'samples': self.n_measure},
                    rand_seed=self.seed)

        return qaoa_inst

    def train(self, data_arr, eta=0.1, n_epochs=100):
        """
        args:
            data_arr - array type with dim n_vis x N, where N = number of datapoints
        kwargs:
            eta - learning rate to apply for update rule
            n_epochs - number of outer-loop optimization steps
        """

        # (0) initialize the weights and biases to whatever
        A = np.asarray(np.random.rand(self.n_vis))
        B = np.asarray(np.random.rand(self.n_hid))
        J = np.asarray(np.random.rand(self.n_vis, self.n_hid))

        H_data = make_Hdata_full(data_arr, self.visible_indices)

        for epoch in range(n_epochs):

            H_co = make_Hco(J, self.visible_indices, self.hidden_indices)
            H_bias = make_Hbias(A, B, self.visible_indices, self.hidden_indices)

            # (1) define cost hamiltonian
            H_C = H_data + H_co + H_bias

            # (2), (3) - thermalize with QAOA, then update weights
            J, A, B = self._thermalize_and_update(H_C, J, A, B, eta)

            if self.debug:
                print("EPOCH %i J-transformations" % epoch)
                print(self.transform(data_arr, J))
        return J, A, B

    def train_batched(self, data_arr, eta=0.1, n_epochs=100):
        """
        A training scheme using batched data hamiltonian
        args:
            < see 'train' method >
        """

        # (0) initialize the weights and biases to whatever
        A = np.asarray(np.random.rand(self.n_vis))
        B = np.asarray(np.random.rand(self.n_hid))
        J = np.asarray(np.random.rand(self.n_vis, self.n_hid))

        for epoch in range(n_epochs):

            for k,data_vec in enumerate(data_arr):
                # The batched hamiltonian corresponds to only a single datapoint
                H_data = make_Hdata_partial(data_arr, k, self.visible_indices)
                H_co = make_Hco(J, self.visible_indices, self.hidden_indices)
                H_bias = make_Hbias(A, B, self.visible_indices, self.hidden_indices)
                # (1) define cost hamiltonian
                H_C = H_data + H_co + H_bias

                # (2), (3) - thermalize with QAOA, then update weights
                J, A, B = self._thermalize_and_update(H_C, J, A, B, eta)

                if self.debug:
                    print("EPOCH %i, k=%i" % (epoch, k))
                    print(self.transform(data_arr, J))

        return J, A, B


    def _thermalize_and_update(self, H_C, J, A, B, eta):

        # (2) run thermalization on this cost hammy
        qaoa_inst = self.get_QAOA(H_C)
        nu_arr, gam_arr = qaoa_inst.get_angles()
        qaoa_prog = qaoa_inst.get_parameterized_program()

        # the hstack is just constructing |beta0...beta_N,gamma0...gamma_N>
        s_final = qaoa_prog(np.hstack((nu_arr, gam_arr)) )

        # (3) updates to J, B are made directly using POSITIONAL INDICES
        deltaJ = np.zeros_like(J) # table of updates for J values
        for p_i, q_i in enumerate(self.visible_indices):
            for p_j, q_j in enumerate(self.hidden_indices):
                deltaJ[p_i][p_j] = eta*self.get_expectation(s_final, sZ(p_i)*sZ(p_j))

        deltaA = np.zeros_like(A) # table of updates for A values
        for p_i, q_i in enumerate(self.visible_indices):
            deltaA[p_i] = eta*self.get_expectation(s_final, sZ(q_i))

        deltaB = np.zeros_like(B) # table of updates for B values
        for p_i, q_i in enumerate(self.hidden_indices):
            deltaB[p_i] = eta*self.get_expectation(s_final, sZ(q_i))

        J = J - deltaJ
        A = A - deltaA
        B = B - deltaB

        # Printing wavefunctions
        wf = self.QVM.wavefunction(s_final)
        wf = wf.amplitudes
        # TODO:
        for state_index in range(2**len(qaoa_inst.qubits)):
            print(qaoa_inst.states[state_index], np.conj(wf[state_index])*wf[state_index])
        # print("s_final")
        # print(np.hstack( (nu_arr, gam_arr)))
        print("DELTA J =")
        print(deltaJ)
        return J, A, B

    def transform(self, data_arr, J):
        """
        ?
        """
        # FIXME: I'm not sure what's going on here...
        print(J)
        # print(data_arr)
        # print(np.dot(data_arr, J))
        return sigmoid(np.dot(data_arr, J))

    def get_Z(self, J, A, B):
        """
        calculate the partition function for the classicle RBM state
        configurations dependent on the trained weights J, b
        args:
            J -
            b -
        """

        vis_configs = get_permus(self.n_vis, mode="neg")
        hid_configs = get_permus(self.n_hid, mode="neg")

        Z_tot = 0
        # iterate over all possible hidden and visible configurations
        for vis_config in vis_configs:
            for hid_config in hid_configs:
                E_ij = 0
                # do a classical energy calculation using H_C (No H_D)
                # As always, J,A,B are indexed with positional indices
                for p_i, v in enumerate(vis_config):
                    for p_j, h in enumerate(hid_config):
                        E_ij += J[p_i][p_j]*v*h + A[p_i]*v + B[p_j]*h
                        # if self.debug:
                        #     print("i=%i j=%i" % (i,j))
                        #     print("J_ij=%3.2f, A_i=%3.2f, B_i=%3.2f" % (J[i][j], A[i], B[j]))
                        #     print("v=%3.2f, h=%3.2f" % (v,h))
                        #     print("E=%3.3f \n" % E_ij)
                Z_tot += np.exp(E_ij) # double negative in the exponent

        return Z_tot

    def visibles_distribution(self, vis_config, J, A, B):
        """
        For a trained RBM, compute the probability distribution over
        possible visible layer configurations
        args:
            vis_config - array of visible unit states for which to calculate the probability
        """

        Z = self.get_Z(J, A, B)

        # The probability for a fixed input config requires iteration over
        # all possible hidden config states
        hid_configs = get_permus(self.n_hid, mode="neg")
        # outer loop is summing a partial partition function
        tot = 0
        for hid_config in hid_configs:
            # inner loops are summing up components of this energy term
            E_ij = 0
            for p_i, v in enumerate(vis_config):
                for p_j, h in enumerate(hid_config):
                    E_ij += J[p_i][p_j]*v*h + A[p_i]*v + B[p_j]*h
            tot += np.exp(E_ij) # double negative
            # if self.debug:
            #     print("i=%i j=%i" % (p_i,p_j))
            #     print("E=", E_ij)
        return tot/Z

if __name__ == "__main__":
    # Globals to keep track of
    HIDDEN_INDICES = [0]
    VISIBLE_INDICES = [1,2,3,4]
    N_EPOCHS = 4
    # API for the QVM
    QVM= QVMConnection()
    # Create a RBM instance and train it on the dataset
