"""
This module contains all the original functions from github:

    https://github.com/FumiKobayashi/Parent_Hamiltonian_as_a_benchmark_problem
    _for_variational_quantum_eigensolvers

for implement and reproduce results from the Parent Hamiltonian paper:

    Fumiyoshi Kobayashi and Kosuke Mitarai and Keisuke Fujii
    Parent Hamiltonian as a benchmark problem for variational quantum
    eigensolvers
    Physical Review A, 105, 5, 2002
    https://doi.org/10.1103%2Fphysreva.105.052415

"""
import string
import logging
import numpy as np
from scipy import linalg
logger = logging.getLogger('__name__')


def Base_10_to_2_list(x_in, nqubit):
    """
    Transform integer to binary representation of nqubit

    Parameters
    ----------

    x_in: int
        integer number for transforming
    nqubit: int
        number of bits for binary reulting number

    Returns
    _______

    binx : list
        list of integers with the binary representation of input
        integer in nqubit bits

    """
    binx = bin(x_in)
    binx = binx[2:]
    l_state = '{:0>'+str(nqubit)+'}'
    binx = l_state.format(binx)
    binx = list(binx)
    binx = [int(i) for i in binx]
    return binx

def z_rotation(angle):
    """
    Implement a Z rotation numpy array.

    Parameters
    ----------

    angle: float
        desired angle for the Z rotation

    Returns
    _______

    zr : numpy array
        Array with the implementation of a Z rotation

    """
    zr = np.array([
        [np.exp(-0.5j*angle), 0],
        [0, np.exp(0.5j*angle)]
    ])
    # original
    #zr = np.array([
    #    [np.exp(0.5j*angle), 0],
    #    [0, np.exp(-0.5j*angle)]
    #])
    return zr

def x_rotation(angle):
    """
    Implement a X rotation numpy array.

    Parameters
    ----------

    angle: float
        desired angle for the Z rotation

    Returns
    _______

    xr : numpy array
        Array with the implementation of a X rotation

    """
    xr = np.array([
        [np.cos(0.5*angle), -1j*np.sin(0.5*angle)],
        [-1j*np.sin(0.5*angle), np.cos(0.5*angle)]
    ])
    # Originally
    #xr = np.array([
    #    [np.cos(0.5*angle), 1j*np.sin(0.5*angle)],
    #    [1j*np.sin(0.5*angle), np.cos(0.5*angle)]
    #])
    return xr


def phasechange_gate():
    """
    Implement a Phase Change Gate as a numpy array.

    Returns
    _______

    phase : numpy array
        Array with the implementation of a Phase change gate

    """
    phase = np.array([
        [1, 1],
        [1, -1]
    ])
    return phase

def deltatensor():
    """
    Implement a delta tensor a numpy array.

    Returns
    _______

    deltaijk : numpy array
        Array with the implementation of a Phase change gate

    """
    deltaijk = np.zeros((2, 2, 2))
    deltaijk[0][0][0] = 1
    deltaijk[1][1][1] = 1
    return deltaijk

def zero_ket():
    """
    Implements the |0> 1 qbit state as a numpy array

    Returns
    _______

    deltaijk : numpy array
        Array with the implementation of a delta tensor

    """

    zeroket = np.array([1, 0])
    return zeroket


def one_qubit_d3_ansatz_unit(angle0, angle1, angle2, angle3, angle4, angle5):
    """
    Implements the MPS of a 1 qubit for the ansatz

    Parameters
    ----------

    angle0, angle1, ..., angle5: float
        desired angle for the the rotations of the ansatz

    Returns
    _______

    deltaijk : numpy array
        Array with the implementation of a delta tensor

    """
    x_rot0 = x_rotation(angle0)
    z_rot0 = z_rotation(angle1)
    x_rot1 = x_rotation(angle2)
    z_rot1 = z_rotation(angle3)
    x_rot2 = x_rotation(angle4)
    z_rot2 = z_rotation(angle5)
    phase_change_gate = phasechange_gate()
    delta_tensor = deltatensor()
    zeroket = zero_ket()

    unit0 = np.einsum(
        'AB, BCi, jCD, jk, DE, E -> Aik',
        z_rot0, delta_tensor, delta_tensor, phase_change_gate, x_rot0, zeroket
    )
    unit1 = np.einsum(
        'AB, BCi, jCD, jk, DE -> AEik',
        z_rot1, delta_tensor, delta_tensor, phase_change_gate, x_rot1
    )
    unit2 = np.einsum(
        'AB, BCi, jCD, jk, DE -> AEik',
        z_rot2, delta_tensor, delta_tensor, phase_change_gate, x_rot2)
    unit = np.einsum('ABij, BCkl, Cmn -> Aijklmn', unit2, unit1, unit0)
    return unit

def nqubit_mps(nqubit, angle_list):
    """
    Prepare the MPS of the ansatz for a given number of quibts

    Parameters
    ----------

    nqubit : int
        number of qubits of the ansatz

    angle_list: 1-D numpy array
        numpy array of 1D with the values of the angles of the ansatz

    Returns
    _______

    state : numpy array
        numpy array with the state of the MPS


    physical_index1: string

    """
    #state_conj : numpy array
    #    numpy array with the complex conjugate of the state of the MPS

    unit = one_qubit_d3_ansatz_unit(
        angle_list[0],
        angle_list[1],
        angle_list[2],
        angle_list[3],
        angle_list[4],
        angle_list[5]
    )

    abc = list(string.ascii_uppercase)
    index_segment = ''
    physical_index1 = 'A'
    state = unit
    for n_ in range(1, nqubit):
        physical_index2 = abc[n_]
        index_segment = '{0}ijklmn, {1}jolpnq -> {0}{1}iokpmq'.format(
            physical_index1, physical_index2
        )
        logger.debug("index_segment: {}".format(index_segment))
        state = np.einsum(index_segment, state, unit)
        #print(index_segment)
        physical_index1 += physical_index2

    #print('{0}iijjkk -> {0}'.format(physical_index1))

    mps_contraction = '{0}iijjkk -> {0}'.format(physical_index1)
    logger.debug("mps contraction: {}".format(mps_contraction))
    state = np.einsum(mps_contraction, state)
    #state_conj = np.conj(state)
    return state, physical_index1
    #return state, state_conj, physical_index1

def nqubit_state_locality_checker(state, physical_index, nqubit):

    state_conj = np.conj(state)
    flag = 1
    abc = string.ascii_lowercase
    for i in range(2, nqubit):
        if flag == 0:
            pass
        elif i < nqubit:
            physical_index_conj = abc[:i]
            index_temp = '{0}, {1}{2} -> {3}{1}'.format(
                physical_index,
                physical_index_conj,
                physical_index[i:],
                physical_index[:i]
            )
            logger.debug('index_temp: {}'.format(index_temp))
            rho = np.einsum(index_temp, state, state_conj)
            #print(index_temp)
            rho_matrix = rho.reshape(2**i, 2**i)
            rank = np.linalg.matrix_rank(rho_matrix)
            locality = '{0}-local density matrix rank:{1}(full_rank:{2})'\
                .format(i, rank, 2**i)
            logger.debug('{}'.format(locality))
            #print('{0}-local density matrix rank:{1}(full_rank:{2})'.format(
            #    i, rank, 2**i))
            if rank < 2**i:
                flag = 0
                local = i
                logger.debug('{}-local state'.format(i))
                #print('{}-local state'.format(i))
            else:
                pass
        else:
            logger.warning('subspace size is higher than {}'.format(nqubit))
            break
    return rho_matrix, rank, local

def null_projector(rho_matrix, rank, local):
    matrix_size = 2**local
    null_space = linalg.null_space(rho_matrix)
    #print(np.linalg.matrix_rank(null_space))
    null_projector_ = 0
    for i in range(matrix_size - rank):
        null_vec = null_space[:, i]
        null_vec = null_vec.reshape(matrix_size, 1)
        null_vec_mat = null_vec @ np.conj(null_vec.T)
        null_projector_ += null_vec_mat
    return null_projector_, matrix_size, null_space

def process_pauliproduct(i, nqubit, local, null_projector):
    l = Base_10_to_n(i, 4)
    size = '{:0>'+str(local)+'}'
    l = size.format(l)
    lis = list(l)
    lis = [int(s) for s in lis]
    #Pauli
    pauliproduct_temp = 1
    for j in range(local):
        pauliproduct_temp = np.kron(pauliproduct_temp, paulioperator(lis[j]))
    paulimatrix = pauliproduct_temp
    #print(paulimatrix)
    pauliproducts_factor = np.einsum(
        'ij,ij->', np.conj(null_projector), paulimatrix)/(2**local)
    #Pauli
    pauli_product_str = l.replace("0", "I").replace("1", "X")\
        .replace("2", "Y").replace("3", "Z")

    pauli_product_lis = list(pauli_product_str)
    pauliproducts_factor_templist = []
    for m in range(nqubit):
        pauli_product = ""
        for k in range(local-1):
            pauli_product += str(pauli_product_lis[k]) + " " + \
                str((k+m)%nqubit) + " "
        pauli_product += str(pauli_product_lis[local-1]) + " " + \
            str((local-1+m)%nqubit)
        pauliproducts_factor_templist.append(
            (pauliproducts_factor.real, pauli_product)
        )
    return pauliproducts_factor_templist

def paulioperator(pauli_index):
    """
    Return the correspondent Pauli matrix.

    Parameters
    ----------

    pauli_index : int
        Number for selecting the Pauli matrix:
        0 -> identity, 1-> X, 2-> Y, 3 -> Z

    Returns
    -------

    pauli = np.array
        2x2 Pauli Matrix

    """
    if pauli_index == 0:
        pauli = np.identity(2, dtype=np.complex128)
    elif pauli_index == 1:
        pauli = np.array([[0, 1], [1, 0]])
    elif pauli_index == 2:
        pauli = np.array([[0, -1j], [1j, 0]])
    elif pauli_index == 3:
        pauli = np.array([[1, 0], [0, -1]])
    return pauli

def Base_10_to_n(X, n):
    if (int(X/n)):
        return Base_10_to_n(int(X/n), n)+str(X%n)
    return str(X%n)

def parent_hamiltonian(ansatz, index, nqubit):
    """
    Creates the local parent Hamiltonian for the ansatz of the
    Parent Hamiltonian paper.

    Parameters
    ----------

    ansatz : numpy array
        numpy array with MPS representation of the ansatz state
    index : string
        indices of the ansatz MPS
    nqubit : int
        number of qubits for the ansatz

    Returns
    -------

    pauli_decomp : list
        Each element of the list is a tuple with coefficient and its
        correspondent pauli string. The complete list built the local
        parent hamiltonian
    rho : numpy array
        Local reduced density matrix of the ansatz


    """
    # Parent Hamiltonian Procedure
    # Getting minimum local density matrix
    rho, rank, local = nqubit_state_locality_checker(ansatz, index, nqubit)
    #Kernel of the minimum density matrix
    null_projector_, matrix_size, _ = null_projector(rho, rank, local)
    # Creation of Pauli operator list
    parallel_list = [process_pauliproduct(i, nqubit, local, null_projector_) \
        for i in range(4**local)]
    pauli_decomp = []
    for i in range(4**local):
        pauli_decomp += parallel_list[i]
    return pauli_decomp, rho, null_projector_
