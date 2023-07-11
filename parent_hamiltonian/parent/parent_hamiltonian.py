"""
Complete implementation of the Parent Hamiltonian following:
    Fumiyoshi Kobayashi and Kosuke Mitarai and Keisuke Fujii
    Parent Hamiltonian as a benchmark problem for variational quantum
    eigensolvers
    Physical Review A, 105, 5, 2002
    https://doi.org/10.1103%2Fphysreva.105.052415
"""
import logging
import numpy as np
from scipy import linalg
import parent_hamiltonian.pauli.pauli_decomposition as pauli
import parent_hamiltonian.contractions.contractions as reduced
logger = logging.getLogger('__name__')


def get_null_projectors(array):
    """
    Given an input matrix this function computes the matriz of projectors
    over the null space of the input matrix

    Parameters
    ----------

    array : numpy array
        Input matrix for computing the projectors over the null space

    Returns
    _______

    h_null : numpy array
        Matrices with the projectors to the null space of the
        input array
    """

    if np.linalg.matrix_rank(array) == len(array):
        text = "PROBLEM! The rank of the matrix is equal to its dimension.\
            Input matrix DOES NOT HAVE null space"
        raise ValueError(text)

    # Compute the basis vector for the null space
    v_null = linalg.null_space(array)
    # Computing the projectors to the null space
    h_null = v_null @ np.conj(v_null.T)
    return h_null

def get_local_reduced_matrices(state):
    """
    Given a MPS representation of a input state computes the local
    reduced density matrices for each qubit of input state.

    Parameters
    ----------

    state : numpy array
        MPS representation of an state
    """
    #Getting the number of qubits of the MPS
    nqubit = state.ndim
    # Indexing for input MPS state
    state_index = list(range(nqubit))
    logger.info('state_index: {}'.format(state_index))
    local_qubits = []
    local_rho = []
    for qb_pos in range(nqubit):
        # Iteration over all the qbits positions
        logger.info('qb_pos: {}'.format(qb_pos))
        group_qbits = 1
        stop = False
        while stop == False:
            # Iteration for grouping one qubit more in each step of the loop
            free_indices = [(qb_pos + k)%nqubit for k in range(group_qbits + 1)]
            logger.debug('free_indices: {}'.format(free_indices))
            # The contraction indices are built
            contraction_indices = [
                i for i in state_index if i not in free_indices
            ]
            logger.debug('contraction_indices: {}'.format(contraction_indices))
            # Computing the reduced density matrix
            rho = reduced.reduced_matrix(
                state, free_indices, contraction_indices)
            # Computes the rank of the obtained reduced matrix
            rank = np.linalg.matrix_rank(rho)
            logger.debug('rank: {}. Dimension: {}'.format(rank, len(rho)))
            if rank < len(rho):
                # Now we can compute a null space for reduced density operator
                logger.debug('Grouped Qubits: {}'.format(free_indices))
                # Store the local qubits for each qubit
                local_qubits.append(free_indices)
                # Store the local reduced density matrices for each qubit
                local_rho.append(rho)
                stop = True
            group_qbits = group_qbits + 1

            if group_qbits == nqubit:
                stop = True
            logger.debug('STOP: {}'.format(stop))
    return local_qubits, local_rho


def parent_hamiltonian(state):
    """
    Given a MPS representation of a input state computes the local
    hamiltonian terms in Pauli Basis. This terms allows to build
    the parent hamiltonian of the input state.

    Parameters
    ----------

    state : numpy array
        MPS representation of an state
    Returns
    _______

    local_qubits : list
       list with the local qubits for each qubit of the initial state
    hamiltonian_coeficients : list
        list of the coefficients for each  hamiltonian term
    hamiltonian_paulis : list
        list of pauli strings for each hamiltonian term.
    hamiltonian_local_qubits : list
        list with the qbuits where the hamiltonian term is applied
    """

    #Getting the number of qubits of the MPS
    nqubit = state.ndim
    # Indexing for input MPS state
    state_index = list(range(nqubit))
    logger.info('state_index: {}'.format(state_index))
    #First we compute the local reduced density matrices for each qbuit
    local_qubits, local_rho = get_local_reduced_matrices(state)

    hamiltonian_coeficients = []
    hamiltonian_paulis = []
    hamiltonian_local_qubits = []
    for rho_step, free_indices in zip(local_rho, local_qubits):
        # Get the null projectors
        rho_null_projector = get_null_projectors(rho_step)
        # Compute paulo decomposition of null projectors
        coefs, paulis = pauli.pauli_decomposition(
            rho_null_projector, len(free_indices)
        )
        hamiltonian_coeficients = hamiltonian_coeficients + coefs
        hamiltonian_paulis = hamiltonian_paulis + paulis
        hamiltonian_local_qubits = hamiltonian_local_qubits \
            + [free_indices for i in paulis]
    return hamiltonian_coeficients, hamiltonian_paulis, hamiltonian_local_qubits

#    #list_of_results = []
#    local_qubits = []
#    local_rho = []
#    null_projectors = []
#    hamiltonian_coeficients = []
#    hamiltonian_paulis = []
#    for qb_pos in range(nqubit):
#        # Iteration over all the qbits positions
#        logger.info('qb_pos: {}'.format(qb_pos))
#        group_qbits = 1
#        stop = False
#        while stop == False:
#        #for group_qbits  in range(1, nqubit):
#            # Iteration for grouping one qubit more in each step of the loop
#            free_indices = [(qb_pos + k)%nqubit for k in range(group_qbits + 1)]
#            logger.debug('free_indices: {}'.format(free_indices))
#            # The contraction indices are built
#            contraction_indices = [
#                i for i in state_index if i not in free_indices
#            ]
#            logger.debug('contraction_indices: {}'.format(contraction_indices))
#            # Computing the reduced density matrix
#            rho = reduced.reduced_matrix(
#                state, free_indices, contraction_indices)
#            # Computes the rank of the obtained reduced matrix
#            rank = np.linalg.matrix_rank(rho)
#            logger.debug('rank: {}. Dimension: {}'.format(rank, len(rho)))
#            if rank < len(rho):
#                # Now we can compute a null space for reduced density operator
#                logger.debug('Grouped Qubits: {}'.format(free_indices))
#                # Now we can compute the null projector of the reduced density operator
#                rho_null_projector = get_null_projectors(rho)
#                coefs, paulis = pauli.pauli_decomposition(
#                    rho_null_projector, len(free_indices)
#                )
#                # Store the local qubits for each qubit
#                local_qubits.append(free_indices)
#                # Store the local reduced density matrices for each qubit
#                local_rho.append(rho)
#                # Store the null projectors for each qubit
#                null_projectors.append(rho_null_projector)
#                # Store the list of hamiltonian coeficients for each qubit
#                hamiltonian_coeficients = hamiltonian_coeficients + coefs
#                # Store the list of hamiltonian pauli strings for each qubit
#                hamiltonian_paulis = hamiltonian_paulis + paulis
#                hamiltonian_local_qubits = hamiltonian_local_qubits \
#                    + [free_indices for i in paulis]
#                #list_of_results.append(
#                #    [qb_pos, free_indices, rho, rho_null_projector, coefs, paulis]
#                #)
#                # We can stop the loop and begin with following qubit
#                stop = True
#            group_qbits = group_qbits + 1
#
#            if group_qbits == nqubit:
#                stop = True
#            logger.debug('STOP: {}'.format(stop))
#
#    return hamiltonian_coeficients, hamiltonian_paulis, hamiltonian_local_qubits

