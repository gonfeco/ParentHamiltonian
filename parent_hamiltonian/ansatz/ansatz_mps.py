"""
This module contains all functions needed for creating MPS for
the ansatz of the Parent Hamiltonian paper in an occidental index
convention:

    Fumiyoshi Kobayashi and Kosuke Mitarai and Keisuke Fujii
    Parent Hamiltonian as a benchmark problem for variational quantum
    eigensolvers
    Physical Review A, 105, 5, 2002
    https://doi.org/10.1103%2Fphysreva.105.052415

Authors: Gonzalo Ferro

"""
import string
import logging
import numpy as np
logger = logging.getLogger('__name__')

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
    # Creation of the numpy arrays of all the tensors needed
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
        'A, AB, BCD, Ci, DjE, Ek -> ijk',
        zeroket, x_rot0, delta_tensor, phase_change_gate, delta_tensor, z_rot0,
    )
    unit1 = np.einsum(
        'iA, ABC, Bj, CkD, Dl -> ijkl',
        x_rot1, delta_tensor, phase_change_gate, delta_tensor, z_rot1
    )
    unit2 = np.einsum(
        'iA, ABC, Bj, CkD, Dl -> ijkl',
        x_rot2, delta_tensor, phase_change_gate, delta_tensor, z_rot2)
    unit = np.einsum('ijA, AklB, Bmno -> ijklmno', unit0, unit1, unit2)
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

    # state_conj : numpy array
    #     numpy array with the complex conjugate of the state of the MPS
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
        #index_segment = '{0}ijklmn, {1}jolpnq -> {0}{1}iokpmq'.format(
        index_segment = 'ijklmn{0}, oipkqm{1} -> ojplqn{0}{1}'.format(
            physical_index1, physical_index2
        )
        state = np.einsum(index_segment, state, unit)
        logger.info(index_segment)
        physical_index1 += physical_index2

    logger.info('iijjkk{0} -> {0}'.format(physical_index1))
    state = np.einsum('iijjkk{0} -> {0}'.format(physical_index1), state)
    #state_conj = np.conj(state)
    return state, physical_index1

