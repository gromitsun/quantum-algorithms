"""
Matrix representation of basic gates
"""
import typing

import numpy as np

######################################
# Pauli matrices
######################################
pauli_x = np.array([
    [0, 1],
    [1, 0]
])

pauli_y = np.array([
    [0, -1j],
    [1j, 0]
])

pauli_z = np.array([
    [1, 0],
    [0, -1]
])

ident = np.array([
    [1, 0],
    [0, 1]
])


def ident_n(n: int) -> np.array:
    """
    n-qubit identity matrix
    """
    return np.identity(2 ** n)


######################################
# Single-qubit rotations
######################################
def rx(theta: typing.SupportsFloat) -> np.array:
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ])


def ry(theta: typing.SupportsFloat) -> np.array:
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [-np.sin(theta / 2), np.cos(theta / 2)]
    ])


def rz(theta: typing.SupportsFloat) -> np.array:
    return np.array([
        [1, 0],
        [0, np.exp(1j * theta)]
    ])


######################################
# Derived gates / matrix operations
######################################
def kron_power(m: np.array, n: int) -> np.array:
    """
    :param m: matrix
    :param n: positive integer
    :return: Kronecker product of n copies of m, i.e.
        m x m x ... x m
    """
    if n == 1:
        return m
    # divide and conquer
    # divide
    prod = kron_power(m, n // 2)
    # combine
    prod = np.kron(prod, prod)
    # handle case when n is odd
    if n % 2 == 1:
        prod = np.kron(m, prod)
    return prod


def mcu(u: np.array, n: int, i: int = -1) -> np.array:
    """
    Multi-controlled-U operator
    :param u: single-qubit (2x2) unitary matrix
    :param n: total number of qubits
    :param i: (optional) target qubit (0-based index)
    :return: matrix representation of multi-controlled-U operator
        targeted on the i-th qubit
    """
    # handle negative i
    if i < 0:
        i += n
    # statics
    a = (ident + pauli_z) / 2
    b = (ident - pauli_z) / 2

    # solve the problem recursively
    def _mcu_recursive(u, n, i):
        # trivial case
        if n == 1:
            return u
        # general case
        if i > 0:
            return np.kron(a, ident_n(n - 1)) \
                   + np.kron(b, _mcu_recursive(u, n - 1, i - 1))
        return np.kron(ident_n(n - 1), a) \
               + np.kron(_mcu_recursive(u, n - 1, i), b)

    return _mcu_recursive(u, n, i)


def exp_real(u: np.array, t: typing.SupportsFloat) -> np.array:
    """
    :param u: unitary matrix
    :param t: time -- real number
    :return: exp(i*u*t)
    """
    return np.cos(t) * ident_n(u.shape[0]) + 1j * np.sin(t) * u


def exp_imag(u: np.array, t: typing.SupportsFloat) -> np.array:
    """
    :param u: unitary matrix
    :param t: time -- real number
    :return: exp(u*t)
    """
    return np.cosh(t) * ident_n(u.shape[0]) + np.sinh(t) * u


def partial_trace(
        m: np.array,
        i: typing.Union[int, typing.Sequence[int]]
) -> typing.Union[typing.SupportsComplex, np.array]:
    """
    Take partial trace of a density matrix on the i-th qubits (0-based indices)
    """
    if not isinstance(i, int):
        return partial_trace(partial_trace(m, i[1:]), i[0])

    n_state = m.shape[0]
    n_qubit = np.log2(n_state)
    n_pre = 2 ** i
    n_post = 2 ** (n_qubit - i - 1)
    return np.trace(
        m.reshape(n_pre, 2, n_post, n_pre, 2, n_post),
        axis1=1, axis2=4
    ).reshape(n_state / 2, n_state / 2)
