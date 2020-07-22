"""
Matrix representation of basic gates
"""
import typing
import numpy as np


def rx(theta: typing.SupportsFloat) -> np.matrix:
    return np.matrix([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ])


def ry(theta: typing.SupportsFloat) -> np.matrix:
    return np.matrix([
        [np.cos(theta/2), -np.sin(theta/2)],
        [-np.sin(theta/2), np.cos(theta/2)]
    ])


def rz(theta: typing.SupportsFloat) -> np.matrix:
    return np.matrix([
        [1, 0],
        [0, np.exp(1j*theta)]
    ])


pauli_x = np.matrix([
    [0, 1],
    [1, 0]
])


pauli_y = np.matrix([
    [0, -1j],
    [1j, 0]
])


pauli_z = np.matrix([
    [1, 0],
    [0, -1]
])
