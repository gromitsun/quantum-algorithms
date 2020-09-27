"""
Arithmetic operations in the phase (Fourier) space
"""
import typing
import math
import qiskit


def qft_add(
        qc: qiskit.QuantumCircuit,
        qreg: qiskit.QuantumRegister,
        value: typing.Optional[float] = 1,
):
    n = len(qreg)
    for k, q in enumerate(qreg):
        denom = pow(2, n-k)
        if value % denom == 0:
            # Skip the rotation if the angle is a multiple of 2pi
            continue
        qc.u1(2*math.pi/denom*value, q)

    return qc
