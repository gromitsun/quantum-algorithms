import math
import typing

import qiskit

from algorithm.qft import qft
from utils.common import int_to_bin


def cphase_operator(scaled_phase: float):
    """
    Return a controlled phase (U1) operator parametrized by
    the number of repeated applications of this operator
    """
    phase = 2 * math.pi * scaled_phase

    def _operator(
            circuit: qiskit.QuantumCircuit,
            n: int,  # number of applications of this operator
            control: qiskit.QuantumRegister,
            target: qiskit.QuantumRegister,
    ) -> None:
        circuit.cu1(phase * n, control, target)

    return _operator


def phase_estimate(
        circuit: qiskit.QuantumCircuit,
        qreg_out: qiskit.QuantumRegister,
        qreg_ancilla: qiskit.QuantumRegister,
        c_op,  # controlled-U operator
        measure: typing.Optional[bool] = True,
        creg: typing.Optional[qiskit.ClassicalRegister] = None,
) -> qiskit.QuantumCircuit:
    """
    Quantum phase estimation
    :param circuit: QuantumCircuit object
    :param qreg_out: output quantum register
    :param qreg_ancilla: target register on which the U operator is applied
    :param c_op: controlled-U operator
    :param measure: (optional) if True, measure the output qubits at the end
    :param creg: (optional) classical register for measurement results
    :return: Quantum circuit
    """
    # initialize output qubits to equal superposition
    list(map(circuit.h, qreg_out))

    # apply controlled-U
    for k, control in enumerate(reversed(qreg_out)):
        c_op(circuit, 2 ** k, control, qreg_ancilla)

    # reverse qubits -- only needed if not using reversed(qreg_out) when applying controlled-U above
    # swap_qubits(qc, qout)

    # apply inverse QFT
    qft(circuit, qreg_out, do_swaps=False, inverse=True)

    # measure
    if measure:
        if creg is None:
            creg = qiskit.ClassicalRegister(len(qreg_out), name='c')
            circuit.add_register(creg)
        for qbit, cbit in zip(qreg_out, creg):
            circuit.measure(qbit, cbit)

    return circuit


##################################################
# Postprocessing utilities
##################################################
def frac_bin_to_decimal(bin_str: str) -> float:
    """
    Convert binary fractional number (str) to decimal
    Most significant bit first (small endian) -- as in qiskit result output
    """
    return sum((bit == '1') and 1 / (2 ** k) for k, bit in enumerate(bin_str, start=1))


def decimal_to_frac_bin(dec: float, n_bits: int) -> str:
    """
    Convert decimal to binary fractional number (str)
    Most significant bit first (small endian) -- as in qiskit result output
    """
    assert 0 <= dec < 1

    res = 0
    for _ in range(n_bits):
        dec *= 2
        bit = int(dec)
        res = (res << 1) | bit
        dec -= bit

    # rounding
    if dec >= 0.5:
        res += 1

    return int_to_bin(res, n_bits=n_bits)
