import math
import typing

import qiskit


##################################################
# Component operators -- one target
##################################################
def init_source_op(circuit: qiskit.QuantumCircuit, qreg, source):
    """Construct the source state from all 0's state"""
    for qbit, sbit in zip(qreg, source):
        if sbit:
            circuit.x(qbit)


def a_op(circuit: qiskit.QuantumCircuit, qreg, source, inv: bool = False):
    """Construct all equal superposition from initial (source) state"""
    for qbit, sbit in zip(qreg, source):
        circuit.rx((-1) ** (inv + sbit) * math.pi / 2, qbit)


def rs_op(circuit: qiskit.QuantumCircuit, qreg):
    """Reflection about initial (source) state"""
    # multi-controlled-Z
    circuit.mcu1(math.pi, qreg[:-1], qreg[-1])


def rt_op(circuit: qiskit.QuantumCircuit, qreg, target):
    """Reflection about target state (t)"""
    # flip qubits corresponding to a zero target bit
    for qbit, tbit in zip(qreg, target):
        if not tbit:
            circuit.x(qbit)
    # multi-controlled-Z
    circuit.mcu1(math.pi, qreg[:-1], qreg[-1])
    # flip qubits corresponding to a zero target bit
    for qbit, tbit in zip(qreg, target):
        if not tbit:
            circuit.x(qbit)


##################################################
# Grover iterate operator
##################################################
def grover_op(circuit: qiskit.QuantumCircuit, qreg, source, target):
    """Operator for one Grover iteration"""
    rt_op(circuit, qreg, target)
    a_op(circuit, qreg, source, inv=False)
    rs_op(circuit, qreg)
    a_op(circuit, qreg, source, inv=True)


##################################################
# Utility functions
##################################################
def optimal_iterations(n_qubits: int, n_targets: int = 1) -> int:
    """
    Optimal number of Grover iterations to get maximum
    probabily of measuring a target state
    """
    return round(0.25 * math.pi / math.asin(math.sqrt(n_targets / 2 ** n_qubits)) - 0.5)


##################################################
# Grover search circuit
##################################################
def grover_circuit(
        target: typing.Union[list, tuple],
        source: typing.Union[list, tuple] = None,
        niter: typing.Union[int, None] = None,
        measure: bool = True,
) -> qiskit.QuantumCircuit:
    # number of qubits
    n = len(target)

    # initial (source) state to start with
    if source is None:
        source = [0] * n

    # get optimal number of iterations
    if niter is None:
        niter = optimal_iterations(n)

    # quantum circuit
    qreg = qiskit.QuantumRegister(n, name='q')
    qc = qiskit.QuantumCircuit(qreg, name='grover')

    # initialize
    # construct source
    init_source_op(qc, qreg, source)
    # construct equal superposition
    a_op(qc, qreg, source)

    # Grover iterations
    for _ in range(niter):
        grover_op(qc, qreg, source, target)

    # measurement
    if measure:
        qc.measure_all()

    return qc
