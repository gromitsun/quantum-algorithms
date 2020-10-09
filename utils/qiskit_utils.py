import typing
import itertools

import numpy as np
import qiskit

from utils.common import int_to_bin


######################################
# Simulators
######################################
def get_statevector(circuit: qiskit.QuantumCircuit) -> np.ndarray:
    backend = qiskit.Aer.get_backend('statevector_simulator')
    job = qiskit.execute(circuit, backend)
    return job.result().get_statevector()


def get_counts(circuit: qiskit.QuantumCircuit, shots: typing.Union[int, None] = None) -> dict:
    backend = qiskit.Aer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots=shots)
    return job.result().get_counts()


def vec_to_dict(vec: typing.Sequence, drop_zeros: bool = False) -> dict:
    n = int(np.log2(len(vec)))
    return {int_to_bin(i, n_bits=n): x for i, x in enumerate(vec) if not drop_zeros or not np.isclose(x, 0)}


def sv_to_prob(sv: np.ndarray, drop_zeros: bool = False, fmt: str = 'dict') -> typing.Union[dict, np.array]:
    prob = (sv * sv.conj()).real
    if fmt == 'dict':
        prob = vec_to_dict(prob, drop_zeros=drop_zeros)
    return prob


def get_probabilities(
        circuit: qiskit.QuantumCircuit,
        drop_zeros: bool = True,
        fmt: str = 'dict'
) -> typing.Union[dict, np.array]:
    sv = get_statevector(circuit)
    return sv_to_prob(sv, drop_zeros=drop_zeros, fmt=fmt)


def get_unitary(circuit: qiskit.QuantumCircuit) -> np.ndarray:
    backend = qiskit.Aer.get_backend('unitary_simulator')
    job = qiskit.execute(circuit, backend)
    return job.result().get_unitary()


######################################
# Gate operations
######################################
def get_inverse(op):
    """
    Get the inverse operator
    :param op: a function with a prototype op(circuit, *qregs, **kwargs)
    :return: a function with the same prototype but applying the inverse operation
    """
    def inv(qc, *qregs, **kwargs):
        # Create a temporary circuit to compute the inverse
        _qc = qiskit.QuantumCircuit(*qregs)
        # Apply op on the temporary circuit
        op(_qc, *qregs, **kwargs)
        # Inverse the temporary circuit
        # and append the instructions to the original circuit
        for inst, qargs, cargs in reversed(_qc.data):
            qc.append(inst.inverse(), qargs, cargs)
        return qc

    return inv


######################################
# Miscellaneous
######################################
QuantumRegisterType = typing.Union[qiskit.QuantumRegister, typing.Sequence[qiskit.circuit.Qubit]]


def create_circuit(*qregs: QuantumRegisterType):
    _qregs = []

    for qreg in qregs:
        if isinstance(qreg, qiskit.QuantumRegister):
            if qreg not in _qregs:
                _qregs.append(qreg)
        elif isinstance(qreg, typing.Sequence):
            for q in qreg:
                if not isinstance(q, qiskit.circuit.Qubit):
                    raise TypeError("Expect Qubit type, got %s", type(q))
                if q.register not in _qregs:
                    _qregs.append(q.register)
        else:
            raise TypeError("Expected QuantumRegister or sequence of Qubits, got %s", type(qreg))

    return qiskit.QuantumCircuit(*_qregs)


class QubitIterator(object):
    def __init__(self, *qregs: QuantumRegisterType):
        self._qregs = qregs
        self._qubits = itertools.chain(*qregs)

    @property
    def qregs(self):
        return self._qregs

    def get(self, n: typing.Optional[int] = None):
        if n is None:
            # return a list of all qubits
            return list(self._qubits)
        # return a list of n qubits
        return [next(self._qubits) for _ in range(n)]
