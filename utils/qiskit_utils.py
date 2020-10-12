import typing
import itertools

import numpy as np
import qiskit

from qiskit.circuit.register import Register
from qiskit.circuit.bit import Bit

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
RegisterType = typing.Union[Register, typing.Sequence[Bit]]
QuantumRegisterType = typing.Union[qiskit.QuantumRegister, typing.Sequence[qiskit.circuit.Qubit]]
ClassicalRegisterType = typing.Union[qiskit.ClassicalRegister, typing.Sequence[qiskit.circuit.Clbit]]

assert issubclass(qiskit.QuantumRegister, Register)
assert issubclass(qiskit.circuit.Qubit, Bit)
assert issubclass(qiskit.ClassicalRegister, Register)
assert issubclass(qiskit.circuit.Clbit, Bit)


def get_unique_registers(*registers: RegisterType):
    _registers = []

    for reg in registers:
        if isinstance(reg, qiskit.QuantumRegister):
            if reg not in _registers:
                _registers.append(reg)
        elif isinstance(reg, typing.Sequence):
            for bit in reg:
                if not isinstance(bit, qiskit.circuit.Qubit):
                    raise TypeError("Expect Qubit type, got %s", type(bit))
                if bit.register not in _registers:
                    _registers.append(bit.register)
        else:
            raise TypeError("Expected QuantumRegister or sequence of Qubits, got %s", type(reg))

    return _registers


def create_circuit(*registers: RegisterType, name: typing.Optional[str] = None) -> qiskit.QuantumCircuit:

    _registers = get_unique_registers(*registers)

    return qiskit.QuantumCircuit(*_registers, name=name)


def create_register(
        num_qubits: int,
        name: typing.Optional[str] = None,
        reg_type: typing.Union[str, type] = 'quantum'
) -> QuantumRegisterType:
    if num_qubits <= 0:
        return []

    if isinstance(reg_type, str):
        if reg_type == 'quantum':
            reg_type = qiskit.QuantumRegister
        elif reg_type == 'classical':
            reg_type = qiskit.ClassicalRegister
        elif reg_type == 'ancilla':
            reg_type = qiskit.AncillaRegister

    assert isinstance(reg_type, type)

    return reg_type(num_qubits, name=name)


def add_registers_to_circuit(circuit, *registers: RegisterType) -> qiskit.QuantumCircuit:
    _registers = get_unique_registers(*registers)

    for reg in _registers:
        if not circuit.has_register(reg):
            circuit.add_register(reg)

    return circuit


class QubitIterator(object):
    def __init__(self, *registers: RegisterType):
        self._registers = registers
        self._bits = itertools.chain(*registers)

    @property
    def registers(self):
        return self._registers

    def get(self, n: typing.Optional[int] = None):
        if n is None:
            # return a list of all remaining qubits
            return list(self._bits)
        # return a list of n qubits
        return [next(self._bits) for _ in range(n)]


def split_register(registers: typing.Sequence[RegisterType], sizes: typing.Sequence[int], keep_extra: bool = False):
    bits = QubitIterator(*registers)
    segments = [bits.get(size) for size in sizes]
    if keep_extra:
        try:
            segments.append(bits.get())
        except StopIteration:
            pass
    return segments
