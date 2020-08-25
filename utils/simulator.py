import typing
import numpy as np
import qiskit

from utils.common import int_to_bin


def get_statevector(circuit: qiskit.QuantumCircuit) -> np.ndarray:
    backend = qiskit.Aer.get_backend('statevector_simulator')
    job = qiskit.execute(circuit, backend)
    return job.result().get_statevector()


def get_counts(circuit: qiskit.QuantumCircuit, shots: typing.Union[int, None] = None) -> dict:
    backend = qiskit.Aer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots=shots)
    return job.result().get_counts()


def sv_to_prob(sv: np.ndarray) -> dict:
    return {int_to_bin(i): v * v.conjugate() for i, v in enumerate(sv)}


def get_probabilities(circuit: qiskit.QuantumCircuit) -> dict:
    sv = get_statevector(circuit)
    return sv_to_prob(sv)


def get_unitary(circuit: qiskit.QuantumCircuit) -> np.ndarray:
    backend = qiskit.Aer.get_backend('unitary_simulator')
    job = qiskit.execute(circuit, backend)
    return job.result().get_unitary()
