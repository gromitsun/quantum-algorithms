import typing
import qiskit


def get_statevector(circuit: qiskit.QuantumCircuit):
    backend = qiskit.BasicAer.get_backend('statevector_simulator')
    job = qiskit.execute(circuit, backend)
    return job.result().get_statevector()

def get_counts(circuit: qiskit.QuantumCircuit, shots: typing.Union[int, None]=None):
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots=shots)
    return job.result().get_counts()

def get_unitary(circuit: qiskit.QuantumCircuit):
    backend = qiskit.BasicAer.get_backend('unitary_simulator')
    job = qiskit.execute(circuit, backend)
    return job.result().get_unitary()
