import typing
import numpy as np
import qiskit


def swap_qubits(circuit: qiskit.QuantumCircuit, qreg) -> None:
    """
    Reverse the order of qubits using swap gates
    Note: qiskit qubit order is most significant last (i.e. q[n-1])
    """
    for i in range(len(qreg)//2):
        circuit.swap(qreg[i], qreg[-i-1])


def qft(
    circuit: qiskit.QuantumCircuit,
    qreg, 
    do_swaps: bool = False,
    inverse: bool = False
) -> None:
    """Circuit for QFT. Assumes most significant bit last."""
    
    if inverse:
        sign = -1
    else:
        qreg = qreg[::-1]
        sign = 1
        
    # swap qubits -- inverse QFT
    if do_swaps and inverse:
        swap_qubits(circuit, qreg)
    
    # do QFT
    for i, target in enumerate(qreg):
        circuit.h(target)
        for k, control in enumerate(qreg[i+1:]):
            circuit.cu1(sign*np.pi/2**(k+1), control, target)
            
    # swap qubits -- forward QFT
    if do_swaps and not inverse:
        swap_qubits(circuit, qreg)


def qiskit_qft(
    circuit: qiskit.QuantumCircuit,
    qreg,
    do_swaps: bool = False,
    inverse: bool = False
) -> None:
    """Qiskit circuit library for QFT"""
    # Qiskit uses most significant bit last convention
    # However, the QFT in the circuit library uses most significant bit first
    # therefore we need to swap the qubits
    inst = qiskit.circuit.library.QFT(num_qubits=len(qreg), do_swaps=do_swaps, inverse=inverse)
    circuit.append(inst, qreg[::-1])


# QFT reorder
def get_basis_states(n_qubits: int) -> typing.List[str]:
    return ['{{:0{:d}b}}'.format(n_qubits).format(x) for x in range(2**n_qubits)]

def qft_get_reorder_idx(n_qubits: int) -> typing.List[int]:
    """Get indices for reordering QFT basis states"""
    return [int(state[::-1], 2) for state in get_basis_states(n_qubits)]

def qft_get_reorder_states(n_qubits: int) -> typing.List[str]:
    """Get reordered basis states in QFT outputs"""
    basis_states = get_basis_states(n_qubits)
    return [basis_states[int(state[::-1], 2)] for state in basis_states]

def reorder_qft(a, axis: int=0) -> np.ndarray:
    """Reorder QFT outputs by reversing qubits"""
    n_qubits = int(np.round(np.log2(len(a))))
    return np.asarray(a).take(qft_get_reorder_idx(n_qubits), axis=axis)
