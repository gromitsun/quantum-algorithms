import unittest
import numpy as np
import qiskit

import utils.qiskit_utils
import algorithm.qft as qft


# Do FFT
def fft_by_qft(x, inverse=False, use_classical_swaps=False):
    # QFT = inverse classical DFT
    inverse = not inverse
    # number of qubits
    n = int(np.log2(len(x)))
    assert len(x) == 2**n
    # construct circuit
    qreg = qiskit.QuantumRegister(n)
    qc = qiskit.QuantumCircuit(qreg)
    # initialize circuit with input state vector
    qc.initialize(x, qreg)
    # reverse qubit order classically -- inverse QFT
    if use_classical_swaps and inverse:
        qreg = qreg[::-1]
    # apply QFT
    qft.qft(qc, qreg, do_swaps=not use_classical_swaps, inverse=inverse)
    # execute circuit
    result = utils.qiskit_utils.get_statevector(qc)
    # reverse qubit order classically -- forward QFT
    if use_classical_swaps:
        result = qft.reorder_qft(result)
    return result


def direct_dft(x, inverse=False):
    n = len(x)
    sign = 1 if inverse else -1
    gen = (sum(x[i] * np.exp(sign * 2j * np.pi * k * i / n) / np.sqrt(n) for i in range(n)) for k in range(n))
    return np.fromiter(gen, np.complex, count=n)


# Normalization
def normalize(x):
    x = np.asarray(x)
    return x / np.linalg.norm(x)


class TestQFT(unittest.TestCase):
    def test_qft(self):
        x = [1, 2-1j, -1j, -1+2j]
        x = normalize(x)

        print("Input: %s" % x)

        print("------- Forwrad FFT -------")
        # direct DFT -- own implementation
        dft = direct_dft(x)
        # direct DFT -- numpy
        np_fft = np.fft.fft(x)
        # QFT -- quantum swaps
        qft_fft = fft_by_qft(x)
        # QFT -- classical swaps
        qft_fft_c = fft_by_qft(x, use_classical_swaps=True)

        # Use the same normalization for classical DFT as QFT
        np_fft = np_fft / np.sqrt(len(x))

        np.testing.assert_array_almost_equal(np_fft, dft)
        np.testing.assert_array_almost_equal(qft_fft, dft)
        np.testing.assert_array_almost_equal(qft_fft_c, dft)

        print("------- Inverse FFT -------")
        # direct DFT -- own implementation
        idft = direct_dft(dft, inverse=True)
        # direct DFT -- numpy
        np_ifft = np.fft.ifft(np_fft)
        # QFT -- quantum swaps
        qft_ifft = fft_by_qft(qft_fft, inverse=True)
        # QFT -- classical swaps
        qft_ifft_c = fft_by_qft(qft_fft_c, inverse=True, use_classical_swaps=True)

        # Use the same normalization for classical DFT as QFT
        np_ifft = np_ifft * np.sqrt(len(np_fft))

        np.testing.assert_array_almost_equal(idft, x)
        np.testing.assert_array_almost_equal(np_ifft, x)
        np.testing.assert_array_almost_equal(qft_ifft, x)
        np.testing.assert_array_almost_equal(qft_ifft_c, x)


if __name__ == '__main__':
    unittest.main()
