import unittest

import numpy as np
import qiskit

import arithmetic.qft_arithmetic as aq
from algorithm.qft import qft
from utils.qiskit_utils import get_statevector


class TestQFTArithmetic(unittest.TestCase):
    def test_qft_add(self):
        value = 2
        n = 5
        qreg = qiskit.QuantumRegister(n)
        qc = qiskit.QuantumCircuit(qreg)
        qft(qc, qreg, do_swaps=False, classical_input=[0] * n)
        aq.qft_add(qc, qreg[::-1], value)
        qft(qc, qreg, do_swaps=False, inverse=True)

        sv = get_statevector(qc)
        expected_sv = np.zeros(2 ** n)
        expected_sv[value] = 1
        np.testing.assert_array_almost_equal(sv, expected_sv)

    def test_qft_add_fraction(self):
        value = 2.5
        n = 5
        qreg = qiskit.QuantumRegister(n)
        qc = qiskit.QuantumCircuit(qreg)
        qft(qc, qreg, do_swaps=False, classical_input=[0] * n)
        aq.qft_add(qc, qreg[::-1], value)
        qft(qc, qreg, do_swaps=False, inverse=True)

        sv = get_statevector(qc)
        prob = (sv * sv.conj()).real

        shift = 2 ** (n - 1) - int(value)
        x_values = np.roll(np.arange(2 ** n) - shift, -shift)
        mean = prob @ x_values
        self.assertAlmostEqual(mean, value, 1)


if __name__ == '__main__':
    unittest.main()
