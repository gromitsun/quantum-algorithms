import random
import unittest

import numpy as np
import qiskit

import algorithm.phase_estimation as pe
from utils.common import int_to_bin
from utils.qiskit_utils import get_unitary, get_counts


class PhaseEstimateTest(unittest.TestCase):

    def test_cphase_operator(self):
        phase = 0.1
        op = pe.cphase_operator(phase)

        for n in range(11):
            control = qiskit.QuantumRegister(1)
            target = qiskit.QuantumRegister(1)
            qc = qiskit.QuantumCircuit(control, target)

            op(qc, n, control, target)
            # unitary matrix from circuit
            u_op = get_unitary(qc)
            # expected unitary matrix
            u_exp = np.diag([1, 1, 1, np.exp(2j * np.pi * phase * n)])

            np.testing.assert_array_almost_equal(
                u_op, u_exp,
                err_msg="Unitary matrices for phase = %s and n = %d do not match" % (phase, n)
            )

    def test_phase_estimate(self):
        # input parameters
        n_qubits = 4
        shots = 4096
        phase = 0.3
        op = pe.cphase_operator(phase)

        # quantum circuit
        q_out = qiskit.QuantumRegister(n_qubits, name='qout')
        q_ancilla = qiskit.QuantumRegister(1, name='ancilla')
        qc = qiskit.QuantumCircuit(q_out, q_ancilla, name='phase_estimate')

        # initialize ancilla qubit to eigenstate of the unitary to estimate
        qc.x(q_ancilla)

        # run QPE
        pe.phase_estimate(qc, q_out, q_ancilla, op)

        # run QASM simulator
        result = get_counts(qc, shots=shots)

        # check state with highest count
        state, count = max(result.items(), key=lambda x: x[1])

        # check result
        self.assertEqual(state, pe.decimal_to_frac_bin(phase, n_bits=n_qubits))
        self.assertGreater(count, 0.5 * shots)

    def test_frac_bin_convert_from_bin(self):
        n_bits = 50
        inputs = [random.random() for _ in range(10)]
        outputs = [
            pe.frac_bin_to_decimal(pe.decimal_to_frac_bin(x, n_bits=n_bits))
            for x in inputs
        ]

        np.testing.assert_array_almost_equal(inputs, outputs)

    def test_frac_bin_convert_from_dec(self):
        n_bits = 10
        inputs = [
            int_to_bin(random.getrandbits(n_bits), n_bits=n_bits)
            for _ in range(10)
        ]
        outputs = [
            pe.decimal_to_frac_bin(pe.frac_bin_to_decimal(x), n_bits=n_bits)
            for x in inputs
        ]

        self.assertListEqual(inputs, outputs)


if __name__ == '__main__':
    unittest.main()
