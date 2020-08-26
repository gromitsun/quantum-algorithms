import unittest
import random
import math
import numpy as np

import algorithm.grover as grover
from utils.qiskit_utils import get_statevector
from utils.common import int_to_bin


class GroverTest(unittest.TestCase):

    def test_grover_search(self):
        # good state to search for (big-endian)
        good_state = [0, 1, 0, 1, 0]

        # construct the circuit
        qc = grover.grover_circuit(target=good_state, measure=False)
        # run the circuit
        res = get_statevector(qc)

        # evaluate results
        # good state binary string (small-endian)
        good_state_str = ''.join(str(x) for x in good_state[::-1])
        n_qubits = len(good_state)
        # get maximum probability state and amplitude
        max_prob_state_i = np.argmax(np.abs(res))
        max_prob_state = int_to_bin(max_prob_state_i, n_bits=n_qubits)
        max_amplitude = res[max_prob_state_i]

        # check state
        self.assertEqual(max_prob_state, good_state_str)

        # check amplitude
        n_iters = grover.optimal_iterations(n_qubits=n_qubits)
        theta = math.asin(1/math.sqrt(2**n_qubits))
        self.assertAlmostEqual(abs(max_amplitude), math.sin((2*n_iters+1)*theta))

    def test_grover_search_random(self):
        n_qubits = random.randint(2, 8)
        # good state binary string (small-endian)
        good_state_int = random.randint(0, 2**n_qubits-1)
        good_state_str = int_to_bin(good_state_int, n_bits=n_qubits)
        # good state to search for (big-endian)
        good_state = [int(x) for x in good_state_str[::-1]]

        # construct the circuit
        qc = grover.grover_circuit(target=good_state, measure=False)
        # run the circuit
        res = get_statevector(qc)

        # evaluate results
        # get maximum probability state and amplitude
        max_prob_state_i = np.argmax(np.abs(res))
        max_prob_state = int_to_bin(max_prob_state_i, n_bits=n_qubits)
        max_amplitude = res[max_prob_state_i]

        msg = "n_qubits = %d, good_state = %s" % (n_qubits, good_state_str)

        # check state
        self.assertEqual(max_prob_state_i, good_state_int, msg='Most probable state does not match input: ' + msg)
        self.assertEqual(max_prob_state, good_state_str, msg='Most probable state does not match input: ' + msg)

        # check amplitude
        n_iters = grover.optimal_iterations(n_qubits=n_qubits)
        theta = math.asin(1/math.sqrt(2**n_qubits))
        self.assertAlmostEqual(
            abs(max_amplitude), math.sin((2*n_iters+1)*theta),
            msg='Maximum amplitude does not match expected: ' + msg
        )


if __name__ == '__main__':
    unittest.main()
