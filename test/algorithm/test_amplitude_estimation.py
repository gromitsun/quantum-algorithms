import unittest
import numpy as np
import qiskit

from algorithm.grover import DiffusionOperator, PhaseOracle, BooleanOracle, GroverIterate
from algorithm.amplitude_estimation import AmplitudeEstimation, counts_to_amplitudes
from algorithm.ae_utils import mle
from utils.qiskit_utils import get_counts


class AmplitudeEstimationTest(unittest.TestCase):

    def _test_amplitude_estimation(self, oracle, num_output_qubits):
        source_state = [0] * oracle.num_state_qubits
        a_op = DiffusionOperator(num_qubits=oracle.num_state_qubits)
        rs_op = PhaseOracle(target_state=source_state, num_control_qubits=1)
        q_op = GroverIterate(a_op=a_op, rs_op=rs_op, oracle=oracle)

        ae = AmplitudeEstimation(a_op, q_op, num_output_qubits=num_output_qubits)

        qc = ae.get_circuit()

        creg = qiskit.ClassicalRegister(num_output_qubits, 'c')
        qc.add_register(creg)

        qc.measure(qc.qregs[1], creg)

        shots = 10000
        res = get_counts(qc, shots)

        a, p = counts_to_amplitudes(res)

        qae = a[np.argmax(p)]
        m = ae.num_output_qubits

        a_opt = mle(qae, a, p, m, shots)

        self.assertAlmostEqual(a_opt, len(oracle.target_states) / 2 ** oracle.num_state_qubits, 3)

    def test_amplitude_estimation_phase_oracle(self):
        target_states = [[0, 0, 0], [0, 1, 1]]
        oracle = PhaseOracle(target_state=target_states, num_control_qubits=1, reverse=True)
        self._test_amplitude_estimation(oracle, num_output_qubits=5)

    def test_amplitude_estimation_boolean_oracle(self):
        target_states = [[0, 0, 0], [0, 1, 1]]
        oracle = BooleanOracle(target_state=target_states, reverse=True).to_phase_oracle(1)
        self._test_amplitude_estimation(oracle, num_output_qubits=5)


if __name__ == '__main__':
    unittest.main()
