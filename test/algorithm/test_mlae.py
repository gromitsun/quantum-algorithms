import unittest
import numpy as np

from algorithm.grover import DiffusionOperator, BooleanOracle
from algorithm.mlae import AOperator, QOperator, run_mlae, get_log_ml_func, get_ml_estimate


class TestMaximumLikelihoodAmplitudeEstimation(unittest.TestCase):
    def test_mlae(self):
        target_states = [[0, 0, 0], [0, 1, 1]]
        oracle = BooleanOracle(target_state=target_states, reverse=False)
        source_state = [0] * oracle.num_state_qubits
        diffusion_op = DiffusionOperator(num_qubits=oracle.num_state_qubits)
        rs_op = BooleanOracle(target_state=source_state).to_phase_oracle()

        a_op = AOperator(diffusion_op, oracle)
        q_op = QOperator(a_op, rs_op)

        ms = 2 ** np.arange(5)
        shots = 1000
        hits = run_mlae(a_op, q_op, num_q_applications=ms, shots=shots)

        theta_opt = get_ml_estimate(get_log_ml_func(shots, hits, ms))
        a_opt = np.sin(theta_opt) ** 2

        self.assertAlmostEqual(a_opt, len(target_states)/2**oracle.num_state_qubits, 3)


if __name__ == '__main__':
    unittest.main()
