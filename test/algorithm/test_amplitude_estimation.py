import unittest
import numpy as np
import qiskit

from algorithm.grover import DiffusionOperator, ControlledPhaseOracle, ControlledGroverIterate
from algorithm.amplitude_estimation import AmplitudeEstimation, counts_to_amplitudes
from algorithm.ae_utils import mle
from utils.qiskit_utils import get_counts


class AmplitudeEstimationTest(unittest.TestCase):
    def test_amplitude_estimation(self):
        num_qubits = 3
        num_output_qubits = 5
        target_states = [[0, 0, 0], [0, 1, 1]]
        source_state = [0] * num_qubits
        a_op = DiffusionOperator(num_qubits=num_qubits)
        rs_op = ControlledPhaseOracle(target_state=source_state)
        oracle = ControlledPhaseOracle(target_state=target_states)
        q_op = ControlledGroverIterate(a_op=a_op, rs_op=rs_op, oracle=oracle)

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
        self.assertAlmostEqual(a_opt, len(target_states)/2**num_qubits, 3)


if __name__ == '__main__':
    unittest.main()
