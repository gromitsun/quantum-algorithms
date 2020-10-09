import typing
import math
import numpy as np

import qiskit

from algorithm.quantum_operator import QuantumOperator, ControlledOperator
from algorithm.qft import qft
from algorithm.grover import DiffusionOperator, ControlledPhaseOracle, ControlledGroverIterate


def state_to_amplitude(state: str):
    m = len(state)
    y = int(state, base=2)
    return math.sin(math.pi*y/2**m)**2


def counts_to_amplitudes(
        counts: typing.Dict[str, float],
        precision: typing.Optional[int] = 5
) -> typing.Tuple[np.array, np.array]:
    estimates = {}
    for state, count in counts.items():
        a = state_to_amplitude(state)
        if precision is not None:
            a = round(a, precision)
        estimates[a] = count + estimates.get(a, 0)

    res = np.array(sorted(estimates.items()))
    return res[:, 0], res[:, 1]


class AmplitudeEstimation(QuantumOperator):
    """
    Amplitude estimation using phase estimation
    """
    def __init__(
            self,
            a_op: QuantumOperator,
            q_op: ControlledOperator,
            num_output_qubits: int,
            name: typing.Optional[str] = 'AmplitudeEstimation',
    ):
        self._num_output_qubits = num_output_qubits
        state_reg = qiskit.QuantumRegister(q_op.num_target_qubits, name='state')
        output_reg = qiskit.QuantumRegister(num_output_qubits, name='output')
        circuit = qiskit.QuantumCircuit(state_reg, output_reg, name=name)

        # reverse qubits
        output_reg = output_reg[::-1]

        # apply A operator
        a_op(circuit, state_reg)

        # apply Hadamard on output register
        circuit.h(output_reg)

        # apply controlled-Q operator
        for k, q in enumerate(output_reg):
            for _ in range(2**k):
                q_op(circuit, q, state_reg)

        # reverse qubits
        output_reg = output_reg[::-1]

        # apply inverse QFT
        qft(circuit, output_reg, do_swaps=False, inverse=True)

        super().__init__(circuit)

    @property
    def num_output_qubits(self):
        return self._num_output_qubits


def test1():
    from algorithm.ae_utils import mle
    from utils.qiskit_utils import get_counts

    num_qubits = 3
    num_output_qubits = 5
    target_states = [[0, 0, 0], [0, 1, 1]]
    source_state = [0] * num_qubits
    a_op = DiffusionOperator(num_qubits=num_qubits)
    rs_op = ControlledPhaseOracle(target_state=source_state)
    oracle = ControlledPhaseOracle(target_state=target_states)
    q_op = ControlledGroverIterate(a_op=a_op, rs_op=rs_op, oracle=oracle)

    ae = AmplitudeEstimation(a_op, q_op, num_output_qubits=num_output_qubits)

    print(ae.draw(fold=-1))

    qc = ae.get_circuit()

    creg = qiskit.ClassicalRegister(num_output_qubits, 'c')
    qc.add_register(creg)

    qc.measure(qc.qregs[1], creg)

    print(qc.draw(fold=-1))

    shots = 10000
    res = get_counts(qc, shots)

    print(res)
    a, p = counts_to_amplitudes(res)
    for _a, _p in zip(a, p / np.sum(p)):
        print("%.5f: %s" % (_a, _p))

    qae = a[np.argmax(p)]
    m = ae.num_output_qubits

    print(mle(qae, a, p, m, shots))


def test2():
    from algorithm.ae_utils import mle
    from utils.qiskit_utils import get_counts, get_statevector

    num_qubits = 3
    num_output_qubits = 5
    target_states = [[0, 0, 0]]
    source_state = [0] * num_qubits
    a_op = DiffusionOperator(num_qubits=num_qubits)
    rs_op = ControlledPhaseOracle(target_state=source_state)
    oracle = ControlledPhaseOracle(target_state=target_states)
    q_op = ControlledGroverIterate(a_op=a_op, rs_op=rs_op, oracle=oracle)

    ae = AmplitudeEstimation(a_op, q_op, num_output_qubits=num_output_qubits)

    qc = qiskit.QuantumCircuit(*q_op.qregs)

    qc.h(q_op.control_qubits)

    a_op(qc, q_op.target_qubits)
    # rs_op(qc, q_op.control_qubits, q_op.target_qubits)
    # oracle(qc, q_op.control_qubits, q_op.target_qubits)
    q_op(qc, q_op.control_qubits, q_op.target_qubits)

    res = get_statevector(qc)

    print(res[::2])
    print(res[1::2])
    # for x in res:
    #     print('  %s' % x)


if __name__ == '__main__':
    test1()
    # import matplotlib.pyplot as plt
    # qiskit.visualization.plot_histogram({value_to_estimation(state): count for state, count in res.items()})
    # plt.tight_layout()
    # plt.show()

