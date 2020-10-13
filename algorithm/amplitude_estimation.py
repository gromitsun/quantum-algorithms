import typing
import math
import numpy as np

import qiskit

from algorithm.quantum_operator import QuantumOperator, ControlledOperator
from algorithm.qft import qft
from utils.qiskit_utils import create_register, create_circuit


def state_to_estimate(state: str):
    """
    Convert a state to an estimate of the squared amplitude
    :param state: binary str
    :return: estiamte of the amplitude squared (i.e. a^2)
    """
    m = len(state)
    y = int(state, base=2)
    return math.sin(math.pi*y/2**m)**2


def counts_to_estimates(
        counts: typing.Dict[str, float],
        precision: typing.Optional[int] = 5
) -> typing.Tuple[np.array, np.array]:
    """
    Convert state counts data to estimated squared amplitudes
    :param counts: dict {state: count/probability}
    :param precision: precision used in rounding converted estimates
    :return: arrays of estimated values and corresponding counts
             sorted in ascending order of values
    """
    estimates = {}
    for state, count in counts.items():
        a = state_to_estimate(state)
        if precision is not None:
            a = round(a, precision)
        estimates[a] = count + estimates.get(a, 0)

    res = np.array(sorted(estimates.items()))
    # return arrays of estimated values and corresponding counts
    # sorted in ascending order of values
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
        super().__init__(name=name)
        self._a_op = a_op
        self._q_op = q_op
        self._num_output_qubits = num_output_qubits

    def _build_internal_circuit(self) -> qiskit.QuantumCircuit:
        state_reg = create_register(self._q_op.num_target_qubits, name='state')
        output_reg = create_register(self.num_output_qubits, name='output')
        ancilla_reg = create_register(
            max(self._a_op.num_ancillas, self._q_op.num_ancillas),
            name='ancilla', reg_type='ancilla'
        )
        circuit = create_circuit(state_reg, output_reg, ancilla_reg, name=self.name)

        # reverse qubits
        output_reg = output_reg[::-1]

        # apply A operator
        self._a_op(circuit, state_reg, ancilla_reg)

        # apply Hadamard on output register
        circuit.h(output_reg)

        # apply controlled-Q operator
        for k, q in enumerate(output_reg):
            for _ in range(2**k):
                self._q_op(circuit, q, state_reg, ancilla_reg)

        # reverse qubits
        output_reg = output_reg[::-1]

        # apply inverse QFT
        qft(circuit, output_reg, do_swaps=False, inverse=True)

        self._set_internal_circuit(circuit)

        return circuit

    @property
    def num_output_qubits(self):
        return self._num_output_qubits
