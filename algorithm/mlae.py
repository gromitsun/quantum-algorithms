import typing
import numpy as np
import scipy.optimize

import qiskit

from algorithm.quantum_operator import QuantumOperator, create_ancillas_for
from algorithm.grover import BooleanOracle
from utils.qiskit_utils import create_register, create_circuit, get_counts


class AOperator(QuantumOperator):
    def __init__(
            self,
            state_loading_op: QuantumOperator,
            oracle: BooleanOracle,
            name: typing.Optional[str] = 'A',
    ):
        super().__init__(name=name)
        self._state_loading_op = state_loading_op
        self._oracle = oracle

    @property
    def oracle(self):
        return self._oracle

    @property
    def state_loading_op(self):
        return self._state_loading_op

    def _build_internal_circuit(self) -> qiskit.QuantumCircuit:
        state_reg = create_register(self.oracle.get_register('state').size, name='state')
        output_reg = create_register(1, name='output')
        ancilla_reg = create_ancillas_for(self.state_loading_op, self.oracle)
        circuit = create_circuit(state_reg, ancilla_reg, name=self.name)
        self.state_loading_op(circuit, state=state_reg, ancilla=ancilla_reg)
        self.oracle(circuit, state=state_reg, output=output_reg, ancilla=ancilla_reg)

        return circuit


class QOperator(QuantumOperator):
    def __init__(
            self,
            a_op: QuantumOperator,
            rs_op: QuantumOperator,
            name: typing.Optional[str] = 'Q',
    ):
        super().__init__(name=name)
        self._a_op = a_op
        self._rs_op = rs_op

    @property
    def a_op(self):
        return self._a_op

    @property
    def rs_op(self):
        return self._rs_op

    def _build_internal_circuit(self) -> qiskit.QuantumCircuit:
        state_reg = create_register(self.rs_op.get_register('state').size, name='state')
        output_reg = create_register(1, name='output')
        ancilla_reg = create_ancillas_for(self.a_op, self.rs_op)
        circuit = create_circuit(state_reg, output_reg, ancilla_reg, name=self.name)

        # Add pi phase to states with output qubit = |0>
        circuit.x(output_reg)
        circuit.u1(np.pi, output_reg)
        circuit.x(output_reg)

        # A^dag
        self.a_op.apply_inverse(circuit)
        # Reflection on source state
        self.rs_op(circuit)
        # A
        self.a_op(circuit)

        return circuit


class MLAE(QuantumOperator):
    """
    Maximum Likelihood Amplitude Estimation (MLAE)
    """

    def __init__(
            self,
            a_op: QuantumOperator,
            q_op: QuantumOperator,
            num_q_applications: int,
            name: typing.Optional[str] = 'MLAE',
    ):
        super().__init__(name=name)
        self._a_op = a_op
        self._q_op = q_op
        self._num_q_applications = num_q_applications

    def _build_internal_circuit(self) -> qiskit.QuantumCircuit:
        state_reg = create_register(self._q_op.get_register('state').size, name='state')
        output_reg = create_register(1, name='output')
        ancilla_reg = create_ancillas_for(self._a_op, self._q_op, name='ancilla')
        circuit = create_circuit(state_reg, output_reg, ancilla_reg, name=self.name)

        # apply A operator
        self._a_op(circuit, state=state_reg, output=output_reg, ancilla=ancilla_reg)

        # apply Q operator
        for _ in range(self.num_q_applications):
            self._q_op(circuit, state=state_reg, output=output_reg, ancilla=ancilla_reg)

        return circuit

    @property
    def num_q_applications(self):
        return self._num_q_applications

    @classmethod
    def from_grover_operators(
            cls,
            diffusion_op: QuantumOperator,
            rs_op: QuantumOperator,
            oracle: BooleanOracle,
            *args,
            **kwargs,
    ):
        a_op = AOperator(diffusion_op, oracle)
        q_op = QOperator(a_op, rs_op)
        return cls(a_op, q_op, *args, **kwargs)


def run_mlae(
        a_op: QuantumOperator,
        q_op: QuantumOperator,
        num_q_applications: typing.Union[int, typing.Sequence[int]],
        shots: typing.Union[int, typing.Sequence[int]],
) -> typing.Union[int, np.array]:
    if not isinstance(num_q_applications, int) or not isinstance(shots, int):
        return np.array([run_mlae(a_op, q_op, int(n), int(s)) for n, s in np.broadcast(num_q_applications, shots)])

    ae = MLAE(a_op, q_op, num_q_applications=num_q_applications)

    qc = ae.get_circuit()

    creg = qiskit.ClassicalRegister(1, 'c')
    qc.add_register(creg)

    qc.measure(ae.get_register('output'), creg)
    res = get_counts(qc, shots)

    return res.get('1', 0)


def get_ml_func(n, h, m):
    n = np.asarray(n)
    h = np.asarray(h)
    m = np.asarray(m)

    def ml_func(theta):
        theta = np.asarray(theta).reshape(-1, 1)
        return np.prod(
            np.abs(np.sin((2 * m + 1) * theta)) ** (2 * h) * np.abs(np.cos((2 * m + 1) * theta)) ** (2 * (n - h)),
            axis=-1)

    return ml_func


def get_log_ml_func(n, h, m, llim=1e-15):
    n = np.asarray(n)
    h = np.asarray(h)
    m = np.asarray(m)

    def ml_func(theta):
        theta = np.asarray(theta).reshape(-1, 1)
        return np.sum(
            2 * h * np.log(np.clip(np.abs(np.sin((2 * m + 1) * theta)), llim, None))
            + 2 * (n - h) * np.log(np.clip(np.abs(np.cos((2 * m + 1) * theta)), llim, None)),
            axis=-1)

    return ml_func


def get_ml_estimate(ml_func, theta=None):
    if theta is None:
        theta = np.linspace(0, np.pi / 2, 100)
    theta_max = theta[np.argmax(ml_func(theta))]
    theta_max_opt = scipy.optimize.fmin(lambda x: -ml_func(x), theta_max)[0]
    return theta_max_opt
