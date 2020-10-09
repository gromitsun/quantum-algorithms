import math
import typing
import numpy as np

import qiskit
import qiskit.aqua.circuits

from algorithm.quantum_operator import QuantumOperator, ControlledOperator, SimpleOperator
from utils.common import state_to_sv, BinarySequenceType


##################################################
# Component operators -- general case
##################################################
class InitializeSourceOperator(SimpleOperator):

    def __init__(
            self,
            num_qubits: typing.Optional[int] = None,
            state_vector: typing.Optional[np.array] = None,
            name: typing.Optional[str] = 'InitSource',
    ):
        assert not (num_qubits is None and state_vector is None)
        if num_qubits is None:
            num_qubits = int(round(math.log2(len(state_vector))))

        if state_vector is None:
            state_vector = np.ones(2 ** num_qubits) / 2 ** (num_qubits - 1)

        # Construct source state from all zero state
        _circuit = qiskit.aqua.circuits.StateVectorCircuit(state_vector).construct_circuit()

        super().__init__(_circuit, name=name)


class DiffusionOperator(SimpleOperator):

    def __init__(
            self,
            num_qubits: typing.Optional[int] = None,
            state_vector: typing.Optional[np.array] = None,
            source_state_vector: typing.Optional[np.array] = None,
            name: typing.Optional[str] = 'Diffusion',
    ):
        assert not (num_qubits is None and state_vector is None and source_state_vector is None)

        if num_qubits is None:
            num_qubits = int(round(math.log2(len(state_vector or source_state_vector))))

        if state_vector is None:
            state_vector = np.ones(2 ** num_qubits) / 2 ** (num_qubits - 1)

        if source_state_vector is None:
            source_state_vector = np.zeros(2 ** num_qubits)
            source_state_vector[0] = 1

        assert len(state_vector) == len(source_state_vector) == 2 ** num_qubits

        # Reverse init source op
        _circuit = qiskit.aqua.circuits.StateVectorCircuit(source_state_vector).construct_circuit().inverse()

        # Prepare desired state specified by sv
        _circuit = qiskit.aqua.circuits.StateVectorCircuit(state_vector).construct_circuit(_circuit)

        super().__init__(_circuit, name=name)


class Oracle(QuantumOperator):
    def __init__(
            self,
            target_state: typing.Union[typing.Sequence[int], typing.Sequence[typing.Sequence[int]]],
            name: typing.Optional[str] = 'Oracle',
    ):
        assert len(target_state) > 0
        if isinstance(target_state[0], int):
            target_state = [target_state]

        self._target_states = target_state

        # Check uniqueness of target states
        assert len(set(''.join(str(x) for x in state) for state in self.target_states)) == len(self.target_states)

        self._num_state_qubits = len(self.target_states[0])

        circuit = self._build_circuit(name=name)

        super().__init__(circuit)

    def _build_circuit(self, name=None) -> qiskit.QuantumCircuit:
        raise NotImplementedError("Abstract class")

    @property
    def num_state_qubits(self):
        return self._num_state_qubits

    @property
    def target_states(self):
        return self._target_states


class ControlledOracle(ControlledOperator):
    def __init__(
            self,
            target_state: typing.Union[typing.Sequence[int], typing.Sequence[typing.Sequence[int]]],
            num_control_qubits: int = 1,
            num_ancilla_qubits: int = 0,
            name: typing.Optional[str] = 'Oracle',
    ):
        assert len(target_state) > 0
        if isinstance(target_state[0], int):
            target_state = [target_state]

        self._target_states = target_state

        # Check uniqueness of target states
        assert len(set(''.join(str(x) for x in state) for state in self.target_states)) == len(self.target_states)

        self._num_state_qubits = len(self.target_states[0])
        self._num_control_qubits = num_control_qubits  # for access by _build_circuit

        circuit = self._build_circuit(name=name)

        super().__init__(circuit, num_control_qubits=num_control_qubits, num_ancilla_qubits=num_ancilla_qubits)

    def _build_circuit(self, name=None) -> qiskit.QuantumCircuit:
        raise NotImplementedError("Abstract class")

    @property
    def num_state_qubits(self):
        return self._num_state_qubits

    @property
    def target_states(self):
        return self._target_states


class PhaseOracle(Oracle):
    def __init__(
            self,
            target_state: typing.Union[typing.Sequence[int], typing.Sequence[typing.Sequence[int]]],
            phase: float = math.pi,
            name: typing.Optional[str] = 'PhaseOracle',
    ):
        self._phase = phase
        super().__init__(target_state, name=name)

    @property
    def phase(self):
        return self._phase

    def _build_circuit(self, name=None) -> qiskit.QuantumCircuit:
        state_reg = qiskit.QuantumRegister(self.num_state_qubits)
        circuit = qiskit.QuantumCircuit(state_reg, name=name)

        ref_state = [1] * self.num_state_qubits

        for state in self.target_states:
            for bit, rbit, q in zip(state, ref_state, state_reg):
                if bool(bit) is not bool(rbit):
                    circuit.x(q)
            circuit.mcu1(self.phase, state_reg[:-1], state_reg[-1])
            ref_state = state

        for bit, q in zip(ref_state, state_reg):
            if not bit:
                circuit.x(q)
        return circuit


class ControlledPhaseOracle(ControlledOracle):
    def __init__(
            self,
            target_state: typing.Union[typing.Sequence[int], typing.Sequence[typing.Sequence[int]]],
            num_control_qubits: int = 1,
            phase: float = math.pi,
            reverse: bool = False,
            name: typing.Optional[str] = 'PhaseOracle',
    ):
        self._phase = phase
        self._reverse = reverse
        super().__init__(target_state, num_control_qubits=num_control_qubits, name=name)

    @property
    def phase(self):
        return self._phase

    @property
    def reverse(self):
        return self._reverse

    def _build_circuit(self, name=None) -> qiskit.QuantumCircuit:
        control_reg = qiskit.QuantumRegister(self._num_control_qubits, name='control')
        state_reg = qiskit.QuantumRegister(self.num_state_qubits, name='target')
        circuit = qiskit.QuantumCircuit(control_reg, state_reg, name=name)

        ref_state = [1] * self.num_state_qubits

        for state in self.target_states:
            for bit, rbit, q in zip(state, ref_state, state_reg):
                if bool(bit) is not bool(rbit):
                    circuit.x(q)
            circuit.mcu1(self.phase, control_reg[:] + state_reg[:-1], state_reg[-1])
            ref_state = state

        for bit, q in zip(ref_state, state_reg):
            if not bit:
                circuit.x(q)

        return circuit


class BooleanOracle(Oracle):
    def __init__(
            self,
            target_state: typing.Union[typing.Sequence[int], typing.Sequence[typing.Sequence[int]]],
            reverse: bool = False,
            name: typing.Optional[str] = 'BooleanOracle',
    ):
        self._reverse = reverse
        super().__init__(target_state, name=name)

    @property
    def reverse(self):
        return self._reverse

    def _build_circuit(self, name=None) -> qiskit.QuantumCircuit:
        state_reg = qiskit.QuantumRegister(self.num_state_qubits)
        output_reg = qiskit.QuantumRegister(1)
        circuit = qiskit.QuantumCircuit(state_reg, output_reg, name=name)

        ref_state = [1] * self.num_state_qubits

        for state in self.target_states:
            for bit, rbit, q in zip(state, ref_state, state_reg):
                if bool(bit) is not bool(rbit):
                    circuit.x(q)
            circuit.mcx(state_reg, output_reg)
            ref_state = state

        for bit, q in zip(ref_state, state_reg):
            if not bit:
                circuit.x(q)

        # Reverse output
        if self.reverse:
            circuit.x(output_reg)

        return circuit

    def to_controlled_phase_oracle(self) -> ControlledOracle:
        control_reg = qiskit.QuantumRegister(1)
        state_reg = qiskit.QuantumRegister(self.num_state_qubits)
        output_reg = qiskit.QuantumRegister(1)
        circuit = qiskit.QuantumCircuit(control_reg, state_reg, output_reg, name='Controlled'+self.name)
        circuit.cx(control_reg, output_reg)
        circuit.h(output_reg)
        circuit = self.apply(circuit, state_reg, output_reg)
        circuit.h(output_reg)
        circuit.cx(control_reg, output_reg)
        ret = ControlledOperator(circuit, num_control_qubits=1, num_ancilla_qubits=1)
        ret.__class__ = ControlledOracle
        return ret

    def to_phase_oracle(self) -> Oracle:
        state_reg = qiskit.QuantumRegister(self.num_state_qubits)
        output_reg = qiskit.QuantumRegister(1)
        circuit = qiskit.QuantumCircuit(state_reg, output_reg, name=self.name)
        circuit.x(output_reg)
        circuit.h(output_reg)
        circuit = self.apply(circuit, state_reg, output_reg)
        circuit.h(output_reg)
        circuit.x(output_reg)
        ret = SimpleOperator(circuit, num_ancilla_qubits=1)
        ret.__class__ = Oracle
        return ret


class GroverIterate(SimpleOperator):
    def __init__(
            self,
            a_op: QuantumOperator,
            rs_op: QuantumOperator,
            oracle: PhaseOracle,
            name: typing.Optional[str] = 'GroverIterate',
    ):
        qreg = qiskit.QuantumRegister(oracle.num_qubits)
        circuit = qiskit.QuantumCircuit(qreg, name=name)

        oracle(circuit, qreg)
        a_op.apply_inverse(circuit, qreg)
        rs_op(circuit, qreg)
        a_op(circuit, qreg)

        super().__init__(circuit)


class ControlledGroverIterate(ControlledOperator):
    def __init__(
            self,
            a_op: QuantumOperator,
            rs_op: ControlledOperator,
            oracle: ControlledPhaseOracle,
            name: typing.Optional[str] = 'GroverIterate',
    ):
        control_reg = qiskit.QuantumRegister(oracle.num_control_qubits)
        target_reg = qiskit.QuantumRegister(oracle.num_target_qubits)
        circuit = qiskit.QuantumCircuit(control_reg, target_reg, name=name)

        # flip sign
        if oracle.num_control_qubits == 1:
            circuit.z(control_reg)
        else:
            circuit.mcu1(math.pi, control_reg[:-1], control_reg[-1])

        oracle(circuit, control_reg, target_reg)
        a_op.apply_inverse(circuit, target_reg)
        rs_op(circuit, control_reg, target_reg)
        a_op(circuit, target_reg)

        super().__init__(circuit, num_control_qubits=oracle.num_control_qubits)


##################################################
# Utility functions
##################################################
def optimal_iterations(num_qubits: int, num_targets: int = 1) -> int:
    """
    Optimal number of Grover iterations to get maximum
    probabily of measuring a target state
    """
    return round(0.25 * math.pi / math.asin(math.sqrt(num_targets / 2 ** num_qubits)) - 0.5)


##################################################
# Grover search circuit
##################################################
def grover_circuit(
        target: typing.Union[typing.Sequence[BinarySequenceType], BinarySequenceType],
        source: BinarySequenceType = None,
        niter: typing.Union[int, None] = None,
        measure: bool = True,
) -> qiskit.QuantumCircuit:

    assert len(target) > 0
    # Input is a single target
    if not isinstance(target[0], typing.Sequence):
        target = [target]

    # number of qubits
    num_qubits = len(target[0])
    # number of targets
    num_targets = len(target)

    # initial (source) state to start with
    if source is None:
        source = [0] * num_qubits

    # get optimal number of iterations
    if niter is None:
        niter = optimal_iterations(num_qubits, num_targets=num_targets)

    # quantum circuit
    qreg = qiskit.QuantumRegister(num_qubits, name='q')
    qc = qiskit.QuantumCircuit(qreg, name='grover')

    # initialize
    # construct source
    source_sv = state_to_sv(source)
    init_source_op = InitializeSourceOperator(state_vector=source_sv)
    init_source_op(qc, qreg)
    # construct equal superposition
    a_op = DiffusionOperator(num_qubits=num_qubits, source_state_vector=source_sv)
    a_op(qc, qreg)

    # Grover iterations
    rs_op = PhaseOracle(target_state=source)
    oracle = PhaseOracle(target_state=target)
    grover_op = GroverIterate(a_op=a_op, rs_op=rs_op, oracle=oracle)
    for _ in range(niter):
        grover_op(qc, qreg)

    # measurement
    if measure:
        qc.measure_all()

    return qc
