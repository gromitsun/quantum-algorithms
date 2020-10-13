import math
import typing
from abc import ABC

import numpy as np

import qiskit
import qiskit.aqua.circuits

from algorithm.quantum_operator import QuantumOperator, ControlledOperator
from utils.common import state_to_sv, get_basis_states, BinarySequenceType
from utils.qiskit_utils import create_circuit, create_register, add_registers_to_circuit


##################################################
# Component operators -- general case
##################################################
class InitializeSourceOperator(QuantumOperator):

    def __init__(
            self,
            num_qubits: typing.Optional[int] = None,
            state_vector: typing.Optional[np.array] = None,
            name: typing.Optional[str] = 'InitSource',
    ):

        super().__init__(name=name)
        assert not (num_qubits is None and state_vector is None)
        if num_qubits is None:
            num_qubits = int(round(math.log2(len(state_vector))))

        if state_vector is None:
            state_vector = np.ones(2 ** num_qubits) / 2 ** (num_qubits - 1)

        self._state_vector = state_vector

    @property
    def state_vector(self):
        return self._state_vector

    def _build_internal_circuit(self) -> qiskit.QuantumCircuit:
        # Construct source state from all zero state
        circuit = qiskit.aqua.circuits.StateVectorCircuit(self.state_vector).construct_circuit()

        self._set_internal_circuit(circuit)

        return circuit


class DiffusionOperator(QuantumOperator):
    """
    Construct state_vector from source_state_vector
    """

    def __init__(
            self,
            num_qubits: typing.Optional[int] = None,
            state_vector: typing.Optional[np.array] = None,
            source_state_vector: typing.Optional[np.array] = None,
            name: typing.Optional[str] = 'Diffusion',
    ):

        assert not (num_qubits is None and state_vector is None and source_state_vector is None)

        if num_qubits is None:
            num_qubits = int(round(math.log2(len(state_vector if state_vector is not None else source_state_vector))))

        if state_vector is None:
            state_vector = np.ones(2 ** num_qubits) / 2 ** (num_qubits - 1)

        if source_state_vector is None:
            source_state_vector = np.zeros(2 ** num_qubits)
            source_state_vector[0] = 1

        assert len(state_vector) == len(source_state_vector) == 2 ** num_qubits

        super().__init__(name=name)

        self._state_vector = state_vector
        self._source_state_vector = source_state_vector

    @property
    def state_vector(self):
        return self._state_vector

    @property
    def source_state_vector(self):
        return self._source_state_vector

    def _build_internal_circuit(self) -> qiskit.QuantumCircuit:
        num_qubits = int(round(math.log2(len(self.state_vector))))
        qreg = qiskit.QuantumRegister(num_qubits, name='state')
        circuit = qiskit.QuantumCircuit(qreg, name=self.name)

        # Reverse init source op
        circuit = qiskit.aqua.circuits.StateVectorCircuit(
            self.source_state_vector
        ).construct_circuit(circuit, qreg).inverse()

        # Prepare desired state specified by sv
        circuit = qiskit.aqua.circuits.StateVectorCircuit(
            self.state_vector
        ).construct_circuit(circuit, qreg)

        self._set_internal_circuit(circuit)

        return circuit


class Oracle(ControlledOperator, ABC):
    def __init__(
            self,
            target_state: typing.Union[typing.Sequence[int], typing.Sequence[typing.Sequence[int]]],
            num_control_qubits: int = 0,
            reverse: bool = False,
            name: typing.Optional[str] = 'Oracle',
    ):
        self._reverse = reverse

        assert len(target_state) > 0
        if isinstance(target_state[0], int):
            target_state = [target_state]

        self._target_states = target_state

        # Check uniqueness of target states
        assert len(set(''.join(str(x) for x in state) for state in self.target_states)) == len(self.target_states)

        self._num_state_qubits = len(self.target_states[0])

        super().__init__(
            num_control_qubits=num_control_qubits,
            name=name,
        )

    @property
    def reverse(self):
        return self._reverse

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
            num_control_qubits: int = 0,
            phase: float = math.pi,
            reverse: bool = False,
            name: typing.Optional[str] = 'PhaseOracle',
    ):
        super().__init__(target_state, num_control_qubits=num_control_qubits, reverse=reverse, name=name)
        self._phase = phase

    @property
    def phase(self):
        return self._phase

    def _build_internal_circuit(self) -> qiskit.QuantumCircuit:
        control_reg = create_register(self.num_control_qubits, name='control')
        state_reg = create_register(self.num_state_qubits, name='state')
        circuit = create_circuit(control_reg, state_reg, name=self.name)

        ref_state = [1] * self.num_state_qubits

        target_states = self.target_states
        if self.reverse:
            target_states = [
                state
                for state in get_basis_states(self.num_state_qubits, out_type='list')
                if state not in target_states
            ]

        for state in target_states:
            for bit, rbit, q in zip(state, ref_state, state_reg):
                if bool(bit) is not bool(rbit):
                    circuit.x(q)
            circuit.mcu1(self.phase, control_reg[:] + state_reg[:-1], state_reg[-1])
            ref_state = state

        for bit, q in zip(ref_state, state_reg):
            if not bit:
                circuit.x(q)

        self._set_internal_circuit(circuit)

        return circuit


class BooleanOracle(Oracle):
    def __init__(
            self,
            target_state: typing.Union[typing.Sequence[int], typing.Sequence[typing.Sequence[int]]],
            reverse: bool = False,
            name: typing.Optional[str] = 'BooleanOracle',
    ):
        super().__init__(target_state, reverse=reverse, name=name)

    def _build_internal_circuit(self) -> qiskit.QuantumCircuit:
        state_reg = qiskit.QuantumRegister(self.num_state_qubits, name='state')
        output_reg = qiskit.QuantumRegister(1, name='output')
        circuit = qiskit.QuantumCircuit(state_reg, output_reg, name=self.name)

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

        self._set_internal_circuit(circuit)

        return circuit

    def to_phase_oracle(self, num_controls: int = 0) -> PhaseOracle:
        control_reg = create_register(num_controls, 'control')
        state_reg = create_register(self.num_state_qubits, 'state')
        output_reg = create_register(1, 'ancilla', reg_type='ancilla')
        circuit = create_circuit(control_reg, state_reg, output_reg, name='Controlled'+self.name)
        if num_controls > 0:
            circuit.mcx(control_reg, output_reg)
        else:
            circuit.x(output_reg)
        circuit.h(output_reg)
        circuit = self.apply(circuit, state_reg, output_reg)
        circuit.h(output_reg)
        if num_controls > 0:
            circuit.mcx(control_reg, output_reg)
        else:
            circuit.x(output_reg)
        ret = PhaseOracle(self.target_states, num_control_qubits=num_controls, reverse=self.reverse)
        ret._set_internal_circuit(circuit)
        return ret


class GroverIterate(ControlledOperator):
    def __init__(
            self,
            a_op: QuantumOperator,
            rs_op: ControlledOperator,
            oracle: Oracle,
            name: typing.Optional[str] = 'GroverIterate',
    ):
        super().__init__(
            num_control_qubits=oracle.num_control_qubits,
            name=name,
        )
        self._a_op = a_op
        self._rs_op = rs_op
        self._oracle = oracle

    def _build_internal_circuit(self) -> qiskit.QuantumCircuit:
        control_reg = create_register(self.num_control_qubits, name='control')
        state_reg = create_register(self._oracle.num_state_qubits, name='state')
        num_ancillas = max(self._oracle.num_ancillas, self._a_op.num_ancillas, self._rs_op.num_ancillas)
        ancilla_reg = create_register(num_ancillas, name='ancilla', reg_type='ancilla')
        circuit = create_circuit(control_reg, state_reg, ancilla_reg, name=self.name)

        self._oracle(circuit, control=control_reg, state=state_reg, ancilla=ancilla_reg)
        self._a_op.apply_inverse(circuit, state_reg, ancilla=ancilla_reg)
        self._rs_op(circuit, control=control_reg, state=state_reg, ancilla=ancilla_reg)
        self._a_op(circuit, state_reg, ancilla=ancilla_reg)

        self._set_internal_circuit(circuit)

        return circuit


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
        oracle_type: str = 'phase',
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
    if oracle_type == 'phase':
        oracle = PhaseOracle(target_state=target)
    elif oracle_type == 'bool':
        oracle = BooleanOracle(target_state=target).to_phase_oracle()
    else:
        raise ValueError("Unknown oracle type %s", oracle_type)
    grover_op = GroverIterate(a_op=a_op, rs_op=rs_op, oracle=oracle)
    ancilla = create_register(grover_op.num_ancillas, name='ancilla')
    add_registers_to_circuit(qc, ancilla)
    for _ in range(niter):
        grover_op(qc, qreg, ancilla=ancilla)

    # measurement
    if measure:
        creg = qiskit.ClassicalRegister(grover_op.get_register('state').size, 'output')
        qc.add_register(creg)
        qc.measure(grover_op.get_register('state'), creg)

    return qc
