import typing
import copy
import itertools
from abc import ABC

import qiskit

from utils.qiskit_utils import QuantumRegisterType, create_circuit, QubitIterator


class QuantumOperator(ABC):
    """
    Base operator class
    """
    def __init__(
            self,
            name: typing.Optional[str] = None,
    ):
        self._circuit = None
        self._name = name

    @property
    def qregs(self) -> typing.List[qiskit.QuantumRegister]:
        return self._get_internal_circuit().qregs

    @property
    def qubits(self) -> typing.List[qiskit.circuit.Qubit]:
        return self._get_internal_circuit().qubits

    @property
    def ancillas(self) -> typing.List[qiskit.circuit.AncillaQubit]:
        return self._get_internal_circuit().ancillas

    @property
    def num_qubits(self) -> int:
        return self._get_internal_circuit().num_qubits

    @property
    def num_ancillas(self) -> int:
        return self._get_internal_circuit().num_ancillas

    @property
    def num_main_qubits(self) -> int:
        return self.num_qubits - self.num_ancillas

    @property
    def name(self) -> str:
        return self._name

    def _get_internal_circuit(self) -> qiskit.QuantumCircuit:
        if self._circuit is None:
            self._build_internal_circuit()
        return self._circuit

    def _set_internal_circuit(self, circuit) -> None:
        self._circuit = circuit
        # Get name from circuit
        if self.name is None:
            self._name = circuit.name
        # Rename circuit with provided name
        else:
            circuit.name = self.name

    def _build_internal_circuit(self) -> qiskit.QuantumCircuit:
        raise NotImplementedError("Abstract class")

    def draw(self, *args, **kwargs):
        return self._get_internal_circuit().draw(*args, **kwargs)

    def get_circuit(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            qregs: typing.Optional[typing.Sequence[QuantumRegisterType]] = None,
            inv: typing.Optional[bool] = False,
    ) -> qiskit.QuantumCircuit:

        # No register provided
        if qregs is None:
            # No circuit provided either
            if circuit is None:
                # Construct new circuit with internal registers
                circuit = qiskit.QuantumCircuit(*self.qregs, name=self.name)
            qregs = circuit.qregs
        # No circuit provided but register is provided
        elif circuit is None:
            circuit = create_circuit(*qregs)

        # Chain qregs into list of qubits
        qubits = list(itertools.chain(*qregs))

        if len(qubits) < self.num_qubits:
            raise RuntimeError('Register provided has insufficient length (%d < %d)', len(qubits), self.num_qubits)

        # Circuit for this operator
        _circuit = self._get_internal_circuit()

        # Inverse?
        if inv:
            _circuit = _circuit.inverse()

        # Append this operator to the circuit to be extended
        circuit.append(_circuit.to_instruction(), qubits)
        return circuit

    def inverse(self):
        inv = copy.deepcopy(self)
        inv._circuit = inv._circuit.inverse()
        return inv

    def apply(self, circuit: qiskit.QuantumCircuit, *qregs: QuantumRegisterType) -> qiskit.QuantumCircuit:
        return self.get_circuit(circuit, qregs)

    def apply_inverse(self, circuit: qiskit.QuantumCircuit, *qregs: QuantumRegisterType) -> qiskit.QuantumCircuit:
        return self.get_circuit(circuit, qregs, inv=True)

    def __call__(self, circuit: qiskit.QuantumCircuit, *qregs: QuantumRegisterType) -> qiskit.QuantumCircuit:
        return self.apply(circuit, *qregs)


class SegmentedOperator(QuantumOperator, ABC):
    """
    A quantum operator with named segments of the quantum registers / qubits
    This class does nothing more than the base QuantumOperator class but:
        1. Keeps track of the number of qubits in each segment
        2. Checks the input register lengths are conforming with the segment sizes
    """

    def __init__(
            self,
            segment_names: typing.Optional[typing.Sequence[str]] = None,
            segment_sizes: typing.Optional[typing.Sequence[typing.Optional[int]]] = None,
            name: typing.Optional[str] = None,
    ):
        super().__init__(name=name)

        if segment_names is None:
            segment_names = ['segment_%d' % i for i in range(len(segment_sizes))]
        else:
            segment_names = list(segment_names)

        if segment_sizes is not None:
            segment_sizes = list(segment_sizes)

        self._segment_sizes = segment_sizes
        self._segment_names = segment_names
        self._segment_qubits = None

    def _set_internal_circuit(self, circuit) -> None:
        super()._set_internal_circuit(circuit)

        if self.segment_sizes is None:
            self._segment_sizes = [self.num_qubits]

        assert len(self.segment_sizes) == len(self.segment_names)

        num_none_sizes = self.segment_sizes.count(None)
        if num_none_sizes == 1:
            self._segment_sizes[self.segment_sizes.index(None)] = \
                self.num_qubits - sum(s or 0 for s in self.segment_sizes)
        elif num_none_sizes > 1:
            missing_segments = [name for name, size in zip(self.segment_names, self.segment_names) if size is None]
            raise ValueError("Cannot determine sizes for segments: %s" % ', '.join(missing_segments))

        qubits = QubitIterator(*self.qregs)
        self._segment_qubits = [qubits.get(size) for size in self.segment_sizes]

    @property
    def segment_names(self) -> typing.List[str]:
        return self._segment_names

    @property
    def segment_sizes(self) -> typing.Optional[typing.List[typing.Optional[int]]]:
        return self._segment_sizes

    def _parse_input_qregs(
            self,
            *qregs: QuantumRegisterType,
            **named_qregs: QuantumRegisterType,
    ) -> typing.Dict[str, QuantumRegisterType]:
        """
        Parse input qregs for apply / apply_inverse
        :param qregs: positional input qregs
        :param named_qregs: named input qregs
        :return: dict of quantum register segments
        """

        qregs_iter = iter(qregs)

        # quantum register segments to be passed to the get_circuit method
        qreg_segments = {}

        for seg_name, seg_size in zip(self.segment_names, self.segment_sizes):
            if seg_size == 0:
                continue
            # get register from (1) named qregs and then (2) positional qregs
            qreg = named_qregs.get(seg_name, next(qregs_iter, None))

            if qreg is not None:
                qreg_segments[seg_name] = qreg

        return qreg_segments

    def get_segment_size(self, name: str) -> int:
        return self.segment_sizes[self.segment_names.index(name)]

    def get_segment_qubits(self, name: str) -> QuantumRegisterType:
        # Make sure the internal circuit is built
        self._get_internal_circuit()
        return self._segment_qubits[self.segment_names.index(name)]

    def get_circuit(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            qreg_segments: typing.Optional[typing.Dict[str, QuantumRegisterType]] = None,
            inv: typing.Optional[bool] = False,
    ) -> qiskit.QuantumCircuit:

        if qreg_segments is None:
            qreg_segments = {}

        # Qubit iterator on the qubits of the input circuit
        qubits = QubitIterator(*circuit.qregs) if circuit is not None else None

        # quantum registers to be passed to the base method
        _qregs = []

        for seg_name, seg_size in zip(self.segment_names, self.segment_sizes):
            if seg_size == 0:
                continue
            # get register from input args
            qreg = qreg_segments.get(seg_name)
            # register not provided for this segment
            if qreg is None:
                # no circuit provided, get from the internal circuit
                if circuit is None:
                    qreg = self.get_segment_qubits(seg_name)
                # get qubits from the provided circuit
                else:
                    qreg = qubits.get(seg_size)
            # allow input to be one single qubit
            elif isinstance(qreg, qiskit.circuit.Qubit):
                qreg = [qreg]

            # check segment size
            assert len(qreg) >= seg_size, \
                "Insufficient size for register %s. Expect %d got %d." % (seg_name, seg_size, len(qreg))

            # add to qubits list
            _qregs.append(qreg)

        return super().get_circuit(circuit, _qregs, inv=inv)

    def apply(
            self,
            circuit: qiskit.QuantumCircuit,
            *qregs: QuantumRegisterType,
            **named_qregs: QuantumRegisterType
    ) -> qiskit.QuantumCircuit:
        qreg_segments = self._parse_input_qregs(*qregs, **named_qregs)
        return self.get_circuit(circuit, qreg_segments)

    def apply_inverse(
            self,
            circuit: qiskit.QuantumCircuit,
            *qregs: QuantumRegisterType,
            **named_qregs: QuantumRegisterType,
    ) -> qiskit.QuantumCircuit:
        qreg_segments = self._parse_input_qregs(*qregs, **named_qregs)
        return self.get_circuit(circuit, qreg_segments, inv=True)


class SimpleOperator(SegmentedOperator, ABC):
    """
    A simple two-segment operator with target and ancilla qubits
    """

    def __init__(
            self,
            num_target_qubits: typing.Optional[int] = None,
            num_ancilla_qubits: int = 0,
            name: typing.Optional[str] = None,
    ):
        super().__init__(
            segment_names=['target', 'ancilla'],
            segment_sizes=[num_target_qubits, num_ancilla_qubits],
            name=name,
        )

    @property
    def num_target_qubits(self):
        return self.get_segment_size('target')

    @property
    def num_ancilla_qubits(self):
        return self.get_segment_size('ancilla')

    @property
    def target_qubits(self):
        return self.get_segment_qubits('target')

    @property
    def ancilla_qubits(self):
        return self.get_segment_qubits('ancilla')


class ControlledOperator(SegmentedOperator, ABC):
    """
    A three-segment operator with control, target and ancilla qubits
    """

    def __init__(
            self,
            num_control_qubits: int = 1,
            num_target_qubits: typing.Optional[int] = None,
            num_ancilla_qubits: int = 0,
            name: typing.Optional[str] = None,
    ):
        super().__init__(
            segment_names=['control', 'target', 'ancilla'],
            segment_sizes=[num_control_qubits, num_target_qubits, num_ancilla_qubits],
            name=name,
        )

    @property
    def num_control_qubits(self):
        return self.get_segment_size('control')

    @property
    def num_target_qubits(self):
        return self.get_segment_size('target')

    @property
    def num_ancilla_qubits(self):
        return self.get_segment_size('ancilla')

    @property
    def control_qubits(self):
        return self.get_segment_qubits('control')

    @property
    def target_qubits(self):
        return self.get_segment_qubits('target')

    @property
    def ancilla_qubits(self):
        return self.get_segment_qubits('ancilla')
