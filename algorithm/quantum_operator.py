import typing
import copy
import itertools

import qiskit

from utils.qiskit_utils import QuantumRegisterType, create_circuit


class QubitIterator(object):
    def __init__(self, *qregs:QuantumRegisterType):
        self._qregs = qregs
        self._qubits = itertools.chain(*qregs)

    @property
    def qregs(self):
        return self._qregs

    def get(self, n: typing.Optional[int] = None):
        if n is None:
            # return a list of all qubits
            return list(self._qubits)
        # return a list of n qubits
        return [next(self._qubits) for _ in range(n)]


class QuantumOperator(object):
    """
    Base operator class
    """
    def __init__(
            self,
            circuit: qiskit.QuantumCircuit,
            name: typing.Optional[str] = None,
    ):
        self._circuit = circuit
        if name is not None:
            self._circuit.name = name

    @property
    def num_qubits(self) -> int:
        return self._circuit.num_qubits

    @property
    def qregs(self) -> typing.List[qiskit.QuantumRegister]:
        return self._circuit.qregs

    @property
    def name(self):
        return self._circuit.name

    def draw(self, *args, **kwargs):
        return self._circuit.draw(*args, **kwargs)

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
                circuit = qiskit.QuantumCircuit(*self._circuit.qregs, name=self.name)
            qregs = circuit.qregs
        # No circuit provided but register is provided
        elif circuit is None:
            circuit = create_circuit(*qregs)

        # Chain qregs into list of qubits
        qubits = list(itertools.chain(*qregs))

        if len(qubits) < self.num_qubits:
            raise RuntimeError('Register provided has insufficient length (%d < %d)', len(qubits), self.num_qubits)

        # Circuit for this operator
        _circuit = self._circuit

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


class SegmentedOperator(QuantumOperator):
    """
    A quantum operator with named segments of the quantum registers / qubits
    This class does nothing more than the base QuantumOperator class but:
        1. Keeps track of the number of qubits in each segment
        2. Checks the input register lengths are conforming with the segment sizes
    """

    def __init__(
            self,
            circuit: qiskit.QuantumCircuit,
            segment_names: typing.Optional[typing.Sequence[str]] = None,
            segment_sizes: typing.Optional[typing.Sequence[int]] = None,
            name: typing.Optional[str] = None,
    ):
        super().__init__(circuit, name=name)

        if segment_sizes is None:
            segment_sizes = [self.num_qubits]
        if segment_names is None:
            segment_names = ['segment_%d' % i for i in range(len(segment_sizes))]

        assert len(segment_sizes) == len(segment_names)

        num_none_sizes = segment_sizes.count(None)
        if num_none_sizes == 1:
            segment_sizes[segment_sizes.index(None)] = self.num_qubits - sum(s or 0 for s in segment_sizes)
        elif num_none_sizes > 1:
            missing_segments = [name for name, size in zip(segment_names, segment_names) if size is None]
            raise ValueError("Cannot determine sizes for segments: %s" % ', '.join(missing_segments))

        self._segment_sizes = segment_sizes
        self._segment_names = segment_names
        qubits = QubitIterator(*circuit.qregs)
        self._segment_qubits = [qubits.get(size) for size in self.segment_sizes]

    @property
    def segment_names(self):
        return self._segment_names

    @property
    def segment_sizes(self):
        return self._segment_sizes

    def get_segment_size(self, name: str) -> int:
        return self.segment_sizes[self.segment_names.index(name)]

    def get_segment_qubits(self, name: str) -> QuantumRegisterType:
        return self._segment_qubits[self.segment_names.index(name)]

    def get_circuit(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            qreg_segments: typing.Optional[typing.Dict[str, QuantumRegisterType]] = None,
            inv: typing.Optional[bool] = False,
    ) -> qiskit.QuantumCircuit:
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


class SimpleOperator(SegmentedOperator):
    """
    A simple two-segment operator with target and ancilla qubits
    """

    def __init__(
            self,
            circuit: qiskit.QuantumCircuit,
            num_ancilla_qubits: int = 0,
            name: typing.Optional[str] = None,
    ):
        super().__init__(
            circuit,
            segment_names=['target', 'ancilla'],
            segment_sizes=[None, num_ancilla_qubits],
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

    def get_circuit(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            target_qubits: typing.Optional[QuantumRegisterType] = None,
            ancilla_qubits: typing.Optional[QuantumRegisterType] = None,
            inv: typing.Optional[bool] = False,
    ) -> qiskit.QuantumCircuit:

        qregs = {
            'target': target_qubits,
            'ancilla': ancilla_qubits,
        }
        return super().get_circuit(circuit, qregs, inv=inv)

    def apply(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            target_qubits: typing.Optional[QuantumRegisterType] = None,
            ancilla_qubits: typing.Optional[QuantumRegisterType] = None,
    ) -> qiskit.QuantumCircuit:
        return self.get_circuit(circuit, target_qubits, ancilla_qubits)

    def apply_inverse(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            target_qubits: typing.Optional[QuantumRegisterType] = None,
            ancilla_qubits: typing.Optional[QuantumRegisterType] = None,
    ) -> qiskit.QuantumCircuit:
        return self.get_circuit(circuit, target_qubits, ancilla_qubits, inv=True)


class ControlledOperator(SegmentedOperator):
    """
    A three-segment operator with control, target and ancilla qubits
    """

    def __init__(
            self,
            circuit: qiskit.QuantumCircuit,
            num_control_qubits: int = 1,
            num_ancilla_qubits: int = 0,
            name: typing.Optional[str] = None,
    ):
        super().__init__(
            circuit,
            segment_names=['control', 'target', 'ancilla'],
            segment_sizes=[num_control_qubits, None, num_ancilla_qubits],
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

    def get_circuit(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            control_qubits: typing.Optional[QuantumRegisterType] = None,
            target_qubits: typing.Optional[QuantumRegisterType] = None,
            ancilla_qubits: typing.Optional[QuantumRegisterType] = None,
            inv: typing.Optional[bool] = False,
    ) -> qiskit.QuantumCircuit:

        qregs = {
            'control': control_qubits,
            'target': target_qubits,
            'ancilla': ancilla_qubits,
        }
        return super().get_circuit(circuit, qregs, inv=inv)

    def apply(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            control_qubits: typing.Optional[QuantumRegisterType] = None,
            target_qubits: typing.Optional[QuantumRegisterType] = None,
            ancilla_qubits: typing.Optional[QuantumRegisterType] = None,
    ) -> qiskit.QuantumCircuit:
        return self.get_circuit(circuit, control_qubits, target_qubits, ancilla_qubits)

    def apply_inverse(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            control_qubits: typing.Optional[QuantumRegisterType] = None,
            target_qubits: typing.Optional[QuantumRegisterType] = None,
            ancilla_qubits: typing.Optional[QuantumRegisterType] = None,
    ) -> qiskit.QuantumCircuit:
        return self.get_circuit(circuit, control_qubits, target_qubits, ancilla_qubits, inv=True)


# class ControlledOperator(QuantumOperator):
#
#     def __init__(
#             self,
#             circuit: qiskit.QuantumCircuit,
#             num_control_qubits: int = 1,
#             num_ancilla_qubits: int = 0,
#             name: typing.Optional[str] = None,
#     ):
#         super().__init__(circuit, name=name)
#
#         self._num_control_qubits = num_control_qubits
#         self._num_ancilla_qubits = num_ancilla_qubits
#         self._num_target_qubits = self.num_qubits - self.num_control_qubits - self.num_ancilla_qubits
#         qubits = QubitIterator(*self.qregs)
#         self._control_qubits = qubits.get(self.num_control_qubits)
#         self._target_qubits = qubits.get(self.num_target_qubits)
#         self._ancilla_qubits = qubits.get(self.num_ancilla_qubits)
#
#     @property
#     def num_control_qubits(self):
#         return self._num_control_qubits
#
#     @property
#     def num_target_qubits(self):
#         return self._num_target_qubits
#
#     @property
#     def num_ancilla_qubits(self):
#         return self._num_ancilla_qubits
#
#     def get_circuit(
#             self,
#             circuit: typing.Optional[qiskit.QuantumCircuit] = None,
#             control_qubits: typing.Optional[QuantumRegisterType] = None,
#             target_qubits: typing.Optional[QuantumRegisterType] = None,
#             ancilla_qubits: typing.Optional[QuantumRegisterType] = None,
#             inv: typing.Optional[bool] = False,
#     ) -> qiskit.QuantumCircuit:
#
#         qubits = QubitIterator(*circuit.qregs) if circuit is not None else None
#
#         if control_qubits is None:
#             if circuit is None:
#                 control_qubits = self._control_qubits
#             else:
#                 control_qubits = qubits.get(self.num_control_qubits)
#         # allow control register to be one single qubit
#         elif isinstance(control_qubits, qiskit.circuit.Qubit):
#             control_qubits = [control_qubits]
#         if target_qubits is None:
#             if circuit is None:
#                 target_qubits = self._target_qubits
#             else:
#                 target_qubits = qubits.get(self.num_target_qubits)
#         if ancilla_qubits is None:
#             if circuit is None:
#                 ancilla_qubits = self._ancilla_qubits
#             else:
#                 ancilla_qubits = qubits.get(self.num_ancilla_qubits)
#
#         assert len(control_qubits) == self.num_control_qubits
#         assert len(target_qubits) == self.num_target_qubits
#         assert len(ancilla_qubits) == self.num_ancilla_qubits
#
#         qregs = [reg for reg in (control_qubits, target_qubits, ancilla_qubits) if reg]
#
#         return super().get_circuit(circuit, qregs, inv=inv)
#
#     def apply(
#             self,
#             circuit: typing.Optional[qiskit.QuantumCircuit] = None,
#             control_qubits: typing.Optional[QuantumRegisterType] = None,
#             target_qubits: typing.Optional[QuantumRegisterType] = None,
#             ancilla_qubits: typing.Optional[QuantumRegisterType] = None,
#     ) -> qiskit.QuantumCircuit:
#         return self.get_circuit(circuit, control_qubits, target_qubits, ancilla_qubits)
#
#     def apply_inverse(
#             self,
#             circuit: typing.Optional[qiskit.QuantumCircuit] = None,
#             control_qubits: typing.Optional[QuantumRegisterType] = None,
#             target_qubits: typing.Optional[QuantumRegisterType] = None,
#             ancilla_qubits: typing.Optional[QuantumRegisterType] = None,
#     ) -> qiskit.QuantumCircuit:
#         return self.get_circuit(circuit, control_qubits, target_qubits, ancilla_qubits, inv=True)
