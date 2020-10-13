import typing
import copy
import itertools
from abc import ABC

import qiskit

from utils.qiskit_utils import QuantumRegisterType, create_circuit, create_register, \
    add_registers_to_circuit, split_register


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
    def named_qregs(self) -> typing.Dict[str, qiskit.QuantumRegister]:
        return {qreg.name: qreg for qreg in self.qregs}

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

    def _parse_input_qregs(
            self,
            qregs: typing.Optional[typing.Sequence[QuantumRegisterType]] = None,
            named_qregs: typing.Optional[typing.Dict[str, QuantumRegisterType]] = None,
            extra_named_qregs: typing.Optional[typing.Dict[str, QuantumRegisterType]] = None,
    ) -> typing.List[QuantumRegisterType]:
        """
        Get quantum arguments according to the qregs specified in this operator.
        For each defined argument in self.qregs:
            (1) First get register from named_qregs,
            (2) If not available get register from qregs
            (3) if not available get register from extra_named_qregs
        All inputs are size checked.
        :param qregs: ordered sequence of qregs
        :param named_qregs: dict of qregs
        :param extra_named_qregs: dict of qregs
        :return: list of qregs corresponding to the required quantum arguments of this operator
        """

        # Iterator over positional qregs
        qregs_iter = iter(qregs)

        # quantum registers to be passed to the get_circuit method
        parsed_qregs = []

        for expected_qreg in self.qregs:
            # Try to get register from the following in order:
            #   (1) named qregs
            #   (2) positional qregs
            #   (3) qreg with the same name from the circuit provided
            qreg = named_qregs.get(
                expected_qreg.name,
                next(
                    qregs_iter,
                    extra_named_qregs.get(expected_qreg.name),
                ),
            )

            # Register not provided
            if qreg is None:
                raise RuntimeError(
                    "Missing register for %s (size = %d)",
                    expected_qreg.name, expected_qreg.size,
                )

            # Allow single Qubit inputs
            if isinstance(qreg, qiskit.circuit.Qubit):
                qreg = [qreg]

            if len(qreg) < expected_qreg.size:
                raise RuntimeError(
                    "Insufficient size for register %s. Expected %d got %d.",
                    expected_qreg.name, expected_qreg.size, len(qreg)
                )

            qreg = qreg[:expected_qreg.size]

            parsed_qregs.append(qreg)

        return parsed_qregs

    def _parse_input(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            qregs: typing.Optional[typing.Sequence[QuantumRegisterType]] = None,
            named_qregs: typing.Optional[typing.Dict[str, QuantumRegisterType]] = None,
    ) -> (qiskit.QuantumCircuit, typing.List[QuantumRegisterType]):
        """
        Parse input qregs for get_circuit()
        :param circuit: circuit to append the operator to.
                        If not provided, a new circuit will be constructed.
        :param qregs: positional input qregs
        :param named_qregs: named input qregs
        :return: dict of quantum register segments
        """

        if qregs is None:
            qregs = []
        if named_qregs is None:
            named_qregs = {}

        if circuit is None:
            named_qregs_from_circuit = self.named_qregs
            circuit = qiskit.QuantumCircuit(name=self.name)
        else:
            named_qregs_from_circuit = {qreg.name: qreg for qreg in circuit.qregs}

        parsed_qregs = self._parse_input_qregs(qregs, named_qregs, named_qregs_from_circuit)

        add_registers_to_circuit(circuit, *parsed_qregs)

        return circuit, parsed_qregs

    def _get_internal_circuit(self) -> qiskit.QuantumCircuit:
        if self._circuit is None:
            circuit = self._build_internal_circuit()
            self._set_internal_circuit(circuit)
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

    @classmethod
    def create(cls, circuit: qiskit.QuantumCircuit, name: typing.Optional[str] = None):
        op = cls(name=name)
        op._set_internal_circuit(circuit)
        return op

    def split_register(self, *qregs):
        return split_register(registers=qregs, sizes=[qreg.size for qreg in self.qregs])

    def draw(self, *args, **kwargs):
        return self._get_internal_circuit().draw(*args, **kwargs)

    def get_register(self, name: str, default: typing.Any = None) -> typing.Union[QuantumRegisterType, typing.Any]:
        for qreg in self.qregs:
            if qreg.name == name:
                return qreg
        return default

    def get_circuit(
            self,
            circuit: typing.Optional[qiskit.QuantumCircuit] = None,
            qregs: typing.Optional[typing.Sequence[QuantumRegisterType]] = None,
            named_qregs: typing.Optional[typing.Dict[str, QuantumRegisterType]] = None,
            inv: typing.Optional[bool] = False,
    ) -> qiskit.QuantumCircuit:
        """
        Get a circuit with this operator applied.
        :param circuit: (Optional) circuit to append the operator to.
                        If not provided, a new circuit will be constructed.
        :param qregs: (Optional) quantum registers to apply this operator to.
                      If not provided, registers from the internal circuit will be used.
        :param named_qregs:
        :param inv:  (bool, Optional) invert the operator
        :return: quantum circuit
        """

        circuit, qregs = self._parse_input(circuit, qregs, named_qregs)

        # Chain qregs into list of qubits
        # Note that this may be a subset of circuit.qubits when both circuit and qregs are provided
        qubits = list(itertools.chain(*qregs))

        if len(qubits) < self.num_qubits:
            raise RuntimeError('Register provided has insufficient length (%d < %d)', len(qubits), self.num_qubits)

        # Circuit for this operator
        _circuit = self._get_internal_circuit()

        # Inverse?
        if inv:
            _circuit = _circuit.inverse()

        # Append this operator to the circuit to be extended
        circuit.append(_circuit.to_instruction(), qubits[:self.num_qubits])
        return circuit

    def inverse(self):
        inv = copy.deepcopy(self)
        inv._circuit = inv._circuit.inverse()
        return inv

    def apply(
            self,
            circuit: qiskit.QuantumCircuit,
            *qregs: typing.Union[QuantumRegisterType, qiskit.circuit.Qubit],
            **named_qregs: typing.Union[QuantumRegisterType, qiskit.circuit.Qubit],
    ) -> qiskit.QuantumCircuit:
        return self.get_circuit(circuit, qregs, named_qregs)

    def apply_inverse(
            self,
            circuit: qiskit.QuantumCircuit,
            *qregs: typing.Union[QuantumRegisterType, qiskit.circuit.Qubit],
            **named_qregs: typing.Union[QuantumRegisterType, qiskit.circuit.Qubit],
    ) -> qiskit.QuantumCircuit:
        return self.get_circuit(circuit, qregs, named_qregs, inv=True)

    def __call__(
            self,
            circuit: qiskit.QuantumCircuit,
            *qregs: typing.Union[QuantumRegisterType, qiskit.circuit.Qubit],
            **named_qregs: typing.Union[QuantumRegisterType, qiskit.circuit.Qubit],
    ) -> qiskit.QuantumCircuit:
        return self.apply(circuit, *qregs, **named_qregs)


class ControlledOperator(QuantumOperator, ABC):
    """
    A two-segment operator with control and target qubits
    """

    def __init__(
            self,
            num_control_qubits: int = 1,
            name: typing.Optional[str] = None,
    ):
        super().__init__(name=name)
        self._num_control_qubits = num_control_qubits

    @property
    def num_control_qubits(self):
        return self._num_control_qubits

    @property
    def num_target_qubits(self):
        return self.num_main_qubits - self.num_control_qubits


def create_ancillas_for(*operators: QuantumOperator, name: typing.Optional[str] = 'ancilla'):
    num_ancillas = max(op.num_ancillas for op in operators)
    return create_register(num_ancillas, name=name, reg_type='ancilla')
