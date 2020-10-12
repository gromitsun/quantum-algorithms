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

    @staticmethod
    def _parse_regs_or_qubits(
            qregs: typing.Sequence[typing.Union[QuantumRegisterType, qiskit.circuit.Qubit]],
    ) -> typing.List[QuantumRegisterType]:
        """
        Parse list of qregs but also accept one single qubit as a register
        :param qregs: list of qregs or single qubits
        :return: list of QuantumRegisterType
        """
        return [[qreg] if isinstance(qreg, qiskit.circuit.Qubit) else qreg for qreg in qregs]

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

    def get_register(self, name) -> typing.Optional[QuantumRegisterType]:
        for qreg in self.qregs:
            if qreg.name == name:
                return qreg

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
        circuit.append(_circuit.to_instruction(), qubits[:self.num_qubits])
        return circuit

    def inverse(self):
        inv = copy.deepcopy(self)
        inv._circuit = inv._circuit.inverse()
        return inv

    def apply(self, circuit: qiskit.QuantumCircuit, *qregs: QuantumRegisterType) -> qiskit.QuantumCircuit:
        return self.get_circuit(circuit, type(self)._parse_regs_or_qubits(qregs))

    def apply_inverse(self, circuit: qiskit.QuantumCircuit, *qregs: QuantumRegisterType) -> qiskit.QuantumCircuit:
        return self.get_circuit(circuit, type(self)._parse_regs_or_qubits(qregs), inv=True)

    def __call__(self, circuit: qiskit.QuantumCircuit, *qregs: QuantumRegisterType) -> qiskit.QuantumCircuit:
        return self.apply(circuit, *qregs)


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
