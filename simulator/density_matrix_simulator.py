import typing
import numpy as np


def _get_dimensions(mat: np.array):
    ndim = len(mat)
    num_qubits = int(np.log2(ndim))
    assert 2 ** num_qubits == ndim
    assert np.ndim(mat) in [1, 2]
    if np.ndim(mat) == 2:
        assert np.shape(mat) == (ndim, ndim)
    return num_qubits, ndim


class DensityMatrixSimulator(object):

    def __init__(self, num_qubits: int = 0):

        self._num_qubits = num_qubits
        self._density_matrix = None
        self.init_density_matrix()

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def density_matrix(self):
        return self._density_matrix

    def init_density_matrix(self):
        m = np.diag([1] + (2**self.num_qubits - 1) * [0])
        self._density_matrix = m

    def add_qubit(self, at: int = -1, state: typing.Optional[np.array] = None):
        if state is None:
            state = np.diag([1, 0])

        assert np.shape(state) == (2, 2)

        # Handle special cases for performance
        if at == 0:
            return self.prepend(state)
        if at == -1:
            return self.append(state)

        m = np.reshape(self.density_matrix, self.num_qubits * [2, 2])
        m = np.moveaxis(m, [2*at, 2*at+1], [0, 1])
        m = np.kron(state, m)
        m = np.moveaxis(m, [0, 1], [2*at, 2*at+1])
        self._density_matrix = m.reshape(2**(self.num_qubits+1), -1)
        self._num_qubits += 1

    def append(self, state: np.array):
        num_qubits, ndim = _get_dimensions(state)

        self._density_matrix = np.kron(self.density_matrix, np.reshape(state, [ndim, ndim]))
        self._num_qubits += num_qubits

    def prepend(self, state: np.array):
        num_qubits, ndim = _get_dimensions(state)

        self._density_matrix = np.kron(np.reshape(state, [ndim, ndim]), self.density_matrix)
        self._num_qubits += num_qubits

    def add_operator(self, op: np.array, qubits: typing.Optional[typing.Sequence[int]] = None):

        if isinstance(qubits, int):
            qubits = [qubits]

        num_qubits, ndim = _get_dimensions(op)
        if qubits is None:
            qubits = np.arange(num_qubits)

        assert len(qubits) == num_qubits

        sum_indices_1 = list(qubits)
        new_indices_1 = list(range(2*self.num_qubits, 2*self.num_qubits + len(qubits)))
        sum_indices_2 = list(self.num_qubits + i for i in qubits)
        new_indices_2 = list(range(2 * self.num_qubits + len(qubits), 2 * self.num_qubits + 2 * len(qubits)))
        old_indices = list(range(2*self.num_qubits))
        out_indices = list(range(2*self.num_qubits))
        for i, ind in enumerate(sum_indices_1):
            out_indices[sum_indices_1[i]] = new_indices_1[i]
        for i, ind in enumerate(sum_indices_2):
            out_indices[sum_indices_2[i]] = new_indices_2[i]

        m = np.reshape(self.density_matrix, 2 * self.num_qubits * [2])
        op = np.reshape(op, 2 * len(qubits) * [2])
        m = np.einsum(
            op.conj(), sum_indices_1 + new_indices_1,
            m, old_indices,
            op, sum_indices_2 + new_indices_2,
            out_indices,
        )
        self._density_matrix = m.reshape(2**self.num_qubits, -1)

    def add_measurement(
            self,
            qubits: typing.Union[int, typing.Sequence[int]],
            basis: typing.Optional[typing.Union[typing.Sequence[np.array], np.array]] = None,
    ):

        if isinstance(qubits, int):
            qubits = [qubits]

        if basis is None:
            ndim = 2**len(qubits)
            basis = np.identity(ndim)
        else:
            for proj in basis:
                # projector as vector: np.ndim(proj) = 1
                # projector as measurement operator: np.ndim(proj) = 2
                num_qubits, _ = _get_dimensions(proj)
                assert len(qubits) == num_qubits

        if len(basis) == 0:
            return

        sum_indices_1 = list(qubits)
        new_indices_1 = list(range(2 * self.num_qubits, 2 * self.num_qubits + len(qubits)))
        sum_indices_2 = list(self.num_qubits + i for i in qubits)
        new_indices_2 = list(range(2 * self.num_qubits + len(qubits), 2 * self.num_qubits + 2 * len(qubits)))
        old_indices = list(range(2 * self.num_qubits))
        out_indices = list(range(2 * self.num_qubits))
        for i, ind in enumerate(sum_indices_1):
            out_indices[sum_indices_1[i]] = new_indices_1[i]
        for i, ind in enumerate(sum_indices_2):
            out_indices[sum_indices_2[i]] = new_indices_2[i]

        m = np.reshape(self.density_matrix, 2 * self.num_qubits * [2])

        _m = 0
        for proj in basis:
            if np.ndim(proj) == 1:
                proj = np.outer(proj.conj().T, proj)
            proj = np.reshape(proj, 2 * len(qubits) * [2])
            _m += np.einsum(
                proj.conj(), sum_indices_1 + new_indices_1,
                m, old_indices,
                proj, sum_indices_2 + new_indices_2,
                out_indices,
            )
        self._density_matrix = np.reshape(_m, [2**self.num_qubits, -1])

    def get_probabilities(self, qubits: typing.Union[int, typing.Sequence[int]]):
        pass

    def get_statevector(self):
        pass
