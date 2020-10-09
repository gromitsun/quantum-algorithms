import typing
import numpy as np


BinarySequenceType = typing.Sequence[typing.Union[int, bool]]
BinaryStrType = typing.Union[BinarySequenceType, str]


def int_to_bin_str(x: int, n_bits: typing.Union[int, None] = None) -> str:
    """int -> binary string (small-endian)"""
    bstr = bin(x)[2:]
    if n_bits is not None:
        return bstr.zfill(n_bits)
    return bstr


def int_to_bin_list(x: int, n_bits: typing.Union[int, None] = None) -> typing.List[int]:
    """int -> binary representation in a list (small-endian)"""
    bstr = int_to_bin_str(x, n_bits=n_bits)
    return [int(digit) for digit in bstr]


def int_to_bin(
        x: int,
        n_bits: typing.Union[int, None] = None,
        out_type: str = 'str'
) -> typing.Union[str, typing.List[int]]:
    if out_type.startswith('s'):
        return int_to_bin_str(x, n_bits=n_bits)
    elif out_type.startswith('l'):
        return int_to_bin_list(x, n_bits=n_bits)
    raise ValueError("'out_type' can only be either 'str' or 'list', got %s instead." % out_type)


def bin_list_to_int(
        bin_list: BinarySequenceType,
) -> float:
    return 2 ** np.arange(len(bin_list)) @ bin_list


def state_to_sv(
        states: typing.Union[BinaryStrType, typing.Sequence[BinaryStrType]],
        weights: typing.Optional[typing.Sequence[float]] = None,
) -> np.array:
    assert len(states) > 0
    if isinstance(states, str) or isinstance(states[0], (int, bool)):
        states = [states]

    if weights is None:
        weights = [1] * len(states)

    num_qubits = len(states[0])

    sv = np.zeros(2**num_qubits)

    weights = np.asarray(weights) / np.linalg.norm(weights)

    for state, weight in zip(states, weights):
        if isinstance(state, str):
            state_i = int(state, base=2)
        else:
            state_i = bin_list_to_int(state)
        sv[state_i] = weight

    return sv
