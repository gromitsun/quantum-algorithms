import typing


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