import typing


def int_to_bin(x: int, n_bits: typing.Union[int, None] = None) -> str:
    bstr = bin(x)[2:]
    if n_bits is not None:
        return bstr.zfill(n_bits)
    return bstr
