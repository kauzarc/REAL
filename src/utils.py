from typing import cast, Tuple, TypeVar, List, Iterable, Callable

from torch import LongTensor

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def shift_tokens_left(input_ids: LongTensor, pad_token_id: int) -> LongTensor:
    shifted_input_ids = cast(LongTensor, input_ids.new_zeros(input_ids.shape))
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = pad_token_id

    assert pad_token_id is not None, "pad_token_id has to be defined."

    return shifted_input_ids


def unit(x: T1) -> T1:
    return x


def split_on_condition(
    seq: Iterable[T1],
    condition: Callable[[T1], bool],
    mapping: Callable[[T1], T2] = unit,
) -> Tuple[List[T2], List[T2]]:
    a, b = [], []
    for item in seq:
        (a if condition(item) else b).append(mapping(item))
    return a, b
