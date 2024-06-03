from typing import cast

from torch import LongTensor


def shift_tokens_left(input_ids: LongTensor, pad_token_id: int) -> LongTensor:
    shifted_input_ids = cast(LongTensor, input_ids.new_zeros(input_ids.shape))
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = pad_token_id

    assert pad_token_id is not None, "pad_token_id has to be defined."

    return shifted_input_ids
