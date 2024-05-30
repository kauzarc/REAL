import os
from collections.abc import Iterable, Sequence
from typing import Generator

from torch import LongTensor


class TripletTokenizer:
    def __init__(
        self,
        id_to_token: Sequence[str],
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
    ):
        self.id_to_token = id_to_token
        self.token_to_id = {token: i for i, token in enumerate(id_to_token)}

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

    def encode(self, text: Iterable[str]) -> LongTensor:
        return LongTensor(
            [self.token_to_id[self.bos_token]]
            + list(self.convert_tokens_to_ids(self.tokenize(sent)))
            + [self.token_to_id[self.eos_token]]
            for sent in text
        )

    def decode(self, token_ids: Iterable[int]) -> str:
        return " ".join(self.convert_ids_to_tokens(token_ids))

    @staticmethod
    def tokenize(text: str):
        return text.split()

    def convert_ids_to_tokens(
        self, token_ids: Iterable[int]
    ) -> Generator[str, None, None]:
        return (self.id_to_token[token_id] for token_id in token_ids)

    def convert_tokens_to_ids(
        self, tokens: Iterable[str]
    ) -> Generator[int, None, None]:
        return (
            self.token_to_id.get(token, self.token_to_id[self.unk_token])
            for token in tokens
        )

    def __call__(self, *args, **kwargs) -> LongTensor:
        return self.encode(*args, **kwargs)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            file.writelines(map(lambda x: x + "\n", self.id_to_token))

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as file:
            return cls(list(line for line in file))
