from dataclasses import dataclass, asdict
from typing import Optional, cast

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, LongTensor
from transformers import (  # type: ignore
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    BartConfig,
)
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput  # type: ignore

from src.utils import shift_tokens_left


@dataclass
class ForwardInputs:
    input_ids: LongTensor
    decoder_input_ids: Optional[LongTensor]


@dataclass
class Batch:
    input_ids: LongTensor
    labels: LongTensor


@dataclass
class ForwardOutput:
    loss: Tensor
    logits: Tensor


class Model(LightningModule):
    def __init__(
        self,
        config: BartConfig,
        tokenizer: PreTrainedTokenizerFast,
        model: BartForConditionalGeneration,
    ):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, inputs: ForwardInputs, labels: LongTensor) -> ForwardOutput:
        outputs: Seq2SeqSequenceClassifierOutput = self.model(
            **asdict(inputs), use_cache=False, output_hidden_states=True
        )
        lprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        labels.masked_fill_(cast(Tensor, labels == -100), self.config.pad_token_id)
        loss, _ = self.loss_fn(lprobs, labels, ignore_index=self.config.pad_token_id)

        return ForwardOutput(loss, outputs.logits)

    def training_step(self, batch: Batch) -> Tensor:
        output = self.forward(
            ForwardInputs(batch.input_ids, batch.labels),
            shift_tokens_left(batch.labels, self.config.pad_token_id),
        )

        raise NotImplementedError()
