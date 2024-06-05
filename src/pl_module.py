from pytorch_lightning import LightningModule
from torch import FloatTensor
from transformers import BartConfig, PreTrainedTokenizerFast, BartForConditionalGeneration  # type: ignore
from transformers.modeling_outputs import Seq2SeqLMOutput  # type: ignore

from data import Batch
from utils import shift_tokens_left


class PLModule(LightningModule):
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

    def forward(self):
        raise NotImplementedError()

    def training_step(self, batch: Batch, batch_idx: int) -> FloatTensor:
        batch_encoding = self.tokenizer(
            text=batch.inputs, text_target=batch.labels, return_tensors="pt"
        )
        inputs = {
            "input_ids": batch_encoding.input_ids,
            "attention_mask": batch_encoding.attention_mask,
            "decoder_input_ids": batch_encoding.labels,
            "labels": shift_tokens_left(
                batch_encoding.labels, self.config.pad_token_id
            ),
        }

        outputs: Seq2SeqLMOutput = self.model(**inputs)
        self.log("loss", outputs.loss)
        return outputs.loss  # type: ignore
