from pytorch_lightning import LightningModule
from torch import FloatTensor
from transformers import BartConfig, PreTrainedTokenizerFast, BartForConditionalGeneration  # type: ignore
from transformers.modeling_outputs import Seq2SeqLMOutput  # type: ignore
from transformers.optimization import AdamW
from omegaconf import OmegaConf

from data import Batch
from utils import shift_tokens_left


class PLModule(LightningModule):
    def __init__(
        self,
        omegaconf: OmegaConf,
        config: BartConfig,
        tokenizer: PreTrainedTokenizerFast,
        model: BartForConditionalGeneration,
    ):
        super().__init__()

        self.save_hyperparameters(omegaconf)
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
        self.log("train_loss", outputs.loss)
        return outputs.loss  # type: ignore

    def configure_optimizers(self) -> AdamW:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if not any(sub_string in name for sub_string in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if any(sub_string in name for sub_string in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(optimizer_grouped_parameters)  # type: ignore
