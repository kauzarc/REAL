from typing import List, Tuple, Any, Union, Dict

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import FloatTensor
from transformers import BartConfig, PreTrainedTokenizerFast, BartForConditionalGeneration, AdamW  # type: ignore
from transformers.modeling_outputs import Seq2SeqLMOutput  # type: ignore
from transformers.optimization import AdamW, get_scheduler

from pl_data_module import Batch
from utils import shift_tokens_left, split_on_condition


class PLModule(LightningModule):
    def __init__(
        self,
        omegaconf: DictConfig,
        config: BartConfig,
        tokenizer: PreTrainedTokenizerFast,
        model: BartForConditionalGeneration,
    ):
        super().__init__()

        self.save_hyperparameters(omegaconf)
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

    def forward(self, batch: Batch, batch_idx: int) -> Seq2SeqLMOutput:
        batch_encoding = self.tokenizer(
            text=batch.inputs, text_target=batch.labels, return_tensors="pt"
        )

        labels = shift_tokens_left(batch_encoding.labels, self.config.pad_token_id)

        if self.hparams.ignore_pad_token_for_loss:
            labels.masked_fill_(labels == self.config.pad_token_id, -100)

        inputs = {
            "input_ids": batch_encoding.input_ids,
            "attention_mask": batch_encoding.attention_mask,
            "decoder_input_ids": batch_encoding.labels,
            "labels": labels,
        }

        return self.model(**inputs)

    def training_step(self, batch: Batch, batch_idx: int) -> FloatTensor:
        outputs = self.forward(batch, batch_idx)
        self.log("train_loss", outputs.loss)
        return outputs.loss  # type: ignore

    def validation_step(self, batch: Batch, batch_idx: int) -> FloatTensor:
        outputs = self.forward(batch, batch_idx)
        self.log("validation_loss", outputs.loss)
        return outputs.loss  # type: ignore

    def configure_optimizers(
        self,
    ) -> Tuple[List[AdamW], List[Dict[str, Union[str, Any]]]]:
        no_decay = ["bias", "LayerNorm.weight"]

        params_decay, params_no_decay = split_on_condition(
            self.model.named_parameters(),
            lambda x: not any(sub_string in x[0] for sub_string in no_decay),
            lambda x: x[1],
        )

        grouped_parameters = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
        ]

        args = {
            "lr": self.hparams.learning_rate,
            "betas": (self.hparams.adam_beta1, self.hparams.adam_beta2),
            "eps": self.hparams.adam_epsilon,
        }

        optimizer = AdamW(grouped_parameters, **args)  # type: ignore
        scheduler = get_scheduler(
            self.hparams.lr_scheduler,
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
