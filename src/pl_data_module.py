from dataclasses import dataclass
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset


@dataclass
class Batch:
    inputs: list[str]
    labels: list[str]


class PLDataModule(LightningDataModule):
    def __init__(
        self,
        conf: DictConfig,
        tokenizer: PreTrainedTokenizerFast,
        model: BartForConditionalGeneration,
    ):
        super().__init__()

        self.conf = conf
        self.tokenizer = tokenizer
        self.model = model

        self.dataset = load_dataset(
            conf.dataset_name,
            data_files={
                "train": conf.train_file,
                "dev": conf.validation_file,
                "test": conf.test_file,
                "tokens": conf.tokens_file,
            },
        )

        label_pad_token_id = (
            -100 if conf.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        )
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, self.model, label_pad_token_id=label_pad_token_id
        )

    def prepare_data(self):
        raise NotImplementedError()

    def preprocess_function(self):
        raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.train_batch_size,
            # sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError()
