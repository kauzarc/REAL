from typing import List, Dict
from typing_extensions import TypedDict

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset


class Batch(TypedDict):
    input_ids: LongTensor
    attention_mask: LongTensor
    labels: LongTensor


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

        self.datasets = load_dataset(
            conf.data.dataset_name,
            data_files={
                "train": conf.data.train_file,
                "dev": conf.data.validation_file,
                "test": conf.data.test_file,
                "tokens": conf.data.tokens_file,
            },
        )

        self.column_names = self.datasets["train"].column_names

        self.train_dataset = None
        self.eval_dataset = None

        label_pad_token_id = (
            -100 if conf.data.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        )
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, self.model, label_pad_token_id=label_pad_token_id
        )

    def prepare_data(self) -> None:
        self.train_dataset = self.datasets["train"].map(
            self.preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.conf.data.overwrite_cache,
            cache_file_name=self.conf.data.train_file.replace(".jsonl", "-")
            + self.conf.data.dataset_name.split("/")[-1].replace(".py", ".cache"),
        )

        self.eval_dataset = self.datasets["validation"].map(
            self.preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.conf.data.overwrite_cache,
            cache_file_name=self.conf.data.validation_file.replace(".jsonl", "-")
            + self.conf.data.dataset_name.split("/")[-1].replace(".py", ".cache"),
        )

    def preprocess_function(self, examples: Dict[str, List]) -> Batch:
        model_inputs = self.tokenizer(
            examples["context"],
            max_length=self.conf.data.max_source_length,
            truncation=True,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["triplets"],
                max_length=self.conf.data.max_target_length,
                truncation=True,
            )

        return Batch(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=labels["input_ids"],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.train.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.conf.train.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.data.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )
