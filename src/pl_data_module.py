from dataclasses import dataclass
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
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

    def prepare_data(self):
        raise NotImplementedError()

    def preprocess_function(self):
        raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError()
