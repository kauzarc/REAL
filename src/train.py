import os
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

from pl_data_module import PLDataModule
from pl_module import PLModule


def callbacks(conf: DictConfig) -> List[Callback]:
    result = []

    if conf.train.apply_early_stopping:
        result.append(
            EarlyStopping(
                monitor=conf.train.monitor_var,
                mode=conf.train.monitor_var_mode,
                patience=conf.train.patience,
            )
        )

    result.append(
        ModelCheckpoint(
            monitor=conf.train.monitor_var,
            dirpath=f"experiments/{conf.train.model_name}",
            save_top_k=conf.train.save_top_k,
            verbose=True,
            save_last=True,
            mode=conf.train.monitor_var_mode,
        )
    )

    result.append(LearningRateMonitor(logging_interval="step"))

    return result


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def main(conf: DictConfig):
    seed_everything(conf.train.seed)

    config = AutoConfig.from_pretrained(conf.model.config_name)

    tokenizer = AutoTokenizer.from_pretrained(conf.model.tokenizer_name)
    with open(conf.data.tokens_file, "r") as tokens_file:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"<{token}>" for token in tokens_file]},
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model.model_name_or_path,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))

    pl_data_module = PLDataModule(conf, tokenizer, model)
    pl_module = PLModule(conf, config, tokenizer, model)

    logger = TensorBoardLogger(
        "tensorboard_logs/" + conf.data.dataset_name.split("/")[-1].replace(".py", ""),
    )

    trainer = Trainer(
        gpus=conf.train.gpus,
        accumulate_grad_batches=conf.train.gradient_acc_steps,
        gradient_clip_val=conf.train.gradient_clip_value,
        val_check_interval=conf.train.val_check_interval,
        callbacks=callbacks(conf),
        max_steps=conf.train.max_steps,
        precision=conf.train.precision,
        amp_level=conf.train.amp_level,
        logger=logger,
        limit_val_batches=conf.train.val_percent_check,
    )
    trainer.fit(
        pl_module, datamodule=pl_data_module, ckpt_path=conf.train.checkpoint_path
    )


if __name__ == "__main__":
    main()
