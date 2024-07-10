from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer, Callback
from pytorch_lightning.loggers import WandbLogger
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

    if conf.apply_early_stopping:
        result.append(
            EarlyStopping(
                monitor=conf.monitor_var,
                mode=conf.monitor_var_mode,
                patience=conf.patience,
            )
        )

    result.append(
        ModelCheckpoint(
            monitor=conf.monitor_var,
            dirpath=f"experiments/{conf.model_name}",
            save_top_k=conf.save_top_k,
            verbose=True,
            save_last=True,
            mode=conf.monitor_var_mode,
        )
    )

    result.append(LearningRateMonitor(logging_interval="step"))

    return result


@hydra.main(config_path="conf/", config_name="config")
def main(conf: DictConfig):
    seed_everything(conf.seed)

    config = AutoConfig.from_pretrained(conf.config_name)

    tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer_name)
    with open(conf.tokens_file, "r") as tokens_file:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"<{token}>" for token in tokens_file]},
            replace_additional_special_tokens=False,
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))

    pl_data_module = PLDataModule(conf, tokenizer, model)
    pl_module = PLModule(conf, config, tokenizer, model)

    wandb_logger = WandbLogger(
        project=conf.dataset_name.split("/")[-1].replace(".py", ""),
        name=conf.model_name_or_path.split("/")[-1],
    )

    trainer = Trainer(
        gpus=conf.gpus,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        val_check_interval=conf.val_check_interval,
        callbacks=callbacks(conf),
        max_steps=conf.max_steps,
        precision=conf.precision,
        amp_level=conf.amp_level,
        logger=wandb_logger,
        limit_val_batches=conf.val_percent_check,
    )
    trainer.fit(pl_module, datamodule=pl_data_module, ckpt_path=conf.checkpoint_path)


if __name__ == "__main__":
    main()
