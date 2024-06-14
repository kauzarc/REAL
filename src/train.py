import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

from pl_data_module import PLDataModule
from pl_module import PLModule


@hydra.main(config_path="conf/", config_name="config")
def main(conf: DictConfig):
    seed_everything(conf.seed)

    config = AutoConfig.from_pretrained(conf.config_name)
    tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )

    pl_data_module = PLDataModule(conf, tokenizer, model)
    pl_module = PLModule(conf, config, tokenizer, model)

    trainer = Trainer()
    trainer.fit(pl_module, datamodule=pl_data_module)


if __name__ == "__main__":
    main()
