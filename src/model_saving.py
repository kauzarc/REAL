from typing import Tuple

from transformers import (  # type: ignore
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartTokenizerFast,
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
)

from src.model import Model, ModelConfig


def load_bart() -> Tuple[BartTokenizerFast, BartForConditionalGeneration]:
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

    return tokenizer, model


def load_rebel() -> Tuple[PreTrainedTokenizerFast, BartForConditionalGeneration]:
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    return tokenizer, model


def copy_params(rebel_model: BartForConditionalGeneration, base_model: Model) -> None:
    def update_key(key: str) -> str:
        if key.startswith("model.encoder.layers."):
            split_key = key.split(".")
            split_key[3] = str(
                int(split_key[3]) + base_model.model.before_decoder_layers
            )

            key = ".".join(split_key)

        return key

    base_model.load_state_dict(
        {
            update_key(key): value
            for key, value in rebel_model.state_dict().items()
            if key
            not in (
                "final_logits_bias",
                "model.shared.weight",
                "model.encoder.embed_tokens.weight",
                "model.decoder.embed_tokens.weight",
                "lm_head.weight",
            )
        },
        strict=False,
    )


def save_tokenizer() -> None:
    bart_tokenizer, _ = load_bart()
    bart_tokenizer.save_pretrained("tokenizer/bart")


def save_model() -> None:
    _, rebel_model = load_rebel()

    model = Model(
        ModelConfig(
            before_decoder_layers=3,
            after_decoder_layer=3,
            **rebel_model.config.to_diff_dict()
        )
    )

    copy_params(rebel_model, model)

    model.save_pretrained("models/base_model")


def main() -> None:
    save_tokenizer()
    save_model()


if __name__ == "__main__":
    main()
