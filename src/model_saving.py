from typing import Tuple

from transformers import (  # type: ignore
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
)

from src.model import Model, ModelConfig

BEFORE_DECODER_LAYERS = 3
AFTER_DECODER_LAYERS = 3


def load_rebel() -> Tuple[PreTrainedTokenizerFast, BartForConditionalGeneration]:
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    return tokenizer, model


def set_params(rebel_model: BartForConditionalGeneration, base_model: Model) -> None:
    def update_key(key: str) -> str:
        if key.startswith("model.encoder.layers."):
            split_key = key.split(".")
            split_key[3] = str(int(split_key[3]) + BEFORE_DECODER_LAYERS)

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


def main() -> None:
    _, rebel_model = load_rebel()

    model = Model(
        ModelConfig(
            decoder_layers=rebel_model.config.decoder_layers
            + BEFORE_DECODER_LAYERS
            + AFTER_DECODER_LAYERS
        )
    )

    set_params(rebel_model, model)

    model.save_pretrained("models/base_model")


if __name__ == "__main__":
    main()
