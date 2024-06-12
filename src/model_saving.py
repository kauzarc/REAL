from typing import Tuple

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    BartTokenizerFast,
    BartForConditionalGeneration,
)


def load_model(
    name: str,
) -> Tuple[BartTokenizerFast, BartForConditionalGeneration]:
    config = AutoConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, config=config)

    return tokenizer, model


def save_bart() -> None:
    tokenizer, model = load_model("facebook/bart-large")
    tokenizer.save_pretrained("models/bart-large")
    model.save_pretrained("models/bart-large")


def save_rebel() -> None:
    tokenizer, model = load_model("Babelscape/rebel-large")
    tokenizer.save_pretrained("models/rebel-large")
    model.save_pretrained("models/rebel-large")


def main() -> None:
    save_bart()
    save_rebel()


if __name__ == "__main__":
    main()
