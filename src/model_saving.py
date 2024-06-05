from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig  # type: ignore

from src.pl_module import PLModule

MODEL_NAME = "rebel"


def main() -> None:
    config = AutoConfig.from_pretrained("Babelscape/rebel-large")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "Babelscape/rebel-large", config=config
    )

    pl_model = PLModule(config, tokenizer, model)

    pl_model.tokenizer.save_pretrained(f"models/{MODEL_NAME}")
    pl_model.model.save_pretrained(f"models/{MODEL_NAME}")


if __name__ == "__main__":
    main()
