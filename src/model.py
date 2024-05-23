import math

import transformers  # type: ignore
from transformers.models.bart import modeling_bart  # type: ignore
from transformers import modeling_outputs  # type: ignore

from torch import nn


class ModelConfig(transformers.BartConfig):
    def __init__(
        self,
        *,
        decoder_pad_token_id=1,
        decoder_vocab_size=50265,
        before_decoder_layers=3,
        after_decoder_layer=3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.decoder_pad_token_id = decoder_pad_token_id
        self.decoder_vocab_size = decoder_vocab_size
        self.before_decoder_layers = before_decoder_layers
        self.after_decoder_layer = after_decoder_layer


class InnerModel(transformers.BartModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.before_decoder_layers = config.before_decoder_layers
        self.after_decoder_layer = config.after_decoder_layer

        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        padding_idx, vocab_size = config.decoder_pad_token_id, config.decoder_vocab_size
        self.decoder.embed_tokens = modeling_bart.BartScaledWordEmbedding(
            vocab_size, config.d_model, padding_idx, embed_scale=embed_scale
        )

        self.decoder.layers.extend(
            modeling_bart.BartDecoderLayer(config)
            for _ in range(self.before_decoder_layers + self.after_decoder_layer)
        )

        self.post_init()


class Model(transformers.BartForConditionalGeneration):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.model = InnerModel(config)
        self.lm_head = nn.Linear(
            config.d_model, self.model.decoder.embed_tokens.num_embeddings, bias=False
        )

        self.post_init()
