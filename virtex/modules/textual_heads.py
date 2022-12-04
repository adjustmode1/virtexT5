r"""
A textual head accepts visual features from the visual backbone, and performs
task specific modeling (captioning, classification etc.) to predict an output
distribution over vocabulary tokens for one or multiple time-steps in the batch.
"""
import functools

import torch
from torch import nn
from typing import Optional

from virtex.modules.embedding import WordAndPositionalEmbedding

from virtex.modules.transformer import *

class TextualHead(nn.Module):
    r"""
    Base class for all textual heads. All child classes can simply inherit
    from :class:`~torch.nn.Module`, however this is kept here for uniform
    type annotations.

    Args:
        visual_feature_size: Size (number of channels) of the input features
            from the visual backbone.
        vocab_size: Number of tokens in the output vocabulary.
        hidden_size: Size of the token embedding vectors, or hidden state vector
            of the language model.
    """

    def __init__(self, visual_feature_size: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.visual_feature_size = visual_feature_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    @property
    def textual_feature_size(self):
        r"""
        Size of the last dimension of output right before the output linear
        layer (which predicts a distribution over vocabulary tokens). This is
        typically same as :attr:`hidden_size` for most modules. This property
        is used to add more modules on top of this.
        """
        return self.hidden_size


class LinearTextualHead(TextualHead):
    r"""
    A textual head containing a single linear layer projecting from the visual
    feature size to the output vocabulary size.

    Args:
        visual_feature_size: Size (number of channels) of the input features from
            the visual backbone.
        vocab_size: Number of tokens in the output vocabulary.
    """

    def __init__(self, visual_feature_size: int, vocab_size: int, **kwargs):
        # For API consistency.
        hidden_size = visual_feature_size
        super().__init__(visual_feature_size, vocab_size, hidden_size)
        self.output = nn.Linear(visual_feature_size, vocab_size)

    def forward(
        self,
        visual_features: torch.Tensor,
        caption_tokens: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Project visual features directly to predict a distribution over
        vocabulary tokens through a single linear layer. This textual head
        ignores arguments ``caption_tokens`` and ``caption_lengths``, they
        are here for API consistency.

        Args:
            visual_features: A tensor of shape ``(batch_size, channels, height,
                width)`` containing features from visual backbone.

        Returns:
            A tensor of shape ``(batch_size, vocab_size)`` containing output
            vocabulary logits.
        """

        # Convert to NHWC and project visual features to textual feature size.
        batch_size, channels, _, _ = visual_features.size()
        visual_features = visual_features.view(batch_size, channels, -1)
        visual_features = visual_features.permute(0, 2, 1)

        # Perform global average pooling of visual features.
        # shape: (batch_size, channels)
        visual_features = visual_features.mean(dim=1)

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self.output(visual_features)
        return output_logits


class TransformerDecoderTextualHead(TextualHead):
    r"""
    A textual head composed of 4 modules chính : (1) input projection (linear
    layer) cho visual features để khớp kích thước với textual features, (2) word
    and positional embedding for input captions, (3) a unidirectional transformer
    decoder, and (4) and output projection (linear layer) to predict a
    distribution over vocabulary tokens. The word embedding weights are tied
    with output projection; the latter still has its own learnable bias.

    .. note::

        For the "bicaptioning" pretraining task, our *textual head* (as defined
        in the paper) must have two transformer decoders: one each to decode
        caption in either direction. This class however will always have one
        transformer per object.

        Refer :class:`~virtex.models.captioning.BidirectionalCaptioningModel`
        source to understand how an object of this class is cloned, along with
        tying embedding and output weights, for bicaptioning.

        Hence, while there are *two objects* of this class, it is pragmatically
        a *single* textual head as a whole, according to the terminology used
        in paper.

    Args:
        visual_feature_size: Size (number of channels) of the input features from
            the visual backbone.
        vocab_size: Number of tokens in the output vocabulary.
        hidden_size: Size of the token embedding vectors, or hidden state vector of
            the language model.
        num_layers: Number of layers in the transformer.
        attention_heads: Number of attention heads in the transformer.
        feedforward_size: Size of feedforward layers in the transformer.
        dropout: Dropout probability for transformer (applied after layernorm).
        norm_first: Whether to apply normalization before or after attention/FF
            layers. The former type are called pre-norm variants (like GPT-2) and
            latter are post-norm variants (like BERT). Default is post-norm.
        mask_future_positions: Whether to mask future positions for self-attention
            over caption tokens. This must be ``True`` for captioning (and
            bicaptioning) tasks to prevent the language model from cheating, and
            ``False`` for masked language modeling, as the self-attention should
            consider all tokens.
        max_caption_length: Maximum length of input captions; this is used to
            create a fixed positional embedding lookup table.
        padding_idx: Token index of ``[PAD]`` token, word embedding for these
            tokens will be a vector of zeroes (and not trainable).
    """

    def __init__(
        self,
        visual_feature_size: int, # _C.MODEL.VISUAL.FEATURE_SIZE sửa lại file virtex/config.py
        vocab_size: int, # 10000
        hidden_size: int, # 1
        num_layers: int, # 1024
        attention_heads: int, # 16
        feedforward_size: int, # 2048
        dropout: float = 0.1,
        norm_first: bool = False,
        mask_future_positions: bool = True, # true
        max_caption_length: int = 64, # 64
        padding_idx: int = 1, # 0
    ):
        super().__init__(visual_feature_size, vocab_size, hidden_size)
        # thay mới
        self.padding_idx = padding_idx
        self.max_caption_length = max_caption_length
        self.vocab_size = vocab_size
        self.attention_heads = attention_heads
        self.feedforward_size = feedforward_size
        self.hidden_size = hidden_size
        self.embedding_scale = np.sqrt(self.hidden_size)
        self.position_encoder = PositionalEncoding(self.max_caption_length, self.hidden_size)
        self.adaptaror = nn.Linear(self.feedforward_size, self.hidden_size)
        self.token_embedder = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx) 
        self.transformer = Transformer(
            in_dim=self.hidden_size,
            ff_dim=512,
            nb_heads=self.attention_heads,
            encoder_depth=num_layers,
            decoder_depth=num_layers,
            pre_norm=False,
        )
        self.generator = nn.Linear(self.hidden_size, self.vocab_size)
        # self.apply(self._init_weights)
        # end thay mới
    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        # src = self.adaptaror(src)  # reduce the dimension for in_dim to hd_dim
        # src = self.position_encoder(src)
        memory = self.transformer.encoder(src)
        return memory 
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt = self.token_embedder(tgt) * self.embedding_scale
        # tgt = self.position_encoder(tgt)
        decoder_input_ids = self.transformer.decoder._shift_right(tgt)
        output = self.transformer.decoder(
          input_ids=decoder_input_ids,
          encoder_hidden_states=memory,
        )
        return output
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.position_encoder(self.adaptaror(src))
        embedded_tgt = self.token_embedder(tgt)
        embedded_tgt = self.position_encoder(embedded_tgt)

        output = self.transformer(
            src=src, 
            tgt=embedded_tgt, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        return self.generator(output[-1])

    @staticmethod
    @functools.lru_cache()
    def make_future_mask(
        size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """
        Generate a mask for "future" positions. Masked positions will be negative
        infinity. This mask is critical for casual language modeling.
        """
        return torch.triu(
            torch.full((size, size), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
