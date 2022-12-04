from typing import Any, Dict, List

import sentencepiece as sp
from torchtext.data.utils import get_tokenizer
import torch
import pickle as pk
class SentencePieceBPETokenizer:
    r"""
    A tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`_
    with BPE sub-routine. It encodes caption strings into list of tokens.

    Args:
        model_path: Path to the ``.model`` file trained by SentencePiece.
    """
    SP_SPACE = u"▁"

    def __init__(self, model_path: str,sos_id:int,eos_id:int):
        self.model_path = model_path

        # Load pretrained tokenizer model.
        self.model = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
        with open(self.model_path,mode="rb") as fp:
            self.vocab = pk.load(fp)
        self.sos_id = torch.tensor([sos_id])
        self.eos_id = torch.tensor([eos_id])

    def get_vocab_size(self) -> int:
        r"""Return number of tokens in vocabulary (including special tokens)."""
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        r"""Get integer ID of a string token (``<unk>`` if does not exist)."""
        # Since tokenizer uses subword regularization, one token may break down to multiple IDs.
        # Keep trying till we get a single ID.
        return self.vocab(token)

    def id_to_token(self, token_id: int) -> str:
        r"""Get string token of an integer ID (``<unk>`` if does not exist)."""
        return self.model.id_to_piece(token_id)

    def encode(self, text: str) -> List[int]:
        r"""Convert a text string to a list of integer token ids."""
        tok = self.model(text.strip().lower()) # mã hóa caption vd "day là chuổi" => ["day","là","chuỗi"]
        idx = torch.tensor(self.vocab(tok)) # vị trí
        caption_tokens = torch.cat([self.sos_id, idx, self.eos_id]).numpy()
        return caption_tokens

    def decode(self, token_ids: List[int]) -> str:
        r"""Convert a sequence of token IDs to a text string."""
        return self.vocab.lookup_tokens(token_ids)
