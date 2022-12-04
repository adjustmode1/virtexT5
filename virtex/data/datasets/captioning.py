import random
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.data import transforms as T
from .coco_captions import CocoCaptionsDataset
import pickle as pk
from torchtext.data.utils import get_tokenizer


# SPECIALS2IDX = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
class CaptioningDataset(Dataset):
    r"""
    A dataset which provides image-caption (forward and backward) pairs from
    a COCO Captions annotation file. This is used for pretraining tasks which
    use captions - bicaptioning, forward captioning and token classification.

    Args:
        data_root: Path to dataset directory containing images and annotations.
        split: Name of COCO 2017 split to read. One of ``{"train", "val"}``.
        tokenizer: Tokenizer which maps word tokens to their integer IDs.
        image_transform: List of image transformations, from either
            `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
            or :mod:`virtex.data.transforms`.
        max_caption_length: Maximum number of tokens to keep in caption tokens.
            Extra tokens will be trimmed from the right end of the token list.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: SentencePieceBPETokenizer,
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 64,
    ):
        self._dset = CocoCaptionsDataset(data_root, split) # load image 
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_caption_length = max_caption_length

    def __len__(self):
        return len(self._dset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        # keys: {"image_id", "image", "captions"}
        instance = self._dset[idx]
        image_id, image, captions = (
            instance["image_id"],
            instance["image"],
            instance["captions"],
        )
        caption = random.choice(captions)
        # Transform image-caption pair and convert image from HWC to CHW format.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.
        image_caption = self.image_transform(image=image, caption=caption) # prepare imgae
        image, caption = image_caption["image"], image_caption["caption"]
        image = np.transpose(image, (2, 0, 1))

        caption_tokens = self.tokenizer.encode(caption) # vị trí

        return {
            "image_id": torch.tensor(image_id,dtype=torch.float),
            "image": torch.tensor(image, dtype=torch.float),
            "caption_tokens": torch.tensor(caption_tokens, dtype=torch.long),
            "noitpac_tokens": torch.tensor(caption_tokens, dtype=torch.long).flip(0), # đảo lộn
            "caption_lengths": torch.tensor(len(caption_tokens), dtype=torch.long),
        }

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        # Pad `caption_tokens` and `masked_labels` up to this length.
        caption_tokens = torch.nn.utils.rnn.pad_sequence(
            [d["caption_tokens"] for d in data],
            batch_first=True,
            padding_value=1,
        )
        noitpac_tokens = torch.nn.utils.rnn.pad_sequence(
            [d["noitpac_tokens"] for d in data],
            batch_first=True,
            padding_value=1,
        )
        return {
            "image_id": torch.stack([d["image_id"] for d in data], dim=0),
            "image": torch.stack([d["image"] for d in data], dim=0),
            "caption_tokens": caption_tokens,
            "noitpac_tokens": noitpac_tokens,
            "caption_lengths": torch.stack([d["caption_lengths"] for d in data]),
        }
class CaptioningDatasetTraining(Dataset):
    r"""
    A dataset which provides image-caption (forward and backward) pairs from
    a COCO Captions annotation file. This is used for pretraining tasks which
    use captions - bicaptioning, forward captioning and token classification.

    Args:
        data_root: Path to dataset directory containing images and annotations.
        split: Name of COCO 2017 split to read. One of ``{"train", "val"}``.
        tokenizer: Tokenizer which maps word tokens to their integer IDs.
        image_transform: List of image transformations, from either
            `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
            or :mod:`virtex.data.transforms`.
        max_caption_length: Maximum number of tokens to keep in caption tokens.
            Extra tokens will be trimmed from the right end of the token list.
    """

    def __init__(
        self,
        id_arr:any,
        features:any,
        captions:any,
        padding_idx:int,
    ):
        self.id_arr = id_arr
        self.feature_arr = features
        self.captions_arr = captions
        self.padding_idx = padding_idx

    def __len__(self):
        return len(self.id_arr)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # keys: {"image_id", "image", "captions"}
        image_id = self.id_arr[idx]
        feature = self.feature_arr[image_id]
        caption_tokens = self.captions_arr[image_id]

        return {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "image": feature,
            "caption_tokens": caption_tokens,
            "noitpac_tokens": caption_tokens.flip(0), # đảo lộn
            "caption_lengths": torch.tensor(len(caption_tokens),dtype=torch.long),
        }
    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # Pad `caption_tokens` and `masked_labels` up to this length.
        caption_tokens = torch.nn.utils.rnn.pad_sequence(
            [d["caption_tokens"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        noitpac_tokens = torch.nn.utils.rnn.pad_sequence(
            [d["noitpac_tokens"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        return {
            "image_id": torch.stack([d["image_id"] for d in data]),
            "image": torch.stack([d["image"] for d in data]),
            "caption_tokens": caption_tokens,
            "noitpac_tokens": noitpac_tokens,
            "caption_lengths": torch.stack([d["caption_lengths"] for d in data]),
        }
