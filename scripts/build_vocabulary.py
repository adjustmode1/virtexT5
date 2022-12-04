import argparse
import json
import os
import tempfile
import unicodedata
from typing import List

import sentencepiece as sp
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator 
import itertools as it
from virtex.utils.common import common_parser
from virtex.config import Config
from rich.progress import track 
from virtex.utils.common import common_parser, common_setup, cycle

import pickle as pk
# fmt: off
parser = argparse.ArgumentParser(
    description="""Build a vocabulary out of captions corpus. This vocabulary
    would be a file which our tokenizer can understand.
    """
)
parser.add_argument(
    "-c", "--captions", default="datasets/coco/annotations/captions_train2017.json",
    help="Path to caption annotations file in COCO format.",
)
parser.add_argument(
    "-s", "--vocab-size", type=int, default=10000,
    help="Total desired size of our vocabulary.",
)
parser.add_argument(
    "-o", "--output-prefix", default="datasets/vocab/coco_10k",
    help="Prefix of the files to be saved. Two files will be saved: "
    "[prefix].model and [prefix].vocab",
)
parser.add_argument(
    "-l", "--do-lower-case", action="store_true",
    help="Whether to lower case the captions before forming vocabulary.",
)
parser.add_argument(
    "-a", "--keep-accents", action="store_true",
    help="Whether to keep accents before forming vocabulary (dropped by default).",
)
# fmt: on


def _read_captions(annotations_path: str) -> List[str]:
    r"""
    Given a path to annotation file, read it and return a list of captions.
    These are not processed by any means, returned from the file as-is.

    Args:
        annotations_path: Path to an annotations file containing captions.

    Returns:
        List of captions from this annotation file.
    """

    _annotations = json.load(open(annotations_path))

    captions: List[str] = []
    for ann in _annotations["annotations"]:
        captions.append(ann["caption"])

    return captions
def yield_tokens(data_iter, tokenizer):
  for sample in track(data_iter, description=f'tokenization process'):
      sample = sample.strip().lower()  # remove trailing keys and lowercase 
      yield tokenizer(sample)

def make_vocab(data_iter, tokenizer, map_specials2idx):
    vocab = build_vocab_from_iterator(
        iterator=yield_tokens(data_iter, tokenizer), 
        specials=list(map_specials2idx.keys()), 
        min_freq=1, 
        special_first=True
    )
    vocab.set_default_index(map_specials2idx['<unk>'])  # index of the <unk> token 
    return vocab 
def main(_A: argparse.Namespace):
    _A = parser.parse_args()
    captions: List[str] = _read_captions(_A.captions)
    # Lower case the captions and remove accents according to arguments.
    for i, caption in enumerate(captions):
        caption = caption.lower() if _A.do_lower_case else caption

        captions[i] = caption

    # Create a temporary directory and dump the captions corpus as a text file
    # with one caption per line. That's how sentencepiece wants its input.
    tmpdir_path = tempfile.mkdtemp()
    with open(os.path.join(tmpdir_path, "captions.txt"), "w") as captions_file:
        for caption in captions:
            captions_file.write(caption + "\n")
    captions = list(captions)
    print(captions)
    SPECIALS2IDX = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    tokenizer = get_tokenizer(tokenizer="spacy", language="en_core_web_sm")
    vocabulary = make_vocab(captions, tokenizer, SPECIALS2IDX) # tạo bộ tách vocabulary
    print('vocaulary was built')
    with open(_A.output_prefix, mode='wb') as fp:
        pk.dump(vocabulary, fp)
if __name__ == "__main__":
    _A = parser.parse_args()

    # No distributed training here, just a single process.
    main(_A)
