import argparse
from collections import Counter
from typing import Any

from loguru import logger
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# fmt: off
from virtex.config import Config
from virtex.factories import (
    PretrainingDatasetFactory, PretrainingModelFactory, OptimizerFactory,
    LRSchedulerFactory,
)
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser, common_setup, cycle
import virtex.utils.distributed as dist
from virtex.utils.timer import Timer

from rich.progress import track
import pickle as pk 


parser = common_parser(
    description="Train a VirTex model (CNN + Transformer) on COCO Captions."
)
group = parser.add_argument_group("Checkpointing and Logging")
group.add_argument(
    "--resume-from", default=None,
    help="Path to a checkpoint to resume training from (if provided)."
)
group.add_argument(
    "--checkpoint-every", type=int, default=2000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
group.add_argument(
    "--log-every", type=int, default=20,
    help="""Log training curves to tensorboard after every these many iterations
    only master process logs averaged loss values across processes.""",
)
group.add_argument(
    "--savefile", type=str, default=None,
    help="""Log training curves to tensorboard after every these many iterations
    only master process logs averaged loss values across processes.""",
)
# fmt: on


def main(_A: argparse.Namespace):

    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device: Any = torch.device("cpu")
    else:
        # Get the current device as set for current distributed process.
        # Check `launch` function in `virtex.utils.distributed` module.
        device = torch.cuda.current_device()

    # Create a config object (this will be immutable) and perform common setup
    # such as logging and setting up serialization directory.
    
    _C = Config(_A.config, _A.config_override)
    common_setup(_C, _A)

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER, SCHEDULER
    # -------------------------------------------------------------------------
    # giống bên vit
    train_dataset = PretrainingDatasetFactory.from_config(_C, split="train")# giống bên vit
    val_dataset = PretrainingDatasetFactory.from_config(_C, split="val")# giống bên vit
    savefile = _A.savefile
    # Make `DistributedSampler`s to shard datasets across GPU processes.
    # Skip this if training on CPUs.
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)  # type: ignore
        if _A.num_gpus_per_machine > 0
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=True)  # type: ignore
        if _A.num_gpus_per_machine > 0
        else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    ) # giống vit
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=val_sampler is None,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=val_dataset.collate_fn,
    ) # giống vit

    model = PretrainingModelFactory.from_config(_C).to(device)
    # -------------------------------------------------------------------------
    #   EXTRACT FEATURE TRAIN LOOP
    # -------------------------------------------------------------------------
    logger.debug('extraction will start')
    accumulator = []
    image_id_arr = []
    caption_token_arr = []
    arr = [x for x in range(len(train_dataset))]
    for sections in track(train_dataset, 'features train extraction'):
        # sections type tensor
        feature = sections["image"]
        image_id = sections["image_id"]
        caption_token = sections["caption_tokens"]
        embedding = model.visual(feature)
        accumulator.append(embedding)
        image_id_arr.append(image_id)
        caption_token_arr.append(caption_token)

    # -------------------------------------------------------------------------
    #   EXTRACT FEATURE VAL LOOP
    # -------------------------------------------------------------------------
    logger.debug('extraction complete')
    # -------------------------------------------------------------------------
    #   SAVE MODEL + FEATURE
    # ------------------------------------------------------------------------- 
    map_arr_id_train = dict(zip(arr, image_id_arr))
    map_img2features_train = dict(zip(image_id_arr, accumulator))
    map_img2captions_train = dict(zip(image_id_arr, caption_token_arr))
    with open(savefile+'/id_train.pkl', mode='wb') as fp:
        pk.dump(map_arr_id_train, fp)
    with open(savefile+'/features_train.pkl', mode='wb') as fp:
        pk.dump(map_img2features_train, fp)
    with open(savefile+'/captions_train.pkl', mode='wb') as fp:
        pk.dump(map_img2captions_train, fp)
    # -------------------------------------------------------------------------
    #   EXTRACT FEATURE VAL LOOP
    # -------------------------------------------------------------------------
    logger.debug('extraction will start')
    accumulator = []
    image_id_arr = []
    caption_token_arr = []
    arr = [x for x in range(len(val_dataset))]

    for sections in track(val_dataset, 'features val extraction'):
        # sections type tensor
        feature = sections["image"]
        image_id = sections["image_id"]
        caption_token = sections["caption_tokens"]
        embedding = model.visual(feature)
        accumulator.append(embedding)
        image_id_arr.append(image_id)
        caption_token_arr.append(caption_token)


    # -------------------------------------------------------------------------
    #   EXTRACT FEATURE VAL LOOP
    # -------------------------------------------------------------------------
    logger.debug('extraction complete')
    # -------------------------------------------------------------------------
    #   SAVE MODEL + FEATURE
    # ------------------------------------------------------------------------- 
    map_arr_id_train = dict(zip(arr, image_id_arr))
    map_img2features_train = dict(zip(image_id_arr, accumulator))
    map_img2captions_train = dict(zip(image_id_arr, caption_token_arr))
    with open(savefile+'/id_val.pkl', mode='wb') as fp:
        pk.dump(map_arr_id_train, fp)
    with open(savefile+'/features_val.pkl', mode='wb') as fp:
        pk.dump(map_img2features_train, fp)
    with open(savefile+'/captions_val.pkl', mode='wb') as fp:
        pk.dump(map_img2captions_train, fp)

if __name__ == "__main__":
    _A = parser.parse_args()

    if _A.num_gpus_per_machine == 0:
        main(_A)
    else:
        # This will launch `main` and set appropriate CUDA device (GPU ID) as
        # per process (accessed in the beginning of `main`).
        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus_per_machine,
            machine_rank=_A.machine_rank,
            dist_url=_A.dist_url,
            args=(_A, ),
        )
