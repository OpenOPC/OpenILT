# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import torch
import torch.utils.data as torchdata

from mtilt.config import configurable
from mtilt.utils.comm import get_world_size
from mtilt.utils.env import seed_all_rng
from mtilt.utils.logger import _log_api_usage

from .catalog import DatasetCatalog
from .common import  DatasetFromList, MapDataset

"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_batch_data_loader", "build_train_loader"
]

def get_dataset_dicts(names):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names

    available_datasets = DatasetCatalog.keys()
    names_set = set(names)
    if not names_set.issubset(available_datasets):
        logger = logging.getLogger(__name__)
        logger.warning(
            "The following dataset names are not registered in the DatasetCatalog: "
            f"{names_set - available_datasets}. "
            f"Available datasets are {available_datasets}"
        )

    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        if len(dataset_dicts) > 1:
            # ConcatDataset does not work for iterable style dataset.
            # We could support concat for iterable as well, but it's often
            # not a good idea to concat iterables anyway.
            return torchdata.ConcatDataset(dataset_dicts)
        return dataset_dicts[0]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts


def build_batch_data_loader(
    dataset,
    total_batch_size,
    *,
    num_workers=0,
    collate_fn=None,
    drop_last: bool = True,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.
        drop_last (bool): if ``True``, the dataloader will drop incomplete batches.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size

    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        worker_init_fn=worker_init_reset_seed,
    )

def _train_loader_from_config(cfg, *, dataset=None):
    if dataset is None:
        assert len(cfg.DATASETS.TRAIN)==1, \
            "now we only support single type benchmark, please mask sure your benchmark only include one instance"
        dataset = get_dataset_dicts(
            cfg.DATASETS.TRAIN
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    return {
        "dataset": dataset,
        "total_batch_size": cfg.TRAINER.IMS_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_train_loader(
    dataset,
    *,
    mapper,
    total_batch_size,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False) # serialize=False

    dataset = MapDataset(dataset, mapper)

    return build_batch_data_loader(
        dataset,
        total_batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)
