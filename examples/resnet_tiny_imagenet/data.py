# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import sys
from typing import Any, Callable, List, Optional

import torch
from composer.core import DataSpec
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import dist
from torch.utils.data import DataLoader
from torchvision import transforms

from sunyata.pytorch.data.tiny_imagenet import TinyImageNet

# Scale by 255 since the collate `pil_image_collate` results in images in range 0-255
# If using ToTensor() and the default collate, remove the scaling by 255
IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def build_imagenet_dataspec(
    data_path: str,
    is_streaming: bool,
    batch_size: int,
    local: Optional[str] = None,
    is_train: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    resize_size: int = -1,
    crop_size: int = 224,
    **dataloader_kwargs,
) -> DataSpec:
    """Builds an ImageNet dataloader for either local or remote data.

    Args:
        data_path (str): Path to the dataset either stored locally or remotely (e.g. in a S3 bucket).
        is_streaming (bool): Whether or not the data is stored locally or remotely (e.g. in a S3 bucket).
        batch_size (int): Batch size per device.
        local (str, optional): If using streaming, local filesystem directory where dataset is cached during operation.
            Default: ``None``.
        is_train (bool, optional): Whether to load the training data or validation data. Default:
            ``True``.
        drop_last (bool, optional): whether to drop last samples. Default: ``True``.
        shuffle (bool, optional): whether to shuffle the dataset. Default: ``True``.
        resize_size (int, optional): The resize size to use. Use ``-1`` to not resize. Default: ``-1``.
        crop_size (int, optional): The crop size to use. Default: ``224``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """

    split = 'train' if is_train else 'val'
    transform: List[torch.nn.Module] = []

    # Add split specific transformations
    if is_train:
        transform += [
            transforms.RandomHorizontalFlip()
        ]

    transform = transforms.Compose(transform)

    device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                          std=IMAGENET_CHANNEL_STD)
    dataset = TinyImageNet(root=data_path, split=split, transform=transform, download=True)
    sampler = dist.get_sampler(dataset,
                                drop_last=drop_last,
                                shuffle=shuffle)

    # DataSpec allows for on-gpu transformations, slightly relieving dataloader bottleneck
    return DataSpec(
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=pil_image_collate,
            **dataloader_kwargs,
        ),
        device_transforms=device_transform_fn,
    )


def check_dataloader():
    """Tests if your dataloader is working locally.

    Run `python data.py my_data_path` to test a local dataset. Run `python
    data.py s3://my-bucket/my-dir/data /tmp/path/to/local` to test streaming.
    """
    data_path = sys.argv[1]
    batch_size = 2
    local = None
    is_streaming = len(sys.argv) > 2
    if is_streaming:
        local = sys.argv[2]

    dataspec = build_imagenet_dataspec(data_path=data_path,
                                       is_streaming=is_streaming,
                                       batch_size=batch_size,
                                       local=local)
    print('Running 5 batchs of dataloader')
    for batch_ix, batch in enumerate(itertools.islice(dataspec.dataloader, 5)):
        print(
            f'Batch id: {batch_ix}; Image batch shape: {batch[0].shape}; Target batch shape: {batch[1].shape}'
        )


if __name__ == '__main__':
    check_dataloader()
