# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.loss import binary_cross_entropy_with_logits, soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models import ComposerClassifier
from torchmetrics import Accuracy, MetricCollection

from sunyata.pytorch.arch.base import Residual
from sunyata.pytorch_lightning.base import BaseModule
from sunyata.pytorch.arch.foldnet import FoldNet, FoldNetCfg, FoldNetRepeat, FoldNetRepeat2, Block2
#from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration


def nll_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = torch.argmax(target, dim=1)
    return F.nll_loss(input, target)
    # return - (input * target).mean()


def build_composer_resnet(
    *,
    model_name: str = 'foldnet',
    loss_name: str = "nll_loss",
    block=Block2,
    hidden_dim: int,
    kernel_size: int,
    fold_num: int,
    patch_size: int,
    num_layers: int,
    num_classes: int = 12   
):
    """Helper function to build a Composer ResNet model.

    Args:
        num_classes (int, optional): Number of classes in the classification task. Default: ``1000``.
    """
    if model_name == 'foldnet':
        model = FoldNetRepeat(FoldNetCfg(block=Block2,hidden_dim=hidden_dim,kernel_size=kernel_size,fold_num=fold_num, patch_size=patch_size, num_layers=num_classes, num_classes=num_classes))
    # elif model_name == 'convmixer-bayes':
    #     model = BayesConvMixer(hidden_dim, kernel_size, patch_size, num_layers, num_classes)
    else:
        raise ValueError("Only support convmixer and convmixer-bayes till now.")

    # Specify model initialization
    def weight_init(w: torch.nn.Module):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)
        if isinstance(w, torch.nn.BatchNorm2d):
            w.weight.data = torch.rand(w.weight.data.shape)
            w.bias.data = torch.zeros_like(w.bias.data)

    model.apply(weight_init)

    # Performance metrics to log other than training loss
    train_metrics = Accuracy()
    val_metrics = MetricCollection([CrossEntropy(), Accuracy()])

    # Choose loss function: either cross entropy or binary cross entropy
    if loss_name == 'cross_entropy':
        loss_fn = soft_cross_entropy
    elif loss_name == 'binary_cross_entropy':
        loss_fn = binary_cross_entropy_with_logits
    else:
        raise ValueError(
            f"loss_name='{loss_name}' but must be either ['cross_entropy', 'binary_cross_entropy']"
        )

    # Wrapper function to convert a image classification PyTorch model into a Composer model
    composer_model = ComposerClassifier(model,
                                        train_metrics=train_metrics,
                                        val_metrics=val_metrics,
                                        loss_fn=loss_fn)
    return composer_model
