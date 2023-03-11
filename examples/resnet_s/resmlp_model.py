import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from composer.loss import binary_cross_entropy_with_logits, soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models import ComposerClassifier
from torchmetrics import Accuracy, MetricCollection
from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration


pair = lambda x: x if isinstance(x, tuple) else (x, x)
def nll_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # target = torch.argmax(target, dim=1)
    return F.nll_loss(input, target)
    # return - (input * target).mean()

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b
    

class PreAffinePostLayerScale(nn.Module):
    def __init__(self, 
                 dim, 
                 depth, 
                 fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn =fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x


class DeepBayesInferResMlp(nn.Module):
    def __init__(self,
                 image_size: int ,
                 patch_size: int ,
                 hidden_dim: int,
                #  depth: int = 16,
                 expansion_factor: int ,
                 num_layers: int,
                 num_classes: int ,
                 channels: int = 3,
                 is_bayes: bool = True,
                 is_prior_as_params: bool = False,

                 ):
        super().__init__()
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0,'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        wrapper = lambda i, fn: PreAffinePostLayerScale(hidden_dim, i+1, fn)

        self.layers = nn.ModuleList([
            nn.Sequential(
            wrapper(i, nn.Conv1d(num_patches, num_patches, 1)),
            wrapper(i, nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim)
            ))
            ) for i in range(num_layers)
        ])

        if not is_bayes:
            self.layers = nn.ModuleList([nn.Sequential(*self.layers)])  # to onr layer
        
        self.embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2 ) * channels, hidden_dim)
        )

        self.digup = nn.Sequential(
            Affine(hidden_dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(hidden_dim, num_classes)
        )
        self.is_bayes = is_bayes
        log_prior = torch.zeros(1, num_classes)
        if is_prior_as_params:
            self.log_prior = nn.Parameter(log_prior)
        else:
            self.register_buffer('log_prior', log_prior)

        # self.logits_bias = nn.Parameter(torch.zeros(1, num_classes))
        self.num_classes = num_classes

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        print(x.shape)
        print("aaaaa")
        log_prior = self.log_prior.repeat(batch_size, 1)
        x = self.embed(x)
        print(x.shape)
        print("bbbbb")
        for layer in self.layers:
            x = layer(x)
            print(x.shape)
            print("ccccc")
            logits = self.digup(x)
            log_prior = log_prior + logits
            # log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
            log_prior = F.log_softmax(log_prior, dim=-1)  # log_bayesian_iteration(log_prior, logits)
            log_prior = log_prior + math.log(self.num_classes)
        return log_prior
    

def build_composer_resnet(
        *,
        model_name: str = 'resmlp',
        loss_name: str = 'bll_loss',
        image_size: int ,
        patch_size: int ,
        hidden_dim: int ,
                #  depth: int = 16,
        expansion_factor: int ,
        num_layers: int ,
        num_classes: int ,
        channels: int = 3,
        is_bayes: bool = True,
        is_prior_as_params: bool = False,):
        # model_name: str = 'convmixer',
        # loss_name: str = 'bll_loss',
        # hidden_dim: int,
        # kernel_size: int,
        # patch_size: int,
        # num_layers: int,
        # num_classes: int = 1000):
    """Helper function to build a Composer ResNet model.

    Args:
        model_name (str, optional): Name of the ResNet model to use, either
            ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']. Default: ``'resnet50'``.
        loss_name (str, optional): Name of the loss function to use, either ['cross_entropy', 'binary_cross_entropy'].
            Default: ``'cross_entropy'``.
        num_classes (int, optional): Number of classes in the classification task. Default: ``1000``.
    """
    # model_fn = getattr(resnet, model_name)

    # model = model_fn(num_classes=num_classes, groups=1, width_per_group=64)
    if model_name == 'resmlp':
        model = DeepBayesInferResMlp(image_size, patch_size, hidden_dim, expansion_factor, num_layers, num_classes, channels, is_bayes, is_prior_as_params)
    else:
        raise ValueError("only support mlp")
    

    # Specify model initialization
    def weight_init(w: torch.nn.Module):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)
        if isinstance(w, torch.nn.BatchNorm2d):
            w.weight.data = torch.rand(w.weight.data.shape)
            w.bias.data = torch.zeros_like(w.bias.data)
        # When using binary cross entropy, set the classification layer bias to -log(num_classes)
        # to ensure the initial probabilities are approximately 1 / num_classes
        # if loss_name == 'binary_cross_entropy' and isinstance(
        #         w, torch.nn.Linear):
        #     w.bias.data = torch.ones(
        #         w.bias.shape) * -torch.log(torch.tensor(w.bias.shape[0]))

    model.apply(weight_init)

    # Performance metrics to log other than training loss
    train_metrics = Accuracy()
    val_metrics = MetricCollection([CrossEntropy(), Accuracy()])

    # Choose loss function: either cross entropy or binary cross entropy
    if loss_name == 'cross_entropy':
        loss_fn = soft_cross_entropy
    elif loss_name == 'binary_cross_entropy':
        loss_fn = binary_cross_entropy_with_logits
    elif loss_name == 'nll_loss':
        loss_fn = nll_loss
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