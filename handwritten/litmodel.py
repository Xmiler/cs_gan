import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: tuple[int]):
        super().__init__()
        self._img_shape = img_shape

        def block(in_feat, out_feat):
            return nn.Sequential(
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.ReLU(inplace=True),
            )

        self._model = nn.Sequential(
            block(latent_dim, 128),
            block(128, 256),
            block(256, 512),
            block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        img = self._model(x)
        img = img.view(img.size(0), *self._img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple[int]):
        super().__init__()

        self._model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
