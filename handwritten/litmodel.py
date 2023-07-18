from pathlib import Path
import numpy as np
import torch
from torch import nn
import torchvision
import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: tuple[int]):
        super().__init__()
        self._img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self._model = nn.Sequential(
            block(latent_dim, 128, normalize=False),
            block(128, 256),
            block(256, 512),
            block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        img = self._model(z)
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
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        y = self._model(x)
        return y


class LitModel(pl.LightningModule):
    def __init__(
            self,
            img_shape: tuple[int],
            latent_dim: int,
            lr: float,
            adam_betas: tuple[float] = (.5, .999),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self._generator = Generator(self.hparams.latent_dim, self.hparams.img_shape)
        self._discriminator = Discriminator(self.hparams.img_shape)

        self._criterion = nn.BCEWithLogitsLoss()

        self._log_gen_imgs_dir = None

    def on_fit_start(self):
        self._log_gen_imgs_dir = Path(self.logger.log_dir) / 'gen_imgs'
        self._log_gen_imgs_dir.mkdir()

    def forward(self, z: torch.Tensor):
        return self._generator(z)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, _ = batch
        batch_size = x.size(0)

        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        gt_f = torch.zeros(batch_size, 1, device=self.device)
        gt_t = torch.ones(batch_size, 1, device=self.device)

        opt_g, opt_d = self.optimizers()

        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        x_g = self(z)
        torchvision.utils.save_image(
            x_g[:8*8],
            self._log_gen_imgs_dir / f'epoch{self.current_epoch:03d}_{batch_idx:05d}.png'
        )
        y = self._discriminator(x_g)
        loss_g = self._criterion(y, gt_t)
        self.log("loss_g", loss_g, prog_bar=True)
        self.manual_backward(loss_g)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        y_r = self._discriminator(x)
        loss_r = self._criterion(y_r, gt_t)
        y_g = self._discriminator(self(z).detach())
        loss_g = self._criterion(y_g, gt_f)
        loss_d = (loss_r + loss_g) / 2
        self.log("loss_d", loss_d, prog_bar=True)
        self.manual_backward(loss_d)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

    def configure_optimizers(self):
        lr, betas = self.hparams.lr, self.hparams.adam_betas
        opt_g = torch.optim.Adam(self._generator.parameters(), lr=lr, betas=betas)
        opt_d = torch.optim.Adam(self._discriminator.parameters(), lr=lr, betas=betas)
        return [opt_g, opt_d], []
