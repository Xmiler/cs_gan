from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl


class LitData(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self._batch_size = batch_size
        self._dims = (1, 28, 28)
        self._data_dir = Path(__file__).parent / 'data'
        assert self._data_dir.is_dir()
        self._dataset_train = None

    @property
    def dims(self):
        return self._dims

    def prepare_data(self):
        MNIST(self._data_dir.as_posix(), train=True, download=True)

    def setup(self, stage: str):
        assert stage == 'fit'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self._dataset_train = MNIST(self._data_dir.as_posix(), train=True, download=False,
                                    transform=transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_train, batch_size=self._batch_size, shuffle=True, num_workers=4)
