from typing import Any, Union, List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from torchvision.datasets import VOCDetection


class VOC(LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 32,
            num_workers: int = 1,
            *args, **kwargs
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args
        self.kwargs = kwargs
        super(VOC).__init__(*args, **kwargs)

    def prepare_data(self, *args, **kwargs):
        VOCDetection(root=self.data_dir, download=True, image_set='trainval')

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = VOCDetection(
            root=self.data_dir,
            download=False,
            image_set='train',
            transforms=self.train_transforms
        )
        self.val_dataset = VOCDetection(
            root=self.data_dir,
            download=True,
            image_set='val',
            transforms=self.val_transforms
        )

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        raise AttributeError("No test set for VOC dataset :(")
