from typing import Any, Union, List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class COCO(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super(COCO).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> Any:
        pass

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        pass