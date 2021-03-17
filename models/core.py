from typing import Tuple, Mapping
from torch import Tensor
from pytorch_lightning import LightningModule


class ObjectDetector(LightningModule):

    def __init__(self, *args, **kwargs):
        super(ObjectDetector, self).__init__(*args, **kwargs)
        pass

    def training_step(
            self,
            batch: Tuple[Tensor, Tensor],
            bath_idx: int,
    ) -> Mapping[str, Tensor]:
        pass

    def validation_step(
            self,
            batch: Tuple[Tensor, Tensor],
            batch_idx: int,
    ) -> Mapping[str, Tensor]:
        pass

    def test_step(
            self,
            batch: Tuple[Tensor, Tensor],
            batch_idx: int,
    ) -> Mapping[str, Tensor]:
        pass
