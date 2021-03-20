from abc import abstractmethod
from typing import Tuple, Mapping, List, Any
from torch import Tensor
from pytorch_lightning import LightningModule
from metrics import MeanAveragePrecision

class ObjectDetector(LightningModule):

    def __init__(self, *args, **kwargs):
        super(ObjectDetector, self).__init__(*args, **kwargs)
        self.map = MeanAveragePrecision(**kwargs)
        self.map_per_class = MAP()

    @abstractmethod
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
        images, targets = batch
        outputs = self.forward(images)
        self.map(outputs, targets)

    def test_step(
            self,
            batch: Tuple[Tensor, Tensor],
            batch_idx: int,
    ) -> Mapping[str, Tensor]:
        images, targets = batch
        outputs = self.forward(images)
        self.map(outputs, targets)
        return
        # TODO INFERENCE

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict({'map': self.map})
