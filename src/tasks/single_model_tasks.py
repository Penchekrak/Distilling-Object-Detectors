from abc import abstractmethod
from typing import Tuple, Mapping, List, Any, Optional, Union
from torch import Tensor
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection, Metric
from src.utils import WandbImageLogger


class TorchvisionRCNNTask(LightningModule):
    def __init__(
            self,
            metric: Optional[Union[List[Metric], Metric, MetricCollection]] = None,
            image_helper: Optional[WandbImageLogger] = None,
            *args,
            **kwargs
    ):
        super(TorchvisionRCNNTask, self).__init__(*args, **kwargs)
        self.image_helper = image_helper
        if isinstance(metric, list):
            self.metric = MetricCollection(metric)
        else:
            self.metric = metric

    @abstractmethod
    def forward(
            self,
            *args,
            **kwargs
    ):
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
    ) -> Any:
        images, targets = batch
        outputs = self.forward(images)
        self.metric(outputs, targets)
        self.image_helper(images, {'ground truth': targets, 'prediction': outputs})

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        metric_value = self.metric
        if isinstance(metric_value, dict):
            self.log_dict(metric_value)
        else:
            self.log(self.metric.__class__.__name__, metric_value)
        self.image_helper.log_to(self.logger.experiment)

    def test_step(
            self,
            batch: Tuple[Tensor, Tensor],
            batch_idx: int,
    ) -> Any:
        images, targets = batch
        outputs = self.forward(images)
        self.metric(outputs, targets)
        self.image_helper(images, {'ground truth': targets, 'prediction': outputs})

    def test_epoch_end(self, outputs: List[Any]) -> None:
        metric_value = self.metric
        if isinstance(metric_value, dict):
            self.log_dict(metric_value)
        else:
            self.log(self.metric.__class__.__name__, metric_value)
        self.image_helper.log_to(self.logger.experiment)
