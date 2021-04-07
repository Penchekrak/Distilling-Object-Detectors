from typing import Tuple, Mapping, List, Any, Optional, Union

from hydra.utils import instantiate
from torch import Tensor
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection, Metric
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

from src.utils import WandbImageLogger
from omegaconf import DictConfig


class TorchvisionRCNNTask(LightningModule):
    def __init__(
            self,
            model: DictConfig,
            optimizer: DictConfig,
            metric: Optional[Union[List[Metric], Metric, MetricCollection]] = None,
            *args,
            **kwargs
    ):
        super(TorchvisionRCNNTask, self).__init__(*args, **kwargs)
        self.model: GeneralizedRCNN = instantiate(model)
        self.optimizer_cfg = optimizer
        if isinstance(metric, list):
            self.metric = MetricCollection(metrics=list(map(instantiate, metric)))
        elif metric is not None:
            self.metric = instantiate(metric)
        else:
            self.metric = None

    def setup(self, stage) -> None:
        class_to_label = self.trainer.datamodule.train_dataset.class_to_label
        label_to_class = {v: k for k, v in class_to_label.items()}
        self.image_helper = WandbImageLogger(label_to_class)

    def configure_optimizers(self):
        return instantiate(self.optimizer_cfg, params=self.model.parameters())

    def forward(
            self,
            *args,
            **kwargs
    ):
        return self.model(*args, **kwargs)

    def training_step(
            self,
            batch: Tuple[Tensor, Tensor],
            bath_idx: int,
    ) -> Mapping[str, Tensor]:
        images, targets = batch
        loss_dict = self.forward(images, targets)
        loss = sum(loss_value for loss_value in loss_dict.values())
        self.log_dict(loss_dict)
        return loss

    def validation_step(
            self,
            batch: Tuple[Tensor, Tensor],
            batch_idx: int,
    ) -> Any:
        images, targets = batch
        outputs = self.forward(images)
        if self.metric is not None:
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
        if self.metric is not None:
            self.metric(outputs, targets)
        self.image_helper(images, {'ground truth': targets, 'prediction': outputs})

    def test_epoch_end(self, outputs: List[Any]) -> None:
        metric_value = self.metric
        if isinstance(metric_value, dict):
            self.log_dict(metric_value)
        else:
            self.log(self.metric.__class__.__name__, metric_value)
        self.image_helper.log_to(self.logger.experiment)
