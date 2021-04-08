from hydra.utils import instantiate
from pytorch_lightning.metrics import MetricCollection

from .boxes import *
from .drawing import *


def make_metric(metric):
    if metric is None:
        return None
    if isinstance(metric, list):
        m = list(map(instantiate, metric))
    elif isinstance(metric, dict):
        m = {name: instantiate(value) for name, value in metric.items()}
    else:
        m = [instantiate(metric)]
    return MetricCollection(metrics=m)
