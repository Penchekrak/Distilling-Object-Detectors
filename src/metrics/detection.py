from typing import List, Dict, Optional, Union

import torch
from pytorch_lightning.metrics import Metric, AveragePrecision
from torch import Tensor

from src.utils.boxes import to_corner_parametrization


def bounding_box_iou(
        prediction_boxes: Tensor,
        ground_truth_boxes: Tensor,
        parametrization: str = 'corners'
) -> Tensor:
    """
    Calculates IoU for a batch of bounding boxes
    :param prediction_boxes: tensor [N, 4] of predicted bounding boxes
    :param ground_truth_boxes: tensor [M, 4] of ground truth bounding boxes
    :param parametrization: string, indicating bbox parametrization, should be one of ['corners', 'centers']
    :return: tensor [N, M] of pair-wise IoU scores
    """
    if parametrization == 'centers':
        prediction_boxes = to_corner_parametrization(prediction_boxes)
        ground_truth_boxes = to_corner_parametrization(ground_truth_boxes)
    elif parametrization == 'corners':
        pass
    else:
        raise ValueError(f"parametrization should be one of ['corners', 'centers'], got {parametrization} instead")
    prediction_batch_size, _ = prediction_boxes.shape
    ground_truth_batch_size, _ = ground_truth_boxes.shape
    prediction_boxes = prediction_boxes.unsqueeze(0).expand(ground_truth_batch_size, -1, -1)
    ground_truth_boxes = ground_truth_boxes.unsqueeze(1).expand(-1, prediction_batch_size, -1)
    prediction_top_left_x, prediction_top_left_y, prediction_bottom_right_x, prediction_bottom_right_y = prediction_boxes.T
    ground_truth_top_left_x, ground_truth_top_left_y, ground_truth_bottom_right_x, ground_truth_bottom_right_y = ground_truth_boxes.T
    intersection_top_left_x = torch.maximum(prediction_top_left_x, ground_truth_top_left_x)
    intersection_top_left_y = torch.maximum(prediction_top_left_y, ground_truth_top_left_y)
    intersection_bottom_right_x = torch.minimum(prediction_bottom_right_x, ground_truth_bottom_right_x)
    intersection_bottom_right_y = torch.minimum(prediction_bottom_right_y, ground_truth_bottom_right_y)

    intersection_area = torch.maximum(torch.zeros_like(intersection_bottom_right_x),
                                      intersection_bottom_right_x - intersection_top_left_x) * \
                        torch.maximum(torch.zeros_like(intersection_bottom_right_y),
                                      intersection_bottom_right_y - intersection_top_left_y)

    prediction_boxes_area = (prediction_bottom_right_x - prediction_top_left_x) * (
            prediction_bottom_right_y - prediction_top_left_y)
    ground_truth_boxes_area = (ground_truth_bottom_right_x - ground_truth_top_left_x) * (
            ground_truth_bottom_right_y - ground_truth_top_left_y)
    union_area = prediction_boxes_area + ground_truth_boxes_area - intersection_area

    return intersection_area / union_area


class MeanBBoxIoUCorrect(Metric):
    """
    Calculates average IoU score of best-fit boxes given ground truth
    ! DOES NOT ACCOUNT CONFIDENCE !
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # MeanBBoxIoUCorrect
        self.add_state('best_boxes_iou', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_boxes', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(
            self,
            predictions: List[Dict[str, Tensor]],
            targets: List[Dict[str, Tensor]]
    ) -> None:
        for prediction, ground_truth in zip(predictions, targets):
            boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']
            true_boxes, true_labels = ground_truth['boxes'], ground_truth['labels']
            iou_scores = bounding_box_iou(boxes, true_boxes)
            best_iou_scores, indices = torch.max(iou_scores, dim=1)
            correct_predictions = labels[indices] == true_labels
            self.best_boxes_iou += torch.sum(best_iou_scores[correct_predictions])
            self.num_boxes += correct_predictions.sum()

    def compute(self):
        if self.num_boxes == 0:
            return torch.tensor(0.0)
        return self.best_boxes_iou / self.num_boxes


class ExactMAPAtThreshold(AveragePrecision):
    """
    Computes exact MAP@t
    """

    def __init__(
            self,
            num_classes: int,
            iou_threshold: float = 0.5,
            non_maximum_suppression: str = 'softnms',
            *args, **kwargs
    ):
        super().__init__(num_classes=num_classes, *args, **kwargs)
        self.iou_threshold = torch.tensor(iou_threshold)
        self.nms = non_maximum_suppression

    def update(
            self,
            predictions: List[Dict[str, Tensor]],
            targets: List[Dict[str, Tensor]],
    ) -> None:
        predictions_ = []
        targets_ = []

        for prediction, ground_truth in zip(predictions, targets):
            # TODO nms
            boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']
            true_boxes, true_labels = ground_truth['boxes'], ground_truth['labels']
            n, = true_labels.shape
            filter_prediction = torch.zeros((n, self.num_classes), dtype=torch.float).type_as(scores)
            if len(boxes) > 0:
                iou_scores = bounding_box_iou(boxes, true_boxes)
                best_iou_scores, indices = torch.max(iou_scores, dim=1)
                mask = best_iou_scores > self.iou_threshold
                indices = indices[mask]
                filter_prediction[indices, labels[mask]] = scores[mask]
            predictions_.append(filter_prediction)
            targets_.append(true_labels)
        predictions_ = torch.cat(predictions_)
        targets_ = torch.cat(targets_)
        super().update(predictions_, targets_)

    def compute(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        average_precision_per_class = super().compute()

        return sum(ap for ap in average_precision_per_class if not ap.isnan()) / \
               sum(1 for ap in average_precision_per_class if not ap.isnan())


class FastMAPAtThreshold(Metric):
    """
    MAP@t
    """

    # TODO
    def __init__(self, n_classes: int,
                 iou_threshold: float = 0.5,
                 non_maximum_suppression: str = 'softnms',
                 recall_segments: Optional[List[float]] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.iou_threshold = iou_threshold
        self.nms = non_maximum_suppression
        if recall_segments is None:
            self.recall_segments = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        else:
            self.recall_segments = recall_segments

    def update(self) -> None:
        pass

    def compute(self):
        pass


class FastMAP(Metric):
    """
    Averages MAPs over thresholds
    """

    # TODO
    def __init__(self, n_classes: int,
                 iou_thresholds: Optional[List[float]] = None,
                 non_maximum_suppression: str = 'softnms',
                 recall_segments: Optional[List[float]] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        if iou_thresholds is None:
            self.iou_threshold = [0.25, 0.5, 0.75]
        else:
            self.iou_thresholds = iou_thresholds
        self.nms = non_maximum_suppression
        if recall_segments is None:
            self.recall_segments = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        else:
            self.iou_thresholds = iou_thresholds

    def update(self) -> None:
        pass

    def compute(self):
        pass
