from typing import List, Dict

import torch
from torchmetrics import Metric
from torch import Tensor

from utils.boxes import to_corner_parametrization


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


class MeanAveragePrecision(Metric):
    def __init__(
            self,
            iou_threshold: float = 0.5,
            non_maximum_suppression: bool = True,
            *args, **kwargs
    ):
        super(MeanAveragePrecision, self).__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold
        self.nms = non_maximum_suppression

    def update(
            self,
            predictions: List[Dict[str, Tensor]],
            targets: List[Dict[str, Tensor]]
    ) -> None:
        for prediction, ground_truth in zip(predictions, targets):
            if self.nms:
                prediction = prediction
            boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']
            true_boxes, true_labels = ground_truth['boxes'], ground_truth['labels']
            iou_scores = bounding_box_iou(boxes, true_boxes)
        pass



