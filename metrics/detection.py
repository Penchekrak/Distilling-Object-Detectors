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
    :param ground_truth_boxes: tensor [N, 4] of ground truth bounding boxes
    :param parametrization: string, indicating bbox parametrization, should be one of ['corners', 'centers']
    :return: tensor [N] of IoU scores
    """
    if parametrization == 'centers':
        prediction_boxes = to_corner_parametrization(prediction_boxes)
        ground_truth_boxes = to_corner_parametrization(ground_truth_boxes)
    elif parametrization == 'corners':
        pass
    else:
        raise ValueError(f"parametrization should be one of ['corners', 'centers'], got {parametrization} instead")
    prediction_bottom_left_x, prediction_bottom_left_y, prediction_top_right_x, prediction_top_right_y = prediction_boxes.T
    ground_truth_bottom_left_x, ground_truth_bottom_left_y, ground_truth_top_right_x, ground_truth_top_right_y = ground_truth_boxes.T
    intersection_bottom_left_x = torch.maximum(prediction_bottom_left_x, ground_truth_bottom_left_x)
    intersection_bottom_left_y = torch.maximum(prediction_bottom_left_y, ground_truth_bottom_left_x)
    intersection_top_right_x = torch.minimum(prediction_top_right_x, ground_truth_top_right_x)
    intersection_top_right_y = torch.minimum(prediction_top_right_y, ground_truth_top_right_y)

    intersection_area = torch.maximum(torch.zeros_like(intersection_top_right_x),
                                      intersection_top_right_x - intersection_bottom_left_x) * \
                        torch.maximum(torch.zeros_like(intersection_top_right_y),
                                      intersection_top_right_y - intersection_bottom_left_y)

    prediction_boxes_area = (prediction_top_right_x - prediction_bottom_left_x) * (
            prediction_top_right_y - prediction_bottom_left_y)
    ground_truth_boxes_area = (ground_truth_top_right_x - ground_truth_bottom_left_x) * (
                ground_truth_top_right_y - ground_truth_bottom_left_y)
    union_area = prediction_boxes_area + ground_truth_boxes_area - intersection_area

    return intersection_area / union_area


class MeanBoxIoU(Metric):

    def __init__(
            self,
            parametrization: str = 'corners',
            *args, **kwargs
    ):
        super(MeanBoxIoU, self).__init__(*args, **kwargs)
        self.parametrization = parametrization

    def update(
            self,
    ):
        pass


class MeanAveragePrecision(Metric):
    def __init__(self, *args, **kwargs):
        super(MeanAveragePrecision, self).__init__(*args, **kwargs)
        pass

    def update(self) -> None:
        pass
