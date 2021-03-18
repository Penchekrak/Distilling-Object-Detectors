import torch
from torch import Tensor



def to_corner_parametrization(
        bounding_box: Tensor
) -> Tensor:
    """
    Converts from center/side-lengths parametrization to corners coordinates
    :param bounding_box: tensor [N, 4] of bounding boxes parameters in the form
    [x-coordinate of center, y-coordinate of center, width, height]
    :return: tensor [N, 4] of bounding box parameters in the form
    [x-coordinate of bottom left corner, y-coordinate of bottom left corner,
     x-coordinate of top right corner, y-coordinate of top right corner]
    """
    center_x, center_y, width, height = bounding_box.T
    half_w, half_h = width / 2, height / 2
    bottom_left_x, bottom_left_y = center_x - half_w, center_y - half_h
    top_right_x, top_right_y = center_x + half_w, center_y + half_h
    return torch.column_stack((bottom_left_x, bottom_left_y, top_right_x, top_right_y))