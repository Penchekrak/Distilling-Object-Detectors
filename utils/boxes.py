from typing import Optional, Tuple

import torch
from torch import Tensor


def is_valid_box(
        bounding_box: Tensor,
        image_shape: Optional[Tuple[int, int]] = None
) -> bool:
    """
    :param bounding_box: tensor [N, 4] of bounding box parameters in the form
    [x-coordinate of top left corner, y-coordinate of top left corner,
     x-coordinate of bottom right corner, y-coordinate of bottom right corner]
    :param image_shape: image shape
    :return: whether box is valid or not
    """
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bounding_box.T
    assert torch.all(top_left_x >= 0)
    assert torch.all(top_left_y >= 0)
    assert torch.all(bottom_right_x > top_left_x)
    assert torch.all(bottom_right_y > top_left_y)
    if image_shape is not None:
        width, height = image_shape
        assert torch.all(bottom_right_x <= width)
        assert torch.all(bottom_right_y <= height)
    return True


def to_corner_parametrization(
        bounding_box: Tensor
) -> Tensor:
    """
    Converts from center/side-lengths parametrization to corners coordinates
    :param bounding_box: tensor [N, 4] of bounding boxes parameters in the form
    [x-coordinate of center, y-coordinate of center, width, height]
    :return: tensor [N, 4] of bounding box parameters in the form
    [x-coordinate of top left corner, y-coordinate of top left corner,
     x-coordinate of bottom right corner, y-coordinate of bottom right corner]
    """
    center_x, center_y, width, height = bounding_box.T
    half_w, half_h = width / 2, height / 2
    top_left_x, top_left_y = center_x - half_w, center_y - half_h
    bottom_right_x, bottom_right_y = center_x + half_w, center_y + half_h
    return torch.column_stack((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
