from utils.boxes import to_corner_parametrization, is_valid_box
import torch
import pytest


def test_to_corner_parametrization():
    #  two such boxes
    # [[1 1 1]  [[0 0 0]
    #  [1 0 1]   [0 1 1]
    #  [1 1 1]]  [0 1 1]]

    center_boxes = torch.tensor([
        [1, 1, 2, 2],
        [1.5, 1.5, 1, 1]
    ], dtype=torch.float)

    corner_boxes = torch.tensor([
        [0, 0, 2, 2],
        [1, 1, 2, 2]
    ], dtype=torch.float)

    assert torch.allclose(to_corner_parametrization(center_boxes), corner_boxes)


def test_is_valid_box():
    # two such boxes
    # [[1 1 0]  [[0 0 0]
    #  [1 1 0]   [0 1 1]
    #  [0 0 0]]  [0 1 1]]

    boxes = torch.tensor([
        [0, 0, 1, 1],
        [1, 1, 2, 2]
    ], dtype=torch.float)

    assert is_valid_box(boxes)

    # two such boxes
    # [[r 1 0]  [[0 0 0]
    #  [1 l 0]   [0 1 1] 1
    #  [0 0 0]]  [0 1 0]]1
    #               1 1  1

    image_shape = (3, 3)
    boxes = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 3.1, 3.1]
    ], dtype=torch.float)

    with pytest.raises(AssertionError):
        assert is_valid_box(boxes, image_shape)
