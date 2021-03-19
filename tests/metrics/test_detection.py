import pytest
import torch

from metrics.detection import bounding_box_iou


def test_bounding_box_iou():
    # iou of overlapping boxes
    # [[┌───┐  ]
    #  [│ ┌─┼─┐]  intersection area = 1
    #  [└─┼─┘ │]  union area = 7
    #  [  └───┘]]

    box1 = torch.tensor([[0, 0, 2, 2]], dtype=torch.float)  # area = 4
    box2 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float)  # area = 4

    assert torch.allclose(bounding_box_iou(box1, box2), torch.tensor(1 / 7))

    # iou of non-overlapping boxes
    # [[┌───┐  ]
    #  [└───┘  ]  intersection area = 0
    #  [  ┌───┐]  union area = 4
    #  [  └───┘]]

    box1 = torch.tensor([[0, 0, 2, 1]], dtype=torch.float)  # area = 2
    box2 = torch.tensor([[1, 2, 3, 3]], dtype=torch.float)  # area = 2

    assert torch.allclose(bounding_box_iou(box1, box2), torch.tensor(0.0))
