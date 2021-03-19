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

    box3 = torch.tensor([[0, 0, 2, 1]], dtype=torch.float)  # area = 2
    box4 = torch.tensor([[1, 2, 3, 3]], dtype=torch.float)  # area = 2

    assert torch.allclose(bounding_box_iou(box3, box4), torch.tensor(0.0))

    # iou of several predictions with gt
    # [[┌───┐  ]
    #  [└───┘  ]
    #  [┌─╔╤══╗]
    #  [└─╚╧══╝]]

    pbox = torch.tensor([[0, 0, 2, 1], [0, 2, 2, 3]], dtype=torch.float)
    gtbox = torch.tensor([[1, 2, 4, 3]], dtype=torch.float)

    assert torch.allclose(bounding_box_iou(pbox, gtbox), torch.tensor([[0.0], [1 / 4]]))