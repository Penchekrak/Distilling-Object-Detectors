from src.utils.boxes import to_corner_parametrization, is_valid_box
import torch
import pytest


def test_to_corner_parametrization():
    #  two such boxes
    #  ┌───┐
    #  │ ┌─┤
    #  └─┴─┘

    corner_boxes = torch.tensor([
        [0, 0, 2, 2],
        [1, 1, 2, 2]
    ], dtype=torch.float)

    center_boxes = torch.tensor([
        [1, 1, 2, 2],
        [1.5, 1.5, 1, 1]
    ], dtype=torch.float)

    assert torch.allclose(to_corner_parametrization(center_boxes), corner_boxes)


def test_is_valid_box():
    # two valid boxes
    #  ┌─┐
    #  └─┼─┐
    #    └─┘

    boxes = torch.tensor([
        [0, 0, 1, 1],
        [1, 1, 2, 2]
    ], dtype=torch.float)

    assert is_valid_box(boxes)

    # two invalid boxes
    #  ┘─┐    ┏━━━┓
    #  └─┌    ┃ ┌─╂─┐
    #         ┗━┿━┛ │
    #           └───┘

    image_shape = (3, 3)
    boxes = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 4, 4]
    ], dtype=torch.float)

    with pytest.raises(AssertionError):
        assert is_valid_box(boxes, image_shape)
