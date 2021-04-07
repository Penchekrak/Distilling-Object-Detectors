import pytest
import torch

from src.metrics import bounding_box_iou, MeanBBoxIoUCorrect, ExactMAPAtThreshold

SAMPLE_BOXES = [
    # iou of overlapping boxes
    #  ┌───┐
    #  │ ┌─┼─┐  intersection area = 1
    #  └─┼─┘ │  union area = 7
    #    └───┘
    (
        torch.tensor([[0, 0, 2, 2]], dtype=torch.float),  # area = 4
        torch.tensor([[1, 1, 3, 3]], dtype=torch.float),  # area = 4
        torch.tensor(1 / 7)  # correct iou
    ),
    # iou of non-overlapping boxes
    #  ┌───┐
    #  └───┘    intersection area = 0
    #    ┌───┐  union area = 4
    #    └───┘
    (
        torch.tensor([[0, 0, 2, 1]], dtype=torch.float),  # area = 2
        torch.tensor([[1, 3, 3, 4]], dtype=torch.float),  # area = 2
        torch.tensor(0.0)  # correct iou
    ),
    # iou of several predictions with gt
    #  ┌────╔╤╗  intersection = 0.5
    #  └────╚╧╝  union area = 4
    #  ┌─╔═╤══╗  intersection = 1
    #  └─╚═╧══╝  union area = 4
    (
        torch.tensor([
            [0, 0, 3.5, 1],  # area = 3.5
            [0, 2, 2, 3]  # area = 2
        ], dtype=torch.float),
        torch.tensor([
            [3, 0, 4, 1],  # area = 1
            [1, 2, 4, 3]  # area = 3
        ], dtype=torch.float),
        torch.tensor([
            [0.5 / 4, 0.0],  # correct iou
            [0.0, 1 / 4]
        ])
    )
]


@pytest.mark.parametrize(['box1', 'box2', 'correct_iou'], SAMPLE_BOXES)
def test_bounding_box_iou(box1, box2, correct_iou):
    assert torch.allclose(bounding_box_iou(box1, box2), correct_iou)


SIMPLE_CASES = [
    # correct class
    #  ┌───┐
    #  │ ┌─┼─┐
    #  └─┼─┘ │
    #    └───┘
    (
        [
            {  # predictions
                'boxes': torch.tensor([[0, 0, 2, 2]], dtype=torch.float),
                'labels': torch.tensor([1], dtype=torch.long),
                'scores': torch.tensor([0.1], dtype=torch.float),
            }
        ],
        [
            {  # ground truth
                'boxes': torch.tensor([[1, 1, 3, 3]], dtype=torch.float),
                'labels': torch.tensor([1], dtype=torch.long),
            }
        ],
        torch.tensor(1 / 7)  # score
    ),
    # incorrect class
    #  ┌───┐
    #  │ ┌─┼─┐
    #  └─┼─┘ │
    #    └───┘
    (
        [
            {  # predictions
                'boxes': torch.tensor([[0, 0, 2, 2]], dtype=torch.float),
                'labels': torch.tensor([0], dtype=torch.long),
                'scores': torch.tensor([0.2], dtype=torch.float),
            }
        ],
        [
            {  # ground truth
                'boxes': torch.tensor([[1, 1, 3, 3]], dtype=torch.float),
                'labels': torch.tensor([1], dtype=torch.long),
            }
        ],
        torch.tensor(0.0)  # score
    ),
]

COMPLEX_CASES = [
    # mean of two for correct classes
    #  ┌────╔╤╗
    #  └────╚╧╝
    #  ┌─╔═╤══╗
    #  └─╚═╧══╝
    (
        [
            {  # predictions
                'boxes': torch.tensor([
                    [0, 0, 3.5, 1],
                    [0, 2, 2, 3]
                ], dtype=torch.float),
                'labels': torch.tensor([1, 2], dtype=torch.long),
                'scores': torch.tensor([0.5, 0.5], dtype=torch.float),
            }
        ],
        [
            {  # ground truth
                'boxes': torch.tensor([
                    [3, 0, 4, 1],
                    [1, 2, 4, 3]
                ], dtype=torch.float),
                'labels': torch.tensor([1, 2], dtype=torch.long),
            }
        ],
        torch.tensor([
            [0.5 * (0.5 / 4 + 1 / 4)],  # score
        ], dtype=torch.float)
    ),
    # mean of two images for correct classes
    #  ┌────╔╤╗    ┌─╔═╤══╗
    #  └────╚╧╝    └─╚═╧══╝
    (
        [
            {  # predictions
                'boxes': torch.tensor([
                    [0, 0, 3.5, 1],
                ], dtype=torch.float),
                'labels': torch.tensor([1], dtype=torch.long),
                'scores': torch.tensor([0.25], dtype=torch.float),
            },
            {
                'boxes': torch.tensor([
                    [0, 2, 2, 3]
                ], dtype=torch.float),
                'labels': torch.tensor([2], dtype=torch.long),
                'scores': torch.tensor([0.75], dtype=torch.float),
            }
        ],
        [
            {  # ground truth
                'boxes': torch.tensor([
                    [3, 0, 4, 1],
                ], dtype=torch.float),
                'labels': torch.tensor([1], dtype=torch.long),
            },
            {
                'boxes': torch.tensor([
                    [1, 2, 4, 3]
                ], dtype=torch.float),
                'labels': torch.tensor([1], dtype=torch.long),
            }
        ],
        torch.tensor([
            [0.5 * (0.5 / 4 + 1 / 4)],  # score
        ], dtype=torch.float)
    ),
]


@pytest.mark.parametrize(['prediction', 'ground_truth', 'score'], SIMPLE_CASES + COMPLEX_CASES)
def test_mean_correct_bbox_iou_single_update(prediction, ground_truth, score):
    metric = MeanBBoxIoUCorrect()
    # check that metric correctly sets
    assert metric == 0
    # check that it correctly updates
    assert metric(prediction, ground_truth) == score
    # and retains value
    assert metric == score
    # correctly resets
    metric.reset()
    assert metric == 0


@pytest.mark.parametrize('case', [SIMPLE_CASES, COMPLEX_CASES])
def test_mean_correct_bbox_iou_multiple_updates(case):
    metric = MeanBBoxIoUCorrect()
    # check that metric correctly sets
    assert metric == 0
    # being updated several times still computes correctly
    scores = []
    for prediction, ground_truth, score in case:
        metric(prediction, ground_truth)
        scores.append(score)
    # and retains value
    assert metric == sum(scores) / len(scores)
    # correctly resets
    metric.reset()
    assert metric == 0

def test_m():
    m = ExactMAPAtThreshold(num_classes=3, iou_threshold=-0.1, compute_on_step=False)
    for prediction, ground_truth, score in COMPLEX_CASES:
        m(prediction, ground_truth)
    s = m.compute()
    print(s)
    assert s > 0
