import torch


def collate_boxes(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    targets = list(targets)
    return images, targets
