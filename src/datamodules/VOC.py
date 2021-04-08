from typing import Any, Union, List, Optional, Tuple

import torch
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from pytorch_lightning import LightningDataModule
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
from src.utils.transforms import make_transforms
from .common import collate_boxes
from torchvision.datasets import VOCDetection
from hydra.utils import to_absolute_path


class VOCDataset(VOCDetection):
    def __init__(
            self,
            consider_classes: Optional[List] = None,
            *args,
            **kwargs
    ):
        super(VOCDataset, self).__init__(*args, **kwargs)
        if consider_classes is None:
            consider_classes = [
                'background',
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor'
            ]
        else:
            if 'background' not in consider_classes:
                consider_classes.insert(0, 'background')
        self.class_to_label = {e: i for i, e in enumerate(consider_classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = imread(self.images[index])
        image = cvtColor(image, COLOR_BGR2RGB)

        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        bboxes = []
        labels = []

        for object in target['annotation']['object']:
            label = object['name']
            if label in self.class_to_label:
                idx = self.class_to_label[label]
                labels.append(idx)
                bbox = tuple(int(object['bndbox'][coord]) for coord in ['xmin', 'ymin', 'xmax', 'ymax'])
                bboxes.append(bbox)

        if not bboxes and not labels:
            width, height = int(target['annotation']['size']['width']), int(target['annotation']['size']['height'])
            bboxes.append((0, 0, width, height))
            labels.append(0)

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        return image, {'boxes': torch.tensor(bboxes), 'labels': torch.tensor(labels)}


class VOC(LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 32,
            num_workers: int = 1,
            train_transforms=None,
            val_transforms=None,
            test_transforms=None,
            classes=None,
            download=False,
            *args, **kwargs
    ):
        self.data_dir = to_absolute_path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args
        self.kwargs = kwargs
        self.classes = classes
        self.download = download
        train_transforms = make_transforms(train_transforms)
        val_transforms = make_transforms(val_transforms)
        test_transforms = make_transforms(test_transforms)
        super(VOC, self).__init__(train_transforms, val_transforms, test_transforms, *args, **kwargs)

    def prepare_data(self, *args, **kwargs):
        VOCDetection(root=self.data_dir, download=self.download, image_set='trainval')

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = VOCDataset(
            root=self.data_dir,
            download=False,
            image_set='train',
            transforms=self.train_transforms,
            consider_classes=self.classes
        )
        self.val_dataset = VOCDataset(
            root=self.data_dir,
            download=False,
            image_set='val',
            transforms=self.val_transforms,
            consider_classes=self.classes
        )

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=collate_boxes, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=collate_boxes, shuffle=False)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        raise AttributeError("No test set for VOC dataset")
