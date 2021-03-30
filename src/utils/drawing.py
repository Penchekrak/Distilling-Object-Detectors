from wandb import Image
from typing import Dict
from torch import Tensor
from itertools import zip_longest
from wandb.wandb_run import Run


class WandbImageLogger(object):
    def __init__(
            self,
            class_id_to_labels: Dict[int, str],
            max_capacity: int = 10,
            name: str = 'Detections',
    ):
        self.max_capacity = max_capacity
        self.storage = []
        self.class_id_to_label = class_id_to_labels
        self.name = name

    def __call__(self, images, descriptors):
        if self.max_capacity > len(self.storage):
            for _, image, descriptor in zip(range(self.max_capacity - len(self.storage)), images, descriptors):
                self.storage.append((image, descriptor))

    def prepare_box_data(
            self,
            output: Dict[str, Tensor],
    ):
        box_data = []
        for box, class_id, score in zip_longest(output['boxes'], output['labels'], output.get('scores', [])):
            dct = {
                "position": {
                    "minX": box[0].item(),
                    "maxX": box[2].item(),
                    "minY": box[1].item(),
                    "maxY": box[4].item(),
                },
                "class_id": class_id.item(),
                "box_caption": f"{self.class_id_to_label[class_id]}",
                "domain": "pixel"
            }
            if score:
                dct["scores"] = {"score": score.item()}
                dct["box_caption"] += f" ({score:.3f})"
            box_data.append(
                dct
            )
        return box_data

    def to_image(
            self,
            image: Tensor,
            descriptors: Dict[str, Dict[str, Tensor]],
    ):
        return Image(
            data_or_path=image.detach().cpu().numpy(),
            boxes={
                key: {
                    'box_data': self.prepare_box_data(value),
                    'class_labels': self.class_id_to_label,
                }
                for key, value in descriptors
            }
        )

    def log_to(
            self,
            logger: Run
    ):
        logger.log({self.name: [self.to_image(image, descriptors) for image, descriptors in self.storage]})
