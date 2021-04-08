from wandb import Image
from typing import Dict, List
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

    def __call__(
            self,
            images: Tensor,
            descriptors: Dict[str, List[Dict[str, Tensor]]]
    ):
        if self.max_capacity > len(self.storage):
            take = min(self.max_capacity - len(self.storage), len(images))
            keys, values = descriptors.keys(), descriptors.values()
            for i in range(take):
                self.storage.append((images[i], {k: v[i] for k, v in zip(keys, values)}))

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
                    "maxY": box[3].item(),
                },
                "class_id": class_id.item(),
                "box_caption": f"{self.class_id_to_label[class_id.item()]}",
                "domain": "pixel",
                "scores": {"score": 1.0}
            }
            if score:
                dct["scores"] = {"score": score.item()}
                dct["box_caption"] += f" ({score.item():.3f})"
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
            data_or_path=image,
            boxes={
                key: {
                    'box_data': self.prepare_box_data(value),
                    'class_labels': self.class_id_to_label,
                }
                for key, value in descriptors.items()
            }
        )

    def log_to(
            self,
            logger: Run
    ):
        logger.log({self.name: [self.to_image(image, descriptors) for image, descriptors in self.storage]})
        del self.storage
        self.storage = []
