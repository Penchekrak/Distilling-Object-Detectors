from pytorch_lightning import LightningDataModule

class COCO(LightningDataModule):
    def __init__(self, num_workers=4):
        super(COCO).__init__()
        self.nw = num_workers
