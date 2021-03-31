import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import FasterRCNN


@hydra.main(config_path='.', config_name='conf.yaml')
def main(cfg: DictConfig):
    dm = instantiate(cfg.data)
    dm.setup()
    print(next(iter(dm.train_dataloader())))

if __name__ == '__main__':
    main()
    nn = FasterRCNN()
