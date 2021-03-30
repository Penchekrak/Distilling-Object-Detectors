import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from albumentations.pytorch import ToTensorV2


@hydra.main(config_path='.', config_name='conf.yaml')
def main(cfg: DictConfig):
    dm = instantiate(cfg.data)
    dm.setup()
    print(dm.train_dataset[0])

if __name__ == '__main__':
    main()
