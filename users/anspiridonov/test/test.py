import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer


@hydra.main(config_path='.', config_name='conf.yaml')
def main(cfg: DictConfig):
    datamodule = instantiate(cfg.data)
    task = instantiate(cfg.task)
    logger = WandbLogger(**cfg.logger)
    trainer = Trainer(**cfg.trainer, logger=logger)
    trainer.fit(model=task, datamodule=datamodule)


if __name__ == '__main__':
    main()

