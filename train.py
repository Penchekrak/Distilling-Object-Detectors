import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything, Trainer
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger


@hydra.main()
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    logger = WandbLogger(**cfg.logger)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    checkpoint = ModelCheckpoint(**cfg.checkpoint, dirpath=logger.save_dir)
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=checkpoint,
        plugins=DDPPlugin(find_unused_parameters=True)
    )
    task = instantiate(cfg.task)
    datamodule = instantiate(cfg.data)
    trainer.fit(model=task, datamodule=datamodule)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
