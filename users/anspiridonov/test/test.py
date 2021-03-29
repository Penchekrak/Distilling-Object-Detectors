import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

@hydra.main(config_path='.', config_name='conf.yaml')
def main(cfg: DictConfig):
    dm = instantiate(cfg.data)
    print(dm)

if __name__ == '__main__':
    main()
