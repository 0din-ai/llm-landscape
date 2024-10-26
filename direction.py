from typing import Union, List
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import logging


from src.llm import load_model
from src.landscape import compute_directions, save_dirs

device = torch.device("cuda:0")

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="compute_dirs", version_base="1.3")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    ndim = cfg.landscape.ndim

    # Step 1: load all model weights
    weights_list = []
    for model_dir in cfg.model_dir_list:
        model = load_model(name=cfg.model.name, model_dir=model_dir, torch_dtype=torch.float32, log=log)
        weights_list.append(model.state_dict())

    # Step 2: compute directions
    vis_type, dirs = compute_directions(
        weights_list=weights_list,
        ndim=ndim,
        norm_type=cfg.landscape.norm_type,
        steps=cfg.landscape.steps,
        blacklist=cfg.landscape.blacklist,
        multiplier=cfg.landscape.multiplier,
        device=device,
    )
    assert (
        vis_type == cfg.landscape.vis_type
    ), f"Direction type mismatch: {vis_type} and {cfg.landscape.vis_type}"

    # Step 3: save dirs to disk
    dirs_path = (
        Path(cfg.landscape.path_to_dirs)
        if cfg.landscape.path_to_dirs
        else Path(vis_type) / cfg.model.name
    )
    log.info(f"Saving {ndim} direction(s) of {vis_type} to {dirs_path.resolve()} ...")
    save_dirs(ndim=ndim, dirs_list=dirs, save_to_path=dirs_path)

    print("Directions computed and saved.")


if __name__ == "__main__":
    main()

