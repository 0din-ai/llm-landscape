import logging
import os
from copy import deepcopy
from functools import partial
from pathlib import Path

import hydra
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.distributed
import torch.multiprocessing as mp
from hydra import compose, initialize, initialize_config_dir, initialize_config_module
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from src.landscape import grid_coords, safety_avg, safety_region
from src.metrics import measure_keywords_ASR
from src.util import merge_output_from_devices

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="landscape", version_base="1.3")
def main(cfg: DictConfig):
    working_dir = Path(cfg.landscape.vis_type) / cfg.model.name
    print(working_dir.absolute())
    merge_output_from_devices(
        file_dir=working_dir, delete_tmp=False
    )

    steps = cfg.landscape.steps

    output_file = working_dir / "output.jsonl"
    with jsonlines.open(output_file, "r") as f:
        output = list(f)

    # for each step - gather all the response from all devices
    step_eval_val = []
    for i in range(steps + 1):
        step_output = output[i]
        assert step_output["step"] == i
        step_eval_val.append(
            measure_keywords_ASR(
                generations=[i["generation"] for i in step_output["output"]]
            )
        )

    coords_x, coords_y = grid_coords(
        steps=cfg.landscape.steps,
        ndim=cfg.landscape.ndim,
        vis_type=cfg.landscape.vis_type,
        multiplier=cfg.landscape.multiplier,
    )
    fig_name = f"{cfg.landscape.vis_type}_{cfg.model.name}_landscape.png"
    fig = px.line(
        x=coords_x,
        y=step_eval_val,
        title=f"{cfg.landscape.vis_type} ASR vs Distance",
    )
    fig.write_image(fig_name, scale=2)

    log.info(f"VISAGE: {safety_avg(coords_x, np.array(step_eval_val))}")


if __name__ == "__main__":
    main()
