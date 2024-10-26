import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from scipy.integrate import simpson
from torch import Tensor


def save_dirs(
    ndim: int,
    dirs_list: List[Dict[str, Tensor]],
    save_to_path: Union[str, Path],
    log: Optional[logging.Logger] = None,
):
    """
    save all dirs into one file
    may not work for large models, we will shard it if that's the case
    """
    assert ndim in [1, 2]
    Path(save_to_path).mkdir(parents=True, exist_ok=True)
    for idx in range(ndim):
        torch.save(dirs_list[idx], save_to_path / f"dirs{idx+1}.pt")


def load_dirs(
    path_to_dirs: Union[Path, str] = None,
):
    """Load pre-computed directions to CPU."""
    dirs_list = []
    for dirs_file in Path(path_to_dirs).glob("*.pt"):
        dirs_list.append(torch.load(dirs_file, map_location="cpu"))
    return dirs_list


def grid_coords(
    steps: int,
    ndim: int,
    vis_type: str,
    multiplier: Optional[List[float]],
    additional_steps_for_interpolation: int = 0,
):
    """
    Generate coordinates for landscape
    """
    if ndim == 1:
        if vis_type == "1D_random":
            assert multiplier is not None and len(multiplier) == 1
            coords_x = np.linspace(
                start=-multiplier[0] / 2,
                stop=multiplier[0] / 2,
                num=steps + 1,
                endpoint=True,
                retstep=False,
            )
        elif vis_type == "1D_interpolation":
            step_size = 1.0 / steps
            coords_x = np.linspace(
                start=-step_size * additional_steps_for_interpolation,
                stop=1 + step_size * additional_steps_for_interpolation,
                num=steps + 2 * additional_steps_for_interpolation + 1,
                endpoint=True,
                retstep=False,
            )
        else:
            raise ValueError(f"{vis_type} not supported")
    elif ndim == 2:
        if vis_type == "2D_interpolation-random":
            step_size = 1.0 / steps
            coords_x = np.linspace(
                start=-step_size * additional_steps_for_interpolation,
                stop=1 + step_size * additional_steps_for_interpolation,
                num=steps + 2 * additional_steps_for_interpolation + 1,
                endpoint=True,
                retstep=False,
            )
            coords_y = np.linspace(
                start=-multiplier[0] / 2,
                stop=multiplier[0] / 2,
                num=steps + 1,
                endpoint=True,
                retstep=False,
            )
        elif vis_type == "2D_random":
            coords_x = np.linspace(
                start=-multiplier[0] / 2,
                stop=multiplier[0] / 2,
                num=steps + 1,
                endpoint=True,
                retstep=False,
            )
            coords_y = np.linspace(
                start=-multiplier[0] / 2,
                stop=multiplier[0] / 2,
                num=steps + 1,
                endpoint=True,
                retstep=False,
            )
        elif vis_type == "2D_interpolation":
            step_size = 1.0 / steps
            coords_x = np.linspace(
                start=-step_size * additional_steps_for_interpolation,
                stop=1 + step_size * additional_steps_for_interpolation,
                num=steps + 2 * additional_steps_for_interpolation + 1,
                endpoint=True,
                retstep=False,
            )
            coords_y = np.linspace(
                start=-step_size * additional_steps_for_interpolation,
                stop=1 + step_size * additional_steps_for_interpolation,
                num=steps + 2 * additional_steps_for_interpolation + 1,
                endpoint=True,
                retstep=False,
            )

    return (coords_x, coords_y if ndim == 2 else None)


def safety_region(coords, vals):
    delta = (coords.max() - coords.min()) / (len(coords) - 1)
    vals = 100 - vals
    area1 = np.trapz(vals, dx=delta)
    print("trapz", area1)

    # area = simpson(vals, dx=delta)
    # print("simpson", area)

    return area1


def safety_avg(coords, vals):
    N = len(vals)
    i = 1
    start = 0
    while i < N // 2:
        if vals[i] <= 99.99:
            if vals[i - 1] > 99.99:
                start = i
                break
        i = i + 1

    i = N - 2
    end = N - 1
    while i > N // 2:
        if vals[i] <= 99.99:
            if vals[i + 1] > 99.99:
                end = i
                break
        i = i - 1
    arr = vals[start : end + 1]
    r_val = 100 - arr
    out = np.mean(r_val)

    return out
