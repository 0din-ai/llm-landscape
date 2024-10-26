from typing import Dict, Callable, Tuple, Optional
from copy import deepcopy
import torch
from torch import nn, Tensor
from time import time
import logging

from src.util import printer


def centralize_initial_weights(weights, dirs, steps) -> Dict[str, Tensor]:
    for name in weights.keys():
        weights[name] -= dirs[name] * steps / 2
    return weights


def shift_initial_weights(weights, dirs, steps) -> Dict[str, Tensor]:
    for name in weights.keys():
        weights[name] -= dirs[name] * steps
    return weights


@torch.no_grad()
def forward_1d_directions(
    model: nn.Module,
    weights_origin: Dict[str, Tensor],
    vis_type: str,
    dirs: Dict[str, Tensor],
    steps: int,
    forward: Callable = None,
    postprocess: Callable = None,
    log: Optional[logging.Logger] = None,
    device: torch.device = None,
    additional_steps_for_interpolation: int = 0,
):
    """
    Model inference over all data at each point along the directions.
    """
    # Weight perturbation is computed on CPU
    assert all([v.device.type == "cpu" for _, v in weights_origin.items()])
    weights = deepcopy(weights_origin)
    if "random" in vis_type:
        log.info("Centralizing the original model weights ...")
        weights = centralize_initial_weights(weights, dirs, steps)
    if "interpolation" in vis_type and additional_steps_for_interpolation != 0:
        log.info(
            f"Shifting the original model weights for {additional_steps_for_interpolation} steps ..."
        )
        weights = shift_initial_weights(
            weights, dirs, additional_steps_for_interpolation
        )

    model.load_state_dict(weights)
    out = []
    total_steps = 2 * additional_steps_for_interpolation + steps
    for i in range(total_steps + 1):
        msg = f"{'#'* 10} Step {i + 1}/{total_steps + 1} {'#'* 10}"
        if device is not None:
            msg = printer(device, msg)
        log.info(msg)
        output = forward(model)
        out.append({"step": i, "output": output})

        if i < total_steps:
            # update weights
            for name in weights.keys():
                weights[name] += dirs[name]
            model.load_state_dict(weights)

    # save outputs
    postprocess(out)


@torch.no_grad()
def forward_2d_directions(
    model: nn.Module,
    weights_origin: Dict[str, Tensor],
    vis_type: str,
    dirs1: Dict[str, Tensor],
    dirs2: Dict[str, Tensor],
    steps: int,
    forward: Callable = None,
    postprocess: Callable = None,
    log: Optional[logging.Logger] = None,
    device: torch.device = None,
    additional_steps_for_interpolation: int = 0,
):
    """
    Model inference over all data at each point along the directions.
    """
    # Weight perturbation is computed on CPU
    assert all([v.device.type == "cpu" for _, v in weights_origin.items()])
    weights = deepcopy(weights_origin)
    if vis_type == "2D_random":
        log.info("Centralizing the original model weights for dirs1 & dirs2...")
        weights = centralize_initial_weights(weights, dirs1, steps)
        weights = centralize_initial_weights(weights, dirs2, steps)
    elif vis_type == "2D_interpolation-random":
        log.info(
            f"Centralizing the original model weights for dirs1 & shifting {additional_steps_for_interpolation} for dirs2 ..."
        )
        weights = shift_initial_weights(
            weights, dirs1, additional_steps_for_interpolation
        )
        weights = centralize_initial_weights(weights, dirs2, steps)
    elif vis_type == "2D_interpolation":
        weights = shift_initial_weights(
            weights, dirs1, additional_steps_for_interpolation
        )
        weights = shift_initial_weights(
            weights, dirs2, additional_steps_for_interpolation
        )
    else:
        raise ValueError

    model.load_state_dict(weights)

    out = []
    curr_step = 0
    if vis_type == "2D_random":
        steps_dirs1 = steps
    else:
        steps_dirs1 = additional_steps_for_interpolation * 2 + steps

    if vis_type == "2D_random" or vis_type == "2D_interpolation-random":
        steps_dirs2 = steps
    else:
        steps_dirs2 = additional_steps_for_interpolation * 2 + steps
    total_steps = (steps_dirs1 + 1) * (steps_dirs2 + 1)
    for i in range(steps_dirs1 + 1):
        output_row = []
        for j in range(steps_dirs2 + 1):
            msg = f"{'#'* 10} Step {curr_step + 1}/{total_steps} {'#'* 10}"
            if device is not None:
                msg = printer(device, msg)
            log.info(msg)
            if i % 2 == 0:
                output = forward(model)
                if j < steps_dirs2:
                    for name in weights.keys():
                        weights[name] += dirs2[name]
                    model.load_state_dict(weights)
            else:
                output = forward(model)
                if j < steps_dirs2:
                    for name in weights.keys():
                        weights[name] -= dirs2[name]
                    model.load_state_dict(weights)

            output_row.append(output)
            curr_step += 1
        if i % 2 == 0:
            out += [
                {"step": idx, "output": re}
                for idx, re in zip(
                    range(i * (steps_dirs2 + 1), (i + 1) * (steps_dirs2 + 1)),
                    output_row,
                )
            ]
        else:
            out += [
                {"step": idx, "output": re}
                for idx, re in zip(
                    range(i * (steps_dirs2 + 1), (i + 1) * (steps_dirs2 + 1)),
                    output_row[::-1],
                )
            ]

        if i < steps_dirs1:
            for name in weights.keys():
                weights[name] += dirs1[name]
            model.load_state_dict(weights)

    assert curr_step == total_steps, f"{curr_step} - {total_steps}"

    # save outputs
    postprocess(out)
