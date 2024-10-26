from typing import Dict, List, Optional, Literal, Tuple, Callable

from copy import deepcopy
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def view_tensor_as_vector(t: Tensor) -> Tensor:
    return t.view(t.numel())


def cosine_similarity(u: Tensor, v: Tensor) -> float:
    u_vec = view_tensor_as_vector(u)
    v_vec = view_tensor_as_vector(v)
    similarity = torch.dot(u_vec, v_vec) / (u_vec.norm() * v_vec.norm())
    return similarity.item()


def diff_model_weights(weights1, weights2):
    diff = dict()
    assert set(weights1.keys()) == set(weights2.keys())
    for name in weights1.keys():
        diff[name] = weights2[name] - weights1[name]

        # print(name, diff[name].norm().item())
    return diff


def compute_orthogonal_vector(vec, orthogonal_to):
    # does not need to be unit vector, b_new = b - (a^T b) / ||a||^2 * a
    out_vec = (
        vec
        - torch.dot(vec, orthogonal_to) / (orthogonal_to.norm() ** 2) * orthogonal_to
    )
    return out_vec


def view_weights_as_vector(weights):
    weights_vec = [
        view_tensor_as_vector(w) if w.dim() > 1 else w for _, w in weights.items()
    ]
    return torch.cat(weights_vec)


def restore_vector_as_weights(vec: Tensor, template: Dict[str, Tensor]):
    out = dict()
    idx = 0
    for k, v in template.items():
        l = v.numel()
        out[k] = vec[idx : idx + l].view_as(v)
        idx += l

    assert idx == len(vec), f"{idx}, {len(vec)}"
    return out


@torch.no_grad()
def rand_norm_gaussian_like(
    weights: Dict[str, Tensor],
    device: torch.device,
    blacklist: List[str] = None,
    norm_type: Optional[str] = "layer",
    orthogonal_to: Dict[str, Tensor] = None,
) -> Dict[str, Tensor]:
    """
    generate random norm directions
    rmsnorm weights should be ignored in random directions
    if no norm type is specified, returns a unit direction
    """
    if orthogonal_to is not None:
        assert weights.keys() == orthogonal_to.keys()

    norm_dirs = dict()
    for name in weights.keys():
        dir = torch.randn(size=weights[name].size())

        if blacklist is not None and any([op in name.lower() for op in blacklist]):
            # currently, only norm is in blacklist and will be set to 0
            norm_dir = dir.fill_(0).to(device)
        else:
            dir = dir.to(device)
            if orthogonal_to is not None:
                print("similarity", cosine_similarity(dir, orthogonal_to[name]))
            # normalize to unit directions
            norm_dir = dir / torch.norm(dir)

            # scale to the magnitude of the model weights
            if norm_type == "layer":
                norm_dir *= weights[name].norm().item()

        norm_dirs[name] = norm_dir

        # norm should be sqrt(dir.numel)
        print(
            name,
            dir.norm().item(),
            weights[name].norm().item(),
            norm_dir.norm().item(),
            norm_dir.dtype,
        )

    return norm_dirs


def get_random_dirs_step_size(dirs, multiplier: float, steps: int) -> Dict[str, Tensor]:
    """
    new weights = weights + current step * unit dir * distance / steps
    This function computes [unit dir * distance / steps] so that it can be easily added to the original model weights
    Compute the step size for random directions
    """
    # print(distance / steps)
    for name in dirs.keys():
        dirs[name] = dirs[name] * multiplier / steps
    return dirs


def get_interpolation_dirs_step_size(
    weights1: Dict[str, Tensor], weights2: Dict[str, Tensor], steps: int
) -> Dict[str, Tensor]:
    dirs = dict()
    assert set(weights1.keys()) == set(weights2.keys())
    for name in weights1.keys():
        dirs[name] = (weights2[name] - weights1[name]) / steps

        print(name, dirs[name].norm().item())
    return dirs


@torch.no_grad()
def compute_directions(
    weights_list: List[Dict[str, Tensor]],
    ndim: int = 1,
    norm_type: str = None,
    steps: int = None,
    blacklist: List[str] = None,
    multiplier: Optional[List[float]] = None,
    device: torch.device = None,
) -> Tuple[str, Tensor, Optional[Tensor]]:
    """
    Compute directions on device
    Always use the first weights as the center
    the multiplier can be different for two coordinates
    """

    if device is None:
        device = torch.device("cpu")

    # move all weights to the same device
    for weights in weights_list:
        weights = {k: v.to(device) for k, v in weights.items()}

    N = len(weights_list)
    assert 1 <= N <= 3, f"#models should be within [1, 3], but {N} is provided."
    weights = weights_list[0]
    if N == 1:
        vis_type = f"{ndim}D_random"
        assert multiplier is not None and len(multiplier) == ndim
        dirs1 = rand_norm_gaussian_like(
            weights, device=device, blacklist=blacklist, norm_type=norm_type
        )
        dirs1 = get_random_dirs_step_size(dirs1, multiplier[0], steps)
        if ndim == 2:
            # dirs2 = rand_norm_gaussian_orthogonal_to(dirs, device=device)
            dirs2 = rand_norm_gaussian_like(
                weights,
                device=device,
                blacklist=blacklist,
                norm_type=norm_type,
                orthogonal_to=dirs1,
            )
            dirs2 = get_random_dirs_step_size(dirs2, multiplier[1], steps)
    elif N == 2:
        vis_type = f"{ndim}D_interpolation"
        # interpolate one direction
        dirs1 = get_interpolation_dirs_step_size(weights, weights_list[1], steps)
        if ndim == 2:
            vis_type = f"{ndim}D_interpolation-random"
            assert multiplier is not None and len(multiplier) == 1
            dirs2 = rand_norm_gaussian_like(
                weights,
                device=device,
                blacklist=blacklist,
                norm_type=norm_type,
                orthogonal_to=dirs1,
            )
            dirs2 = get_random_dirs_step_size(dirs2, multiplier[0], steps)
    else:
        vis_type = f"{ndim}D_interpolation"
        # interpolatte two directions
        dirs1 = get_interpolation_dirs_step_size(weights, weights_list[1], steps)
        dirs2 = get_interpolation_dirs_step_size(weights, weights_list[2], steps)

    print(f"Generating directions for {vis_type} ...")

    dirs = [dirs1] if ndim == 1 else [dirs1, dirs2]

    return vis_type, dirs
