import logging
import os
from copy import deepcopy
from functools import partial
from pathlib import Path

import hydra
import torch
import torch.distributed
import torch.multiprocessing as mp
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.distributed import barrier, destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.dataset import AdvBench, dataloader_prompt
from src.landscape import forward_1d_directions, forward_2d_directions, load_dirs
from src.llm import forward_llama, forward_llama_logits, load_model, load_tokenizer
from src.util import merge_output_from_devices, printer, save_response

log = logging.getLogger(__name__)

_FORWARD_LLM = {
    "llama2": forward_llama,
    "vicuna": forward_llama,
    "mistral": forward_llama,
    "llama3": forward_llama,
    "llama2-mmlu": forward_llama_logits,
}


@hydra.main(config_path="config", config_name="landscape", version_base="1.3")
def main(cfg: DictConfig):
    ddp_setup()
    device = int(os.environ["LOCAL_RANK"])

    if device == 0:
        log.info(OmegaConf.to_yaml(cfg, resolve=True))

    # important for ddp to have deterministic results
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    ndim = cfg.landscape.ndim
    working_dir = Path(cfg.landscape.vis_type) / cfg.model.name

    # Step 1: load directions
    log.info(
        printer(
            device,
            f"Loading direction(s) of {cfg.landscape.vis_type} from {working_dir.resolve()} ...",
        )
    )
    dirs_list = load_dirs(working_dir)

    # Step 2: load models, tokenizers
    model = load_model(
        name=cfg.model.name,
        model_dir=cfg.model_path,
        torch_dtype=torch.float32,
        log=log,
    )
    model.eval()
    weights_origin = deepcopy(model.state_dict())
    model.to(device)
    # model = DDP(model, device_ids=[device])

    tokenizer = load_tokenizer(
        name=cfg.model.name,
        tokenizer_dir=cfg.tokenizer_dir,
        padding_side="left",
        log=log,
    )
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # Step 3: load dataset
    log.info(
        printer(device, f"Loading dataset from {cfg.dataset.test_dataset._target_} ...")
    )
    dataset = instantiate(
        cfg.dataset.test_dataset,
        tokenizer=tokenizer,
        max_seq_len=model.config.max_position_embeddings,
    )
    dataloader = dataloader_prompt(
        dataset,
        tokenizer,
        bs=cfg.dataset.batch_size,
        sampler=DistributedSampler(dataset),
    )

    # Step 4: evaluate and save
    # TODO: if ddp is used, need to change the filename to include the device id
    if ndim == 1:
        forward_1d_directions(
            model=model,
            weights_origin=weights_origin,
            vis_type=cfg.landscape.vis_type,
            dirs=dirs_list[0],
            steps=cfg.landscape.steps,
            additional_steps_for_interpolation=cfg.landscape.additional_steps_for_interpolation,
            forward=lambda m: _FORWARD_LLM[cfg.model.name](
                model=m,
                dataloader=dataloader,
                tokenizer=tokenizer,
                device=device,
                **cfg.model.kwargs,
            ),
            postprocess=partial(
                save_response,
                save_to=(
                    cfg.landscape.save_output_to
                    if cfg.landscape.save_output_to
                    else working_dir / "tmp"
                ),
                filename=f"device{device}.jsonl",
                log=log,
            ),
            device=device,
            log=log,
        )
    elif ndim == 2:
        forward_2d_directions(
            model=model,
            weights_origin=weights_origin,
            vis_type=cfg.landscape.vis_type,
            dirs1=dirs_list[0],
            dirs2=dirs_list[1],
            steps=cfg.landscape.steps,
            additional_steps_for_interpolation=cfg.landscape.additional_steps_for_interpolation,
            forward=lambda m: _FORWARD_LLM[cfg.model.name](
                model=m,
                dataloader=dataloader,
                tokenizer=tokenizer,
                device=device,
                **cfg.model.kwargs,
            ),
            postprocess=partial(
                save_response,
                save_to=(
                    cfg.landscape.save_output_to
                    if cfg.landscape.save_output_to
                    else working_dir / "tmp"
                ),
                filename=f"device{device}.jsonl",
                log=log,
            ),
            device=device,
            log=log,
        )

    # torch.distributed.barrier()
    # if device == 0:
    #     merge_output_from_devices(file_dir=working_dir, delete_tmp=True)

    destroy_process_group()


def ddp_setup():
    init_process_group(backend="nccl")


if __name__ == "__main__":
    main()
